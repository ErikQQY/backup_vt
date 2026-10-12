#!/usr/bin/env python3
"""Run the bundled legacy VortexTransformer on a NetCDF velocity field.

The checkpoint does not consume a raw ``[u, v]`` image.  Its companion
TorchScript module consumes pathline features shaped ``[B, 16, 256, 7]`` and
returns one vortex-membership probability for each of the 256 seed points.

This entry point follows the layout observed in the bundled
``demoValidationDataset``:

* every inference window is a 32-by-32 NetCDF grid patch;
* 256 pathlines start on a row-major 16-by-16 regular seed grid;
* positions and velocities are expressed in patch-local ``[-1, 1]`` units;
* time remains in the NetCDF's physical coordinate;
* a pathline may leave its seed patch and is padded only after it leaves the
  full flow domain.

The repository does not contain the complete historical raw-NetCDF adapter.
In particular, the old demo's observer-transformed velocity/IVD values cannot
be reproduced bit-for-bit from ``cylinder2d.nc`` with a simple affine scaling.
This script therefore provides a documented, testable best-effort adapter; it
does not claim scientific calibration on a new flow without held-out labels.

The model predicts a vortex region, not an explicit vortex center.  Reported
locations are therefore probability-weighted centroids of thresholded,
connected prediction regions.

Example
-------
MPLCONFIGDIR=/tmp/pyflowvis-mpl .venv/bin/python \
    infer_vortex_transformer.py --time-index 1000 --device cpu
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

import detect_cylinder_vortices as common


PATHLINE_LENGTH = 16
SEED_GRID_SIZE = 16
SEEDS_PER_PATCH = SEED_GRID_SIZE * SEED_GRID_SIZE
FEATURE_COUNT = 7
PADDING_VALUE = -1000.0


@dataclass(frozen=True)
class GridTile:
    """One physical patch whose local coordinate system is ``[-1, 1]^2``."""

    index: int
    x_start: int
    y_start: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @property
    def x_scale(self) -> float:
        return 2.0 / (self.xmax - self.xmin)

    @property
    def y_scale(self) -> float:
        return 2.0 / (self.ymax - self.ymin)


def _grid_starts(axis_size: int, patch_points: int, stride_points: int) -> list[int]:
    if patch_points < 2:
        raise ValueError("--patch-points must be at least 2.")
    if patch_points > axis_size:
        raise ValueError(
            f"A {patch_points}-point patch does not fit an axis of size {axis_size}."
        )
    if stride_points < 1:
        raise ValueError("--patch-stride-points must be positive.")

    final_start = axis_size - patch_points
    starts = list(range(0, final_start + 1, stride_points))
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return starts


def make_grid_tiles(
    x: np.ndarray,
    y: np.ndarray,
    patch_points: int,
    stride_points: int,
) -> list[GridTile]:
    """Tile coordinate arrays in the same grid-index convention as validation."""

    x_starts = _grid_starts(len(x), patch_points, stride_points)
    y_starts = _grid_starts(len(y), patch_points, stride_points)
    tiles: list[GridTile] = []
    for y_start in y_starts:
        for x_start in x_starts:
            tiles.append(
                GridTile(
                    index=len(tiles),
                    x_start=x_start,
                    y_start=y_start,
                    xmin=float(x[x_start]),
                    xmax=float(x[x_start + patch_points - 1]),
                    ymin=float(y[y_start]),
                    ymax=float(y[y_start + patch_points - 1]),
                )
            )
    return tiles


def regular_seeds(tile: GridTile) -> np.ndarray:
    """Return the row-major 16-by-16 t0 seed layout used by the checkpoint."""

    seed_x, seed_y = np.meshgrid(
        np.linspace(tile.xmin, tile.xmax, SEED_GRID_SIZE),
        np.linspace(tile.ymin, tile.ymax, SEED_GRID_SIZE),
    )
    seeds = np.column_stack([seed_x.ravel(), seed_y.ravel()])
    if seeds.shape != (SEEDS_PER_PATCH, 2):
        raise AssertionError(f"Unexpected seed shape {seeds.shape}.")
    return seeds


def _inside_full_domain(points: np.ndarray, flow: common.FlowWindow) -> np.ndarray:
    """Match the legacy integrator's strict spatial-domain test."""

    return (
        (points[:, 0] > flow.x[0])
        & (points[:, 0] < flow.x[-1])
        & (points[:, 1] > flow.y[0])
        & (points[:, 1] < flow.y[-1])
    )


def integrate_tile_pathlines(
    flow: common.FlowWindow,
    tile: GridTile,
    start_time: float,
    duration: float,
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one checkpoint input with shape ``[16, 256, 7]``.

    Feature order is ``[x, y, t, IVD, distance, u, v]``.  The returned mask is
    only for output projection: obstacle seeds still participate in the model
    input so the tensor retains the regular training layout.
    """

    seeds = regular_seeds(tile)
    positions = seeds.copy()
    normalized_seeds = np.column_stack(
        [
            (seeds[:, 0] - tile.xmin) * tile.x_scale - 1.0,
            (seeds[:, 1] - tile.ymin) * tile.y_scale - 1.0,
        ]
    )
    times = np.linspace(start_time, start_time + duration, PATHLINE_LENGTH)
    features = np.full(
        (PATHLINE_LENGTH, SEEDS_PER_PATCH, FEATURE_COUNT),
        PADDING_VALUE,
        dtype=np.float32,
    )

    # Boundary seeds retain their t0 sample but do not advance, matching the
    # legacy C++ integrator.  Interior trajectories may cross the seed tile and
    # remain valid until they leave the complete NetCDF domain.
    active = _inside_full_domain(positions, flow)
    global_mean_curl = flow.curl.mean(axis=(1, 2), dtype=np.float64)

    for step, time in enumerate(times):
        ids = np.arange(SEEDS_PER_PATCH) if step == 0 else np.flatnonzero(active)
        if ids.size:
            current = positions[ids]
            velocity = flow.velocity(current, float(time))
            normalized = np.column_stack(
                [
                    (current[:, 0] - tile.xmin) * tile.x_scale - 1.0,
                    (current[:, 1] - tile.ymin) * tile.y_scale - 1.0,
                ]
            )
            vorticity = flow._sample(flow.curl, current, float(time))
            mean_vorticity = float(
                np.interp(float(time), flow.times, global_mean_curl)
            )

            features[step, ids, 0:2] = normalized
            features[step, ids, 2] = float(time)
            features[step, ids, 3] = np.abs(vorticity - mean_vorticity)
            features[step, ids, 4] = np.linalg.norm(
                normalized - normalized_seeds[ids], axis=1
            )
            features[step, ids, 5] = velocity[:, 0] * tile.x_scale
            features[step, ids, 6] = velocity[:, 1] * tile.y_scale

        if step == PATHLINE_LENGTH - 1:
            continue

        ids = np.flatnonzero(active)
        if not ids.size:
            continue
        time = float(time)
        dt = float(times[step + 1] - times[step])
        point = positions[ids]
        k1 = flow.velocity(point, time)
        k2 = flow.velocity(point + 0.5 * dt * k1, time + 0.5 * dt)
        k3 = flow.velocity(point + 0.5 * dt * k2, time + 0.5 * dt)
        k4 = flow.velocity(point + dt * k3, time + dt)
        new_point = point + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        accepted = np.all(np.isfinite(new_point), axis=1)
        accepted &= _inside_full_domain(new_point, flow)
        positions[ids[accepted]] = new_point[accepted]
        active[ids[~accepted]] = False

    # During training this t0 slot temporarily held the binary seed label.  The
    # dataset loader removed it before forwarding the tensor.
    features[0, :, 4] = 0.0

    output_valid = np.ones(SEEDS_PER_PATCH, dtype=bool)
    if obstacle_radius > 0.0:
        center = np.asarray(obstacle_center, dtype=np.float64)
        output_valid &= (
            np.linalg.norm(seeds - center[None, :], axis=1) > obstacle_radius
        )
    return features, seeds, output_valid


def predict_tiles(
    model,
    features: Sequence[np.ndarray],
    device: str,
    batch_size: int,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for begin in range(0, len(features), batch_size):
        batch = np.stack(features[begin : begin + batch_size])
        tensor = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
        expected = (PATHLINE_LENGTH, SEEDS_PER_PATCH, FEATURE_COUNT)
        if tuple(tensor.shape[1:]) != expected:
            raise ValueError(
                f"Expected model input [B,{expected[0]},{expected[1]},"
                f"{expected[2]}], got {tuple(tensor.shape)}."
            )
        with torch.inference_mode():
            output = model(tensor)
        if tuple(output.shape) != (len(batch), SEEDS_PER_PATCH):
            raise RuntimeError(
                "The companion TorchScript model returned an unexpected shape: "
                f"{tuple(output.shape)}."
            )
        predictions.append(output.detach().cpu().numpy())
    return np.concatenate(predictions, axis=0)


def probability_map(
    x: np.ndarray,
    y: np.ndarray,
    points: np.ndarray,
    scores: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize sparse predictions with nearest-cell Gaussian averaging."""

    def nearest_indices(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
        right = np.clip(np.searchsorted(axis, values), 0, len(axis) - 1)
        left = np.clip(right - 1, 0, len(axis) - 1)
        choose_left = np.abs(values - axis[left]) <= np.abs(axis[right] - values)
        return np.where(choose_left, left, right)

    weighted = np.zeros((len(y), len(x)), dtype=np.float64)
    weights = np.zeros_like(weighted)
    xi = nearest_indices(x, points[:, 0])
    yi = nearest_indices(y, points[:, 1])
    np.add.at(weighted, (yi, xi), scores)
    np.add.at(weights, (yi, xi), 1.0)
    smooth_weighted = gaussian_filter(weighted, sigma=sigma, mode="nearest")
    smooth_weights = gaussian_filter(weights, sigma=sigma, mode="nearest")
    probability = np.divide(
        smooth_weighted,
        smooth_weights,
        out=np.full_like(smooth_weighted, np.nan),
        where=smooth_weights > 1e-5,
    )
    return probability, smooth_weights


def save_detection_csv(
    output_path: Path,
    detections: Sequence[tuple[float, float, float, int]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "x", "y", "peak_probability", "component_pixels"])
        for detection_id, (x, y, peak, pixels) in enumerate(detections, 1):
            writer.writerow(
                [
                    detection_id,
                    f"{x:.8g}",
                    f"{y:.8g}",
                    f"{peak:.8g}",
                    int(pixels),
                ]
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run best_checkpoint.pth.tar on a NetCDF flow and plot predicted "
            "vortex-region seeds plus approximate connected-region centers."
        )
    )
    parser.add_argument("--input", type=Path, default=Path("dataset/cylinder2d.nc"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("trainedVortexTransformer/best_checkpoint.pth.tar"),
    )
    parser.add_argument(
        "--torchscript",
        type=Path,
        default=Path("trainedVortexTransformer/BackUp_best_checkpoint.pthtraced.pt"),
        help="Deterministic architecture/forward paired with the checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/vortex_transformer_inference.png"),
    )
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=Path("outputs/vortex_transformer_seed_scores.csv"),
    )
    parser.add_argument(
        "--detections-output",
        type=Path,
        default=Path("outputs/vortex_transformer_locations.csv"),
    )
    parser.add_argument("--time-index", type=int, default=1000)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument(
        "--patch-points",
        type=int,
        default=32,
        help="NetCDF grid points per patch axis; 32 matches bundled validation.",
    )
    parser.add_argument(
        "--patch-stride-points",
        type=int,
        default=32,
        help="Patch stride in NetCDF grid points; 32 matches bundled validation.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    parser.add_argument(
        "--min-component-pixels",
        type=int,
        default=50,
        help="Reject small thresholded blobs; tune this when grid resolution changes.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto"
    )
    parser.add_argument("--obstacle-x", type=float, default=0.0)
    parser.add_argument("--obstacle-y", type=float, default=0.0)
    parser.add_argument(
        "--obstacle-radius",
        type=float,
        default=None,
        help="Default: use the NetCDF radius variable when present.",
    )
    parser.add_argument("--ivd-percentile", type=float, default=95.0)
    parser.add_argument("--ivd-smooth-sigma", type=float, default=1.0)
    parser.add_argument("--ivd-min-component-pixels", type=int, default=8)
    parser.add_argument("--ivd-obstacle-margin", type=float, default=1.5)
    parser.add_argument("--ivd-border-cells", type=int, default=2)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if args.smooth_sigma < 0.0:
        raise ValueError("--smooth-sigma cannot be negative.")
    if args.min_component_pixels < 1:
        raise ValueError("--min-component-pixels must be positive.")
    if not 0.0 < args.ivd_percentile < 100.0:
        raise ValueError("--ivd-percentile must be between 0 and 100.")


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)

    flow, start_time, snapshot_u, snapshot_v, constants = common.load_flow_window(
        args.input, args.time_index, args.duration
    )
    tiles = make_grid_tiles(
        flow.x, flow.y, args.patch_points, args.patch_stride_points
    )
    obstacle_center = (args.obstacle_x, args.obstacle_y)
    obstacle_radius = (
        constants.get("radius", 0.0)
        if args.obstacle_radius is None
        else args.obstacle_radius
    )

    ivd, ivd_threshold, mean_vorticity, ivd_detections = (
        common.detect_ivd_vortices(
            flow,
            obstacle_center,
            obstacle_radius,
            threshold=None,
            percentile=args.ivd_percentile,
            smooth_sigma=args.ivd_smooth_sigma,
            min_pixels=args.ivd_min_component_pixels,
            obstacle_margin=args.ivd_obstacle_margin,
            border_cells=args.ivd_border_cells,
        )
    )

    print(
        f"Reading {args.input}: grid={len(flow.x)}x{len(flow.y)}, "
        f"t=[{start_time:g}, {start_time + args.duration:g}]"
    )
    print(
        "Adapter note: seed/patch/time layout follows the bundled legacy demo; "
        "historical observer-space u/v and IVD preprocessing is unavailable."
    )
    print(
        f"Building {len(tiles)} patches; model input per patch is "
        f"[{PATHLINE_LENGTH}, {SEEDS_PER_PATCH}, {FEATURE_COUNT}]"
    )
    feature_blocks: list[np.ndarray] = []
    seed_blocks: list[np.ndarray] = []
    valid_blocks: list[np.ndarray] = []
    for tile in tiles:
        features, seeds, valid = integrate_tile_pathlines(
            flow,
            tile,
            start_time,
            args.duration,
            obstacle_center,
            obstacle_radius,
        )
        feature_blocks.append(features)
        seed_blocks.append(seeds)
        valid_blocks.append(valid)

    device = common.choose_device(args.device)
    model, epoch = common.load_model(args.checkpoint, args.torchscript, device)
    print(f"Loaded checkpoint epoch {epoch} on {device}; state_dict matched strictly.")
    score_blocks = predict_tiles(model, feature_blocks, device, args.batch_size)
    points, scores, tile_ids = common.merge_duplicate_seeds(
        seed_blocks, score_blocks, valid_blocks
    )
    dense_probability, support = probability_map(
        flow.x, flow.y, points, scores, args.smooth_sigma
    )
    if obstacle_radius > 0.0:
        grid_x, grid_y = np.meshgrid(flow.x, flow.y)
        inside_obstacle = np.hypot(
            grid_x - obstacle_center[0], grid_y - obstacle_center[1]
        ) <= obstacle_radius
        dense_probability[inside_obstacle] = np.nan
        support[inside_obstacle] = 0.0
    detections = common.component_centers(
        dense_probability,
        support,
        flow.x,
        flow.y,
        args.threshold,
        args.min_component_pixels,
    )
    detections.sort(key=lambda item: (item[0], item[1]))

    common.save_scores_csv(
        args.scores_output, points, scores, tile_ids, args.threshold
    )
    save_detection_csv(args.detections_output, detections)
    common.plot_detection(
        args.output,
        flow,
        snapshot_u,
        snapshot_v,
        ivd,
        ivd_threshold,
        ivd_detections,
        dense_probability,
        points,
        scores,
        detections,
        args.threshold,
        start_time,
        obstacle_center,
        obstacle_radius,
    )

    positive_count = int(np.count_nonzero(scores >= args.threshold))
    print(
        f"Model-positive seeds: {positive_count}/{len(scores)} at "
        f"p >= {args.threshold:g}; approximate connected regions: "
        f"{len(detections)}"
    )
    for detection_id, (x, y, peak, pixels) in enumerate(detections, 1):
        print(
            f"  M{detection_id}: x={x:.5f}, y={y:.5f}, "
            f"peak={peak:.3f}, pixels={pixels}"
        )
    print(
        f"IVD reference: mean vorticity={mean_vorticity:.6g}, "
        f"threshold={ivd_threshold:.6g}, regions={len(ivd_detections)}"
    )
    print(f"Visualization: {args.output}")
    print(f"Per-seed scores: {args.scores_output}")
    print(f"Approximate locations: {args.detections_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
