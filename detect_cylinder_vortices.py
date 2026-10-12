#!/usr/bin/env python3
"""Run the pretrained VortexTransformer on ``dataset/cylinder2d.nc``.

The legacy checkpoint was trained on 16-step, 256-pathline samples.  This
script slides square patches over the long cylinder domain, integrates the
same four-arm cross pathlines used for training, builds the seven model
features, and maps the per-pathline probabilities back to the physical flow.

Example
-------
MPLCONFIGDIR=/tmp/pyflowvis-mpl .venv/bin/python \
    detect_cylinder_vortices.py --device cpu
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
import netCDF4 as nc
import numpy as np
import torch

# The inference command is normally run headlessly (CI, SSH, or Codex shell).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from scipy.ndimage import gaussian_filter, label


PATHLINE_LENGTH = 16
GROUPS_PER_PATCH = 64
LINES_PER_GROUP = 4
LINES_PER_PATCH = GROUPS_PER_PATCH * LINES_PER_GROUP
FEATURE_COUNT = 7
PADDING_VALUE = -1000.0


@dataclass(frozen=True)
class Tile:
    """A square physical-domain patch mapped to the training domain [-1, 1]^2."""

    index: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @property
    def size(self) -> float:
        return self.xmax - self.xmin


@dataclass(frozen=True)
class IVDDetection:
    """One connected high-IVD region."""

    x: float
    y: float
    peak: float
    pixels: int
    rotation_sign: int


class FlowWindow:
    """Velocity and IVD interpolation over a small NetCDF time window."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        times: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
    ) -> None:
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.times = np.asarray(times, dtype=np.float64)
        self.u = np.asarray(u, dtype=np.float32)
        self.v = np.asarray(v, dtype=np.float32)

        if self.u.shape != (len(self.times), len(self.y), len(self.x)):
            raise ValueError(
                "Velocity arrays must have shape [T,Y,X]; got "
                f"{self.u.shape} for axes "
                f"({len(self.times)}, {len(self.y)}, {len(self.x)})."
            )
        if not (
            np.all(np.diff(self.x) > 0)
            and np.all(np.diff(self.y) > 0)
            and np.all(np.diff(self.times) > 0)
        ):
            raise ValueError("x, y and time coordinates must be strictly increasing.")

        # omega = dv/dx - du/dy, matching FLowUtils/vortexCriteria.py.
        dvdx = np.gradient(self.v, self.x, axis=2, edge_order=2)
        dudy = np.gradient(self.u, self.y, axis=1, edge_order=2)
        self.curl = np.asarray(dvdx - dudy, dtype=np.float32)

    @staticmethod
    def _bracket(axis: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, ...]:
        high = np.searchsorted(axis, values, side="right")
        high = np.clip(high, 1, len(axis) - 1)
        low = high - 1
        weight = (values - axis[low]) / (axis[high] - axis[low])
        return low, high, np.clip(weight, 0.0, 1.0)

    def _sample(self, values: np.ndarray, points: np.ndarray, time: float) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        x0, x1, wx = self._bracket(self.x, points[:, 0])
        y0, y1, wy = self._bracket(self.y, points[:, 1])
        t0, t1, wt_array = self._bracket(
            self.times, np.asarray([time], dtype=np.float64)
        )
        t0_i, t1_i, wt = int(t0[0]), int(t1[0]), float(wt_array[0])

        def bilinear(frame: np.ndarray) -> np.ndarray:
            lower = frame[y0, x0] * (1.0 - wx) + frame[y0, x1] * wx
            upper = frame[y1, x0] * (1.0 - wx) + frame[y1, x1] * wx
            return lower * (1.0 - wy) + upper * wy

        at_t0 = bilinear(values[t0_i])
        at_t1 = bilinear(values[t1_i])
        return at_t0 * (1.0 - wt) + at_t1 * wt

    def velocity(self, points: np.ndarray, time: float) -> np.ndarray:
        return np.stack(
            [self._sample(self.u, points, time), self._sample(self.v, points, time)],
            axis=-1,
        )

    def ivd(self, points: np.ndarray, time: float, tile: Tile) -> np.ndarray:
        # Training used the absolute deviation from patch-mean vorticity.
        x_mask = (self.x >= tile.xmin) & (self.x <= tile.xmax)
        y_mask = (self.y >= tile.ymin) & (self.y <= tile.ymax)
        if not np.any(x_mask) or not np.any(y_mask):
            raise ValueError(f"Tile {tile.index} contains no NetCDF grid points.")
        patch_mean = self.curl[:, y_mask][:, :, x_mask].mean(axis=(1, 2))
        mean_at_time = float(
            np.interp(time, self.times, patch_mean, left=patch_mean[0], right=patch_mean[-1])
        )
        return np.abs(self._sample(self.curl, points, time) - mean_at_time)


def _load_torch_checkpoint(path: Path) -> dict:
    """Use weights-only loading when the installed PyTorch supports it."""

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"{path} does not contain a 'state_dict'.")
    return checkpoint


def load_model(checkpoint_path: Path, torchscript_path: Path, device: str):
    """Load the legacy architecture, then strictly replace it with checkpoint weights."""

    checkpoint = _load_torch_checkpoint(checkpoint_path)
    model = torch.jit.load(str(torchscript_path), map_location=device)
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint/model mismatch: {incompatible}")
    model.eval()
    return model, int(checkpoint.get("epoch", -1))


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_flow_window(
    nc_path: Path, time_index: int, duration: float
) -> tuple[FlowWindow, float, np.ndarray, np.ndarray, dict[str, float]]:
    """Read only the frames needed by the requested integration interval."""

    with nc.Dataset(nc_path, "r") as dataset:
        required = {"u", "v", "xdim", "ydim", "tdim"}
        missing = sorted(required.difference(dataset.variables))
        if missing:
            raise ValueError(f"NetCDF file is missing variables: {missing}")

        x = np.asarray(dataset.variables["xdim"][:], dtype=np.float64)
        y = np.asarray(dataset.variables["ydim"][:], dtype=np.float64)
        all_times = np.asarray(dataset.variables["tdim"][:], dtype=np.float64)
        if not 0 <= time_index < len(all_times) - 1:
            raise ValueError(f"--time-index must be in [0, {len(all_times) - 2}].")

        start_time = float(all_times[time_index])
        target_time = start_time + float(duration)
        end_index = int(np.searchsorted(all_times, target_time, side="left"))
        if end_index >= len(all_times):
            raise ValueError(
                f"Requested interval ends at t={target_time:g}, after the file's "
                f"last time t={all_times[-1]:g}."
            )

        frame_slice = slice(time_index, end_index + 1)
        times = all_times[frame_slice]
        u = np.asarray(dataset.variables["u"][frame_slice], dtype=np.float32)
        v = np.asarray(dataset.variables["v"][frame_slice], dtype=np.float32)
        snapshot_u = u[0].copy()
        snapshot_v = v[0].copy()
        constants = {}
        for name in ("radius", "Re", "nu"):
            if name in dataset.variables:
                constants[name] = float(np.asarray(dataset.variables[name][:]).flat[0])

    return FlowWindow(x, y, times, u, v), start_time, snapshot_u, snapshot_v, constants


def make_tiles(x: np.ndarray, y: np.ndarray, tile_size: float, stride: float) -> list[Tile]:
    ymin, ymax = float(y[0]), float(y[-1])
    xmin, xmax = float(x[0]), float(x[-1])
    if tile_size <= 0.0 or stride <= 0.0:
        raise ValueError("Tile size and stride must be positive.")
    if tile_size > xmax - xmin + 1e-12 or tile_size > ymax - ymin + 1e-12:
        raise ValueError("Tile size cannot exceed either spatial domain extent.")

    def starts(low: float, high: float) -> list[float]:
        last = high - tile_size
        values = list(np.arange(low, last + 0.5 * stride, stride, dtype=float))
        values = [min(value, last) for value in values if value <= last + 1e-9]
        if not values or not math.isclose(values[-1], last, abs_tol=1e-9):
            values.append(last)
        # np.arange plus a forced final tile can produce a duplicate.
        return list(dict.fromkeys(round(value, 12) for value in values))

    result = []
    for y0 in starts(ymin, ymax):
        for x0 in starts(xmin, xmax):
            result.append(
                Tile(len(result), x0, x0 + tile_size, y0, y0 + tile_size)
            )
    return result


def four_arm_cross_seeds(tile: Tile) -> np.ndarray:
    """Generate 8x8 groups in training order: left, down, right, up."""

    grid_count = 8
    start_x = tile.xmin + 0.1 * tile.size
    end_x = tile.xmin + 0.9 * tile.size
    start_y = tile.ymin + 0.1 * tile.size
    end_y = tile.ymin + 0.9 * tile.size
    dx = ((end_x - start_x) / (grid_count - 1)) / 3.0
    dy = ((end_y - start_y) / (grid_count - 1)) / 3.0

    seeds = []
    for center_y in np.linspace(start_y, end_y, grid_count):
        for center_x in np.linspace(start_x, end_x, grid_count):
            seeds.extend(
                [
                    (center_x - dx, center_y),
                    (center_x, center_y - dy),
                    (center_x + dx, center_y),
                    (center_x, center_y + dy),
                ]
            )
    result = np.asarray(seeds, dtype=np.float64)
    if result.shape != (LINES_PER_PATCH, 2):
        raise AssertionError(f"Unexpected seed shape {result.shape}.")
    return result


def _inside_tile(points: np.ndarray, tile: Tile) -> np.ndarray:
    return (
        (points[:, 0] > tile.xmin)
        & (points[:, 0] < tile.xmax)
        & (points[:, 1] > tile.ymin)
        & (points[:, 1] < tile.ymax)
    )


def integrate_tile_pathlines(
    flow: FlowWindow,
    tile: Tile,
    start_time: float,
    duration: float,
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RK4-integrate one tile and return [L,K,7] training-compatible features."""

    seeds = four_arm_cross_seeds(tile)
    positions = seeds.copy()
    seed_normalized = np.column_stack(
        [
            2.0 * (seeds[:, 0] - tile.xmin) / tile.size - 1.0,
            2.0 * (seeds[:, 1] - tile.ymin) / tile.size - 1.0,
        ]
    )
    times = np.linspace(start_time, start_time + duration, PATHLINE_LENGTH)
    features = np.full(
        (PATHLINE_LENGTH, LINES_PER_PATCH, FEATURE_COUNT),
        PADDING_VALUE,
        dtype=np.float32,
    )
    valid = _inside_tile(positions, tile)
    if obstacle_radius > 0.0:
        offset = positions - np.asarray(obstacle_center)[None, :]
        valid &= np.linalg.norm(offset, axis=1) > obstacle_radius
    initially_valid = valid.copy()

    position_scale = 2.0 / tile.size
    velocity_scale = position_scale * duration

    for step, time in enumerate(times):
        ids = np.flatnonzero(valid)
        if ids.size:
            current = positions[ids]
            velocity = flow.velocity(current, float(time))
            ivd = flow.ivd(current, float(time), tile)
            normalized = np.column_stack(
                [
                    2.0 * (current[:, 0] - tile.xmin) / tile.size - 1.0,
                    2.0 * (current[:, 1] - tile.ymin) / tile.size - 1.0,
                ]
            )
            features[step, ids, 0:2] = normalized
            features[step, ids, 2] = (time - start_time) / duration
            # For a square spatial map and normalized time, omega' = duration*omega.
            features[step, ids, 3] = ivd * duration
            features[step, ids, 4] = np.linalg.norm(
                normalized - seed_normalized[ids], axis=1
            )
            features[step, ids, 5:7] = velocity * velocity_scale

        if step == PATHLINE_LENGTH - 1 or ids.size == 0:
            continue

        time = float(time)
        dt = float(times[step + 1] - times[step])
        point = positions[ids]
        k1 = flow.velocity(point, time)
        k2 = flow.velocity(point + 0.5 * dt * k1, time + 0.5 * dt)
        k3 = flow.velocity(point + 0.5 * dt * k2, time + 0.5 * dt)
        k4 = flow.velocity(point + dt * k3, time + dt)
        new_point = point + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        finite = np.all(np.isfinite(new_point), axis=1)
        stays_inside = _inside_tile(new_point, tile)
        if obstacle_radius > 0.0:
            offset = new_point - np.asarray(obstacle_center)[None, :]
            stays_inside &= np.linalg.norm(offset, axis=1) > obstacle_radius
        accepted = finite & stays_inside
        positions[ids[accepted]] = new_point[accepted]
        valid[ids[~accepted]] = False

    # The generator temporarily stores labels in distance[0], but the dataset
    # loader removes them before training.  The inference value is therefore 0.
    features[0, initially_valid, 4] = 0.0
    return features, seeds, initially_valid


def predict_tiles(
    model,
    features: Sequence[np.ndarray],
    device: str,
    batch_size: int,
) -> np.ndarray:
    predictions = []
    for begin in range(0, len(features), batch_size):
        batch = np.stack(features[begin : begin + batch_size])
        tensor = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
        if tensor.shape[1:] != (
            PATHLINE_LENGTH,
            LINES_PER_PATCH,
            FEATURE_COUNT,
        ):
            raise ValueError(f"Unexpected model input shape {tuple(tensor.shape)}.")
        with torch.inference_mode():
            output = model(tensor)
        predictions.append(output.detach().cpu().numpy())
    return np.concatenate(predictions, axis=0)


def merge_duplicate_seeds(
    seed_blocks: Sequence[np.ndarray],
    score_blocks: np.ndarray,
    valid_blocks: Sequence[np.ndarray],
    precision: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average predictions where overlapping tiles place the same physical seed."""

    accum: dict[tuple[float, float], list[float]] = {}
    tile_ids: dict[tuple[float, float], list[int]] = {}
    for tile_id, (seeds, scores, valid) in enumerate(
        zip(seed_blocks, score_blocks, valid_blocks)
    ):
        for point, score, is_valid in zip(seeds, scores, valid):
            if not is_valid:
                continue
            key = (round(float(point[0]), precision), round(float(point[1]), precision))
            accum.setdefault(key, []).append(float(score))
            tile_ids.setdefault(key, []).append(tile_id)
    points = np.asarray(list(accum), dtype=np.float64)
    scores = np.asarray([np.mean(accum[key]) for key in accum], dtype=np.float32)
    source_tile = np.asarray([tile_ids[key][0] for key in accum], dtype=np.int32)
    return points, scores, source_tile


def probability_map(
    x: np.ndarray,
    y: np.ndarray,
    points: np.ndarray,
    scores: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize sparse seed probabilities using normalized Gaussian averaging."""

    weighted = np.zeros((len(y), len(x)), dtype=np.float64)
    weights = np.zeros_like(weighted)
    xi = np.clip(np.searchsorted(x, points[:, 0]), 0, len(x) - 1)
    yi = np.clip(np.searchsorted(y, points[:, 1]), 0, len(y) - 1)
    np.add.at(weighted, (yi, xi), scores)
    np.add.at(weights, (yi, xi), 1.0)
    smooth_weighted = gaussian_filter(weighted, sigma=sigma, mode="nearest")
    smooth_weights = gaussian_filter(weights, sigma=sigma, mode="nearest")
    dense = np.divide(
        smooth_weighted,
        smooth_weights,
        out=np.full_like(smooth_weighted, np.nan),
        where=smooth_weights > 1e-5,
    )
    return dense, smooth_weights


def component_centers(
    probability: np.ndarray,
    support: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    min_pixels: int,
) -> list[tuple[float, float, float, int]]:
    mask = (probability >= threshold) & (support > 1e-5)
    components, count = label(mask)
    detections = []
    for component_id in range(1, count + 1):
        yy, xx = np.nonzero(components == component_id)
        if len(xx) < min_pixels:
            continue
        component_scores = probability[yy, xx]
        weights = np.maximum(component_scores - threshold, 1e-6)
        center_x = float(np.average(x[xx], weights=weights))
        center_y = float(np.average(y[yy], weights=weights))
        detections.append(
            (center_x, center_y, float(np.nanmax(component_scores)), len(xx))
        )
    return detections


def detect_ivd_vortices(
    flow: FlowWindow,
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
    threshold: float | None,
    percentile: float,
    smooth_sigma: float,
    min_pixels: int,
    obstacle_margin: float,
    border_cells: int,
) -> tuple[np.ndarray, float, float, list[IVDDetection]]:
    """Detect connected vortex regions from instantaneous IVD.

    For a two-dimensional flow, IVD is

        IVD(x, t) = |omega(x, t) - spatial_mean(omega(t))|,

    where ``omega = dv/dx - du/dy``.  Positive- and negative-vorticity
    components are labelled separately so adjacent counter-rotating vortices
    are not merged by the absolute value.
    """

    curl = np.asarray(flow.curl[0], dtype=np.float64)
    grid_x, grid_y = np.meshgrid(flow.x, flow.y)
    valid = np.isfinite(curl)

    if obstacle_radius > 0.0:
        distance_to_obstacle = np.hypot(
            grid_x - obstacle_center[0], grid_y - obstacle_center[1]
        )
        valid &= distance_to_obstacle > obstacle_margin * obstacle_radius

    border_cells = max(0, int(border_cells))
    if border_cells:
        if 2 * border_cells >= min(curl.shape):
            raise ValueError("--ivd-border-cells removes the entire flow grid.")
        valid[:border_cells, :] = False
        valid[-border_cells:, :] = False
        valid[:, :border_cells] = False
        valid[:, -border_cells:] = False
    if not np.any(valid):
        raise ValueError("No valid fluid cells remain for IVD calculation.")

    mean_vorticity = float(np.mean(curl[valid]))
    ivd = np.abs(curl - mean_vorticity)
    if smooth_sigma > 0.0:
        ivd = gaussian_filter(ivd, sigma=smooth_sigma, mode="nearest")

    if threshold is None:
        resolved_threshold = float(np.percentile(ivd[valid], percentile))
    else:
        resolved_threshold = float(threshold)
    if not np.isfinite(resolved_threshold) or resolved_threshold <= 0.0:
        raise ValueError("The resolved IVD threshold must be finite and positive.")

    detections = []
    for rotation_sign, sign_mask in (
        (1, curl >= mean_vorticity),
        (-1, curl < mean_vorticity),
    ):
        components, component_count = label(
            valid & sign_mask & (ivd >= resolved_threshold)
        )
        for component_id in range(1, component_count + 1):
            yy, xx = np.nonzero(components == component_id)
            if len(xx) < min_pixels:
                continue
            component_ivd = ivd[yy, xx]
            weights = np.maximum(component_ivd - resolved_threshold, 1e-8)
            detections.append(
                IVDDetection(
                    x=float(np.average(flow.x[xx], weights=weights)),
                    y=float(np.average(flow.y[yy], weights=weights)),
                    peak=float(np.max(component_ivd)),
                    pixels=len(xx),
                    rotation_sign=rotation_sign,
                )
            )

    detections.sort(key=lambda detection: (detection.x, detection.y))
    masked_ivd = np.where(valid, ivd, np.nan)
    return masked_ivd, resolved_threshold, mean_vorticity, detections


def save_scores_csv(
    output_path: Path,
    points: np.ndarray,
    scores: np.ndarray,
    tile_ids: np.ndarray,
    threshold: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "probability", "is_vortex", "source_tile"])
        for point, score, tile_id in zip(points, scores, tile_ids):
            writer.writerow(
                [
                    f"{point[0]:.8g}",
                    f"{point[1]:.8g}",
                    f"{score:.8g}",
                    int(score >= threshold),
                    int(tile_id),
                ]
            )


def plot_detection(
    output_path: Path,
    flow: FlowWindow,
    snapshot_u: np.ndarray,
    snapshot_v: np.ndarray,
    ivd: np.ndarray,
    ivd_threshold: float,
    ivd_detections: Sequence[IVDDetection],
    probability: np.ndarray,
    points: np.ndarray,
    scores: np.ndarray,
    model_detections: Sequence[tuple[float, float, float, int]],
    model_threshold: float,
    start_time: float,
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
) -> None:
    speed = np.hypot(snapshot_u, snapshot_v)
    figure, (ivd_axis, model_axis) = plt.subplots(
        2, 1, figsize=(16, 8.0), sharex=True, constrained_layout=True
    )
    stream_skip = max(1, len(flow.x) // 320)
    # NetCDF stores nominally regular coordinates as float32; recreate exact
    # linspaces because Matplotlib's streamplot validates regular spacing.
    stream_x_full = np.linspace(flow.x[0], flow.x[-1], len(flow.x))
    stream_y = np.linspace(flow.y[0], flow.y[-1], len(flow.y))

    def draw_streamlines(axis) -> None:
        axis.streamplot(
            stream_x_full[::stream_skip],
            stream_y,
            snapshot_u[:, ::stream_skip],
            snapshot_v[:, ::stream_skip],
            color="white",
            density=(4.0, 1.2),
            linewidth=0.45,
            arrowsize=0.55,
            minlength=0.05,
        )

    def draw_obstacle(axis) -> None:
        if obstacle_radius > 0.0:
            axis.add_patch(
                Circle(
                    obstacle_center,
                    obstacle_radius,
                    facecolor="0.15",
                    edgecolor="white",
                    linewidth=1.0,
                    zorder=9,
                )
            )

    # Upper panel: direct IVD criterion.
    ivd_image = ivd_axis.pcolormesh(
        flow.x, flow.y, ivd, shading="auto", cmap="magma", rasterized=True
    )
    draw_streamlines(ivd_axis)
    finite_ivd = np.ma.masked_invalid(ivd)
    if np.any(np.asarray(finite_ivd >= ivd_threshold)):
        ivd_axis.contour(
            flow.x,
            flow.y,
            finite_ivd,
            levels=[ivd_threshold],
            colors=["#ffe66d"],
            linewidths=1.25,
        )
    for detection_id, detection in enumerate(ivd_detections, 1):
        marker_color = "#00e5ff" if detection.rotation_sign > 0 else "#ff4fd8"
        ivd_axis.scatter(
            [detection.x],
            [detection.y],
            marker="x",
            s=66,
            linewidths=1.8,
            c=marker_color,
            zorder=7,
        )
        ivd_axis.annotate(
            f"I{detection_id}\n{detection.peak:.1f}",
            (detection.x, detection.y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=6.5,
            color=marker_color,
            weight="bold",
            zorder=8,
        )
    draw_obstacle(ivd_axis)
    ivd_axis.set(
        xlim=(flow.x[0], flow.x[-1]),
        ylim=(flow.y[0], flow.y[-1]),
        ylabel="y",
        title=(
            f"IVD vortex detection at t={start_time:g}: "
            f"|omega - mean(omega)| >= {ivd_threshold:.3g} "
            f"({len(ivd_detections)} connected regions)"
        ),
        aspect="equal",
    )
    if ivd_detections:
        ivd_axis.scatter(
            [], [], marker="x", c="#00e5ff", label="counter-clockwise"
        )
        ivd_axis.scatter([], [], marker="x", c="#ff4fd8", label="clockwise")
        ivd_axis.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ivd_colorbar = figure.colorbar(ivd_image, ax=ivd_axis, pad=0.01)
    ivd_colorbar.set_label("IVD")

    # Lower panel: pretrained VortexTransformer inference.
    speed_image = model_axis.pcolormesh(
        flow.x, flow.y, speed, shading="auto", cmap="viridis", rasterized=True
    )
    draw_streamlines(model_axis)
    finite_probability = np.ma.masked_invalid(probability)
    if np.any(np.asarray(finite_probability >= model_threshold)):
        model_axis.contour(
            flow.x,
            flow.y,
            finite_probability,
            levels=[model_threshold],
            colors=["#ffb000"],
            linewidths=1.3,
        )

    positive = scores >= model_threshold
    model_axis.scatter(
        points[positive, 0],
        points[positive, 1],
        s=10,
        c=scores[positive],
        cmap="autumn",
        vmin=model_threshold,
        vmax=1.0,
        edgecolors="none",
        alpha=0.9,
        label=f"positive pathline seeds (p >= {model_threshold:g})",
        zorder=5,
    )
    for detection_id, (x_center, y_center, peak, _pixels) in enumerate(
        model_detections, 1
    ):
        model_axis.scatter(
            [x_center],
            [y_center],
            marker="x",
            s=70,
            linewidths=2.0,
            c="red",
            zorder=7,
        )
        model_axis.annotate(
            f"M{detection_id}\n{peak:.2f}",
            (x_center, y_center),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=7,
            color="red",
            weight="bold",
            zorder=8,
        )

    draw_obstacle(model_axis)
    model_axis.set(
        xlim=(flow.x[0], flow.x[-1]),
        ylim=(flow.y[0], flow.y[-1]),
        xlabel="x",
        ylabel="y",
        title=(
            f"best_checkpoint.pth.tar model inference at t={start_time:g} "
            f"({len(model_detections)} connected regions)"
        ),
        aspect="equal",
    )
    if np.any(positive):
        model_axis.legend(loc="upper right", fontsize=8, framealpha=0.85)
    speed_colorbar = figure.colorbar(speed_image, ax=model_axis, pad=0.01)
    speed_colorbar.set_label("velocity magnitude")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect vortices in cylinder2d.nc with the pretrained VortexTransformer."
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
        help="Legacy architecture shell; weights are replaced from --checkpoint.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/cylinder2d_vortex_detection.png"))
    parser.add_argument("--scores-output", type=Path, default=Path("outputs/cylinder2d_vortex_scores.csv"))
    parser.add_argument("--time-index", type=int, default=1000)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument(
        "--tile-size",
        type=float,
        default=None,
        help="Square patch size; default is the full y extent (1.0 for cylinder2d).",
    )
    parser.add_argument(
        "--tile-stride",
        type=float,
        default=None,
        help="Sliding stride; default is half a tile.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--smooth-sigma", type=float, default=3.0)
    parser.add_argument("--min-component-pixels", type=int, default=8)
    parser.add_argument(
        "--ivd-threshold",
        type=float,
        default=None,
        help=(
            "Absolute IVD threshold. By default it is selected from "
            "--ivd-percentile over valid fluid cells."
        ),
    )
    parser.add_argument(
        "--ivd-percentile",
        type=float,
        default=95.0,
        help="Adaptive IVD threshold percentile used when --ivd-threshold is omitted.",
    )
    parser.add_argument("--ivd-smooth-sigma", type=float, default=1.0)
    parser.add_argument("--ivd-min-component-pixels", type=int, default=8)
    parser.add_argument(
        "--ivd-obstacle-margin",
        type=float,
        default=1.5,
        help="Ignore cells within this multiple of the cylinder radius.",
    )
    parser.add_argument(
        "--ivd-border-cells",
        type=int,
        default=2,
        help="Ignore this many outer grid cells when thresholding IVD.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--obstacle-x", type=float, default=0.0)
    parser.add_argument("--obstacle-y", type=float, default=0.0)
    parser.add_argument(
        "--obstacle-radius",
        type=float,
        default=None,
        help="Default: read 'radius' from the NetCDF file.",
    )
    parser.add_argument(
        "--show-tiles",
        action="store_true",
        help="Draw patch boundaries in the output image (debugging).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if args.duration <= 0.0 or args.batch_size < 1:
        raise ValueError("--duration and --batch-size must be positive.")
    if not 0.0 < args.ivd_percentile < 100.0:
        raise ValueError("--ivd-percentile must be between 0 and 100.")
    if args.ivd_smooth_sigma < 0.0:
        raise ValueError("--ivd-smooth-sigma cannot be negative.")
    if args.ivd_min_component_pixels < 1:
        raise ValueError("--ivd-min-component-pixels must be positive.")
    if args.ivd_obstacle_margin < 1.0:
        raise ValueError("--ivd-obstacle-margin must be at least 1.")
    if args.ivd_border_cells < 0:
        raise ValueError("--ivd-border-cells cannot be negative.")

    flow, start_time, snapshot_u, snapshot_v, constants = load_flow_window(
        args.input, args.time_index, args.duration
    )
    y_extent = float(flow.y[-1] - flow.y[0])
    tile_size = y_extent if args.tile_size is None else args.tile_size
    tile_stride = 0.5 * tile_size if args.tile_stride is None else args.tile_stride
    tiles = make_tiles(flow.x, flow.y, tile_size, tile_stride)
    obstacle_center = (args.obstacle_x, args.obstacle_y)
    obstacle_radius = (
        constants.get("radius", 0.0)
        if args.obstacle_radius is None
        else args.obstacle_radius
    )
    ivd, ivd_threshold, mean_vorticity, ivd_detections = detect_ivd_vortices(
        flow,
        obstacle_center,
        obstacle_radius,
        args.ivd_threshold,
        args.ivd_percentile,
        args.ivd_smooth_sigma,
        args.ivd_min_component_pixels,
        args.ivd_obstacle_margin,
        args.ivd_border_cells,
    )

    device = choose_device(args.device)
    print(
        f"Reading {args.input}: {len(flow.times)} frames, "
        f"t=[{start_time:g}, {start_time + args.duration:g}], "
        f"grid={len(flow.x)}x{len(flow.y)}"
    )
    print(
        f"Generating {len(tiles)} square tiles; each model input is "
        f"[16, {LINES_PER_PATCH}, 7]"
    )
    feature_blocks, seed_blocks, valid_blocks = [], [], []
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

    model, epoch = load_model(args.checkpoint, args.torchscript, device)
    print(f"Loaded checkpoint epoch {epoch} on {device}; all state_dict keys matched.")
    score_blocks = predict_tiles(model, feature_blocks, device, args.batch_size)
    points, scores, tile_ids = merge_duplicate_seeds(
        seed_blocks, score_blocks, valid_blocks
    )
    dense_probability, support = probability_map(
        flow.x, flow.y, points, scores, args.smooth_sigma
    )
    model_detections = component_centers(
        dense_probability,
        support,
        flow.x,
        flow.y,
        args.threshold,
        args.min_component_pixels,
    )

    save_scores_csv(args.scores_output, points, scores, tile_ids, args.threshold)
    plot_detection(
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
        model_detections,
        args.threshold,
        start_time,
        obstacle_center,
        obstacle_radius,
    )
    if args.show_tiles:
        # Keep the main plotting path uncluttered; add rectangles in a small
        # second debug image only when explicitly requested.
        debug_path = args.output.with_name(f"{args.output.stem}_tiles{args.output.suffix}")
        fig, ax = plt.subplots(figsize=(16, 4.2), constrained_layout=True)
        ax.pcolormesh(flow.x, flow.y, np.hypot(snapshot_u, snapshot_v), shading="auto")
        for tile in tiles:
            ax.add_patch(
                Rectangle(
                    (tile.xmin, tile.ymin),
                    tile.size,
                    tile.size,
                    fill=False,
                    edgecolor="white",
                    linewidth=0.6,
                )
            )
        ax.set_aspect("equal")
        fig.savefig(debug_path, dpi=180)
        plt.close(fig)

    positive_count = int(np.count_nonzero(scores >= args.threshold))
    print(
        f"IVD: mean vorticity={mean_vorticity:.6g}, "
        f"threshold={ivd_threshold:.6g}, "
        f"connected vortex regions={len(ivd_detections)}"
    )
    for detection_id, detection in enumerate(ivd_detections, 1):
        rotation = "CCW" if detection.rotation_sign > 0 else "CW"
        print(
            f"  I{detection_id}: x={detection.x:.5f}, y={detection.y:.5f}, "
            f"peak={detection.peak:.3f}, pixels={detection.pixels}, "
            f"rotation={rotation}"
        )
    print(
        f"Model: positive seeds={positive_count}/{len(scores)}, "
        f"connected vortex regions={len(model_detections)}"
    )
    for detection_id, (x_center, y_center, peak, pixels) in enumerate(
        model_detections, 1
    ):
        print(
            f"  M{detection_id}: x={x_center:.5f}, y={y_center:.5f}, "
            f"peak={peak:.3f}, pixels={pixels}"
        )
    print(f"Visualization: {args.output}")
    print(f"Per-seed scores: {args.scores_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
