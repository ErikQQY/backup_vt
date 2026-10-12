#!/usr/bin/env python3
"""Detect vortices in ``cylinder2d.nc`` with PathlineTransformerV1.

This entry point is the V1/feature-adaptive counterpart of ``detect.py``.  It
keeps the trained input contract exactly at ``[B, 16, 256, 7]`` with channels
``[x, y, time, IVD, distance, u, v]`` and uses the
``FEATURE_ADAPTIVE_MESH_SEEDING`` layout from
``VatistasFlowDatasetGenerator.py``:

* 64-by-64 jittered background-mesh candidates in each local ``[-1,1]^2``
  patch;
* 205 seeds from the top-30-percent vorticity region and 51 background seeds;
* the same feature monitor, weighted farthest initialization, weighted CVT,
  snapping, and final shuffle used during dataset generation;
* up to 32 generator-style reseeding passes for trajectories with at most two
  valid samples.

The training generator evaluates curl and IVD from an analytical/autograd
velocity evaluator.  A NetCDF field has no such evaluator, so this script uses
the finite-difference vorticity and trilinear interpolation already implemented
by ``detect.FlowWindow``.  All remaining spatial, temporal, velocity, distance,
padding, and ordering conventions follow the training path.

On CPU, ``--backend auto`` uses the supplied TorchScript adapter.  Its graph is
traced specifically for ``L=16`` and ``K=256`` and contains CPU device
constants.  On CUDA/MPS, auto uses the eager ``PathlineTransformerV1`` built
from ``config/segmentation/pathline_transformer_v1.yaml`` and strictly loads
the same checkpoint.

Example
-------
MPLCONFIGDIR=/tmp/pyflowvis-mpl .venv/bin/python detect_test.py \
    --input dataset/cylinder2d.nc --time-index 1000 --device cpu
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml

import detect as common


PATHLINE_LENGTH = common.PATHLINE_LENGTH
SEEDS_PER_PATCH = common.SEEDS_PER_PATCH
FEATURE_COUNT = common.FEATURE_COUNT
PADDING_VALUE = common.PADDING_VALUE
LINES_PER_GROUP = 4
FEATURE_ADAPTIVE_MODE = "feature-adaptive-mesh"


# ---------------------------------------------------------------------------
# PathlineTransformerV1 loading
# ---------------------------------------------------------------------------


class _EagerPathlineAdapter(torch.nn.Module):
    """Expose BaseSeg's tuple input as the traced adapter's tensor interface."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pathlines: torch.Tensor) -> torch.Tensor:
        # PathlineTransformerV1 ignores the vector-field image, but BaseSeg's
        # historical API still expects it as tuple element zero.
        unused_field = torch.empty(
            (pathlines.shape[0], 0),
            device=pathlines.device,
            dtype=pathlines.dtype,
        )
        return self.model((unused_field, pathlines))


def _load_torch_checkpoint(path: Path) -> dict:
    """Safely load the training checkpoint and validate its outer contract."""

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"{path} does not contain a 'state_dict'.")
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError(f"{path} contains an empty or invalid state_dict.")
    return checkpoint


def _load_and_validate_v1_config(path: Path) -> dict:
    """Read the declared training architecture and reject silent mismatches."""

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    model_cfg = payload.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError(f"{path} does not contain a model mapping.")
    encoder = model_cfg.get("encoder_args")
    if not isinstance(encoder, dict):
        raise ValueError(f"{path} does not contain model.encoder_args.")

    expected = {
        "NAME": "PathlineTransformerV1",
        "KpathlinePerGroup": 4,
        "in_channels": 7,
        "dmodel": 144,
        "num_encoder_layers": 3,
        "k": 16,
        "temporal_sampling_ratio": 0.5,
        "group_sampling_ratio": 0.625,
        "full_inference": False,
    }
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual = encoder.get(key)
        if isinstance(expected_value, float):
            matches = isinstance(actual, (int, float)) and math.isclose(
                float(actual), expected_value, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            matches = actual == expected_value
        if not matches:
            mismatches.append(f"{key}={actual!r} (expected {expected_value!r})")
    if model_cfg.get("NAME") != "BaseSeg":
        mismatches.append(
            f"model.NAME={model_cfg.get('NAME')!r} (expected 'BaseSeg')"
        )
    if mismatches:
        raise ValueError(
            f"{path} is not the expected PathlineTransformerV1 training config: "
            + "; ".join(mismatches)
        )
    return model_cfg


def _resolve_device(requested: str) -> str:
    device = common.choose_device(requested)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is false.")
    if device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS was requested, but this PyTorch build cannot use MPS.")
    return device


def load_v1_model(
    checkpoint_path: Path,
    torchscript_path: Path,
    config_path: Path,
    device: str,
    backend: str,
) -> tuple[torch.nn.Module, int, str]:
    """Strictly load V1 through its traced or eager inference path."""

    model_cfg = _load_and_validate_v1_config(config_path)
    checkpoint = _load_torch_checkpoint(checkpoint_path)
    state_dict = checkpoint["state_dict"]
    resolved_backend = (
        "torchscript" if backend == "auto" and device == "cpu" else backend
    )
    if backend == "auto" and device != "cpu":
        resolved_backend = "eager"

    if resolved_backend == "torchscript":
        if device != "cpu":
            raise ValueError(
                "The supplied traced graph contains CPU device constants. Use "
                "--device cpu or select --backend eager for CUDA/MPS."
            )
        model = torch.jit.load(str(torchscript_path), map_location="cpu")

        # V1's trace is an adapter around BaseSeg, so its root keys are
        # model.encoder.* while the checkpoint keys are encoder.*.  Loading
        # into the wrapped model preserves strict checking without rewriting
        # the checkpoint.
        wrapped_model = getattr(model, "model", None)
        if wrapped_model is not None:
            incompatible = wrapped_model.load_state_dict(state_dict, strict=True)
        else:
            script_keys = set(model.state_dict())
            checkpoint_keys = set(state_dict)
            if script_keys == checkpoint_keys:
                compatible_state = state_dict
            elif script_keys == {f"model.{key}" for key in checkpoint_keys}:
                compatible_state = {
                    f"model.{key}": value for key, value in state_dict.items()
                }
            else:
                raise RuntimeError(
                    "TorchScript/checkpoint parameter names do not match the "
                    "PathlineTransformerV1 adapter contract."
                )
            incompatible = model.load_state_dict(compatible_state, strict=True)
    elif resolved_backend == "eager":
        # Import only when eager execution is selected; CPU TorchScript users
        # do not need to initialize the complete model registry.
        from DeepUtils.models import build_model_from_cfg

        base_model = build_model_from_cfg(model_cfg)
        incompatible = base_model.load_state_dict(state_dict, strict=True)
        model = _EagerPathlineAdapter(base_model).to(device)
    else:
        raise ValueError(f"Unknown model backend: {resolved_backend}")

    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint/model mismatch: {incompatible}")
    model.eval()
    return model, int(checkpoint.get("epoch", -1)), resolved_backend


# ---------------------------------------------------------------------------
# FEATURE_ADAPTIVE_MESH_SEEDING
# ---------------------------------------------------------------------------


def _local_to_physical(patch: common.Patch, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return np.column_stack(
        [
            patch.xmin + 0.5 * (points[:, 0] + 1.0) * patch.width,
            patch.ymin + 0.5 * (points[:, 1] + 1.0) * patch.height,
        ]
    )


def _feature_mesh_candidates(
    resolution: int,
    boundary_margin: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Match the generator's jittered cell-centred background mesh."""

    if resolution < 8:
        raise ValueError("--feature-mesh-resolution must be at least 8.")
    if not 0.0 <= boundary_margin < 1.0:
        raise ValueError("--feature-boundary-margin must be in [0,1).")
    lower = -1.0 + boundary_margin
    upper = 1.0 - boundary_margin
    spacing = (upper - lower) / resolution
    coordinates = lower + (np.arange(resolution) + 0.5) * spacing
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    candidates = np.column_stack([xx.ravel(), yy.ravel()])
    candidates += rng.uniform(
        -0.20 * spacing, 0.20 * spacing, size=candidates.shape
    )
    return np.clip(candidates, lower, upper).astype(np.float64, copy=False)


def _weighted_cvt_mesh_points(
    candidates: np.ndarray,
    target_size: np.ndarray,
    point_count: int,
    rng: np.random.Generator,
    iterations: int,
) -> np.ndarray:
    """Generator-identical weighted farthest insertion and discrete CVT."""

    candidates = np.asarray(candidates, dtype=np.float64)
    target_size = np.asarray(target_size, dtype=np.float64)
    point_count = int(point_count)
    iterations = int(iterations)
    if point_count == 0:
        return np.empty((0, 2), dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != 2:
        raise ValueError("Adaptive-mesh candidates must be shaped [N,2].")
    if target_size.shape != (len(candidates),):
        raise ValueError("target_size must contain one value per candidate.")
    if point_count > len(candidates):
        raise ValueError(
            f"Only {len(candidates)} eligible mesh vertices remain for "
            f"{point_count} requested seeds."
        )
    if iterations < 0:
        raise ValueError("--feature-mesh-iterations cannot be negative.")

    eps = np.finfo(np.float64).eps
    density = 1.0 / np.maximum(target_size, eps) ** 2
    density /= max(float(np.max(density)), eps)
    first = int(rng.choice(len(candidates), p=density / density.sum()))
    selected = np.empty(point_count, dtype=np.int64)
    selected[0] = first
    minimum_distance_sq = np.sum(
        (candidates - candidates[first]) ** 2, axis=1
    )
    minimum_distance_sq[first] = 0.0
    available = np.ones(len(candidates), dtype=bool)
    available[first] = False
    for index in range(1, point_count):
        score = minimum_distance_sq * density
        score[~available] = -1.0
        chosen = int(np.argmax(score))
        selected[index] = chosen
        available[chosen] = False
        distance_sq = np.sum(
            (candidates - candidates[chosen]) ** 2, axis=1
        )
        minimum_distance_sq = np.minimum(minimum_distance_sq, distance_sq)

    centers = candidates[selected].copy()
    for _ in range(iterations):
        distance_sq = np.sum(
            (candidates[:, None, :] - centers[None, :, :]) ** 2, axis=2
        )
        owner = np.argmin(distance_sq, axis=1)
        relaxed = centers.copy()
        for center_index in range(point_count):
            member_ids = np.flatnonzero(owner == center_index)
            if not member_ids.size:
                continue
            weights = density[member_ids]
            centroid = np.average(candidates[member_ids], axis=0, weights=weights)
            nearest_local = np.argmin(
                np.sum((candidates[member_ids] - centroid) ** 2, axis=1)
            )
            relaxed[center_index] = candidates[member_ids[nearest_local]]
        if np.array_equal(relaxed, centers):
            break
        centers = relaxed
    return centers


def feature_adaptive_mesh_seeds(
    flow: common.FlowWindow,
    patch: common.Patch,
    start_time: float,
    rng: np.random.Generator,
    high_fraction: float,
    high_vorticity_percentile: float,
    mesh_resolution: int,
    mesh_size_ratio: float,
    monitor_power: float,
    mesh_iterations: int,
    clip_percentile: float,
    boundary_margin: float,
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
    obstacle_seeding_margin: float,
) -> np.ndarray:
    """Return 256 shuffled physical seeds using the training mesh algorithm."""

    high_count = int(round(SEEDS_PER_PATCH * high_fraction))
    low_count = SEEDS_PER_PATCH - high_count
    candidates = _feature_mesh_candidates(
        mesh_resolution, boundary_margin, rng
    )
    physical_candidates = _local_to_physical(patch, candidates)

    # A positive margin is an opt-in cylinder adaptation.  The default zero
    # leaves the generator's complete square candidate domain unchanged.
    if obstacle_seeding_margin > 0.0 and obstacle_radius > 0.0:
        center = np.asarray(obstacle_center, dtype=np.float64)
        keep = np.linalg.norm(
            physical_candidates - center[None, :], axis=1
        ) > obstacle_seeding_margin * obstacle_radius
        candidates = candidates[keep]
        physical_candidates = physical_candidates[keep]

    # For square patches, conversion to local coordinates multiplies curl by
    # one positive constant.  Ranking and normalized monitor values are thus
    # unchanged, so physical finite-difference curl is sufficient here.
    vorticity = flow.sample(
        flow.vorticity, physical_candidates, float(start_time)
    )
    magnitude = np.abs(
        np.nan_to_num(vorticity, nan=0.0, posinf=0.0, neginf=0.0)
    )
    tie_break = rng.random(len(candidates))
    order = np.lexsort((tie_break, magnitude))
    high_candidate_count = int(
        round(
            len(candidates)
            * (1.0 - high_vorticity_percentile / 100.0)
        )
    )
    if high_candidate_count < high_count or (
        len(candidates) - high_candidate_count
    ) < low_count:
        raise ValueError(
            "The adaptive background mesh has too few high/low-vorticity "
            "candidates for the requested 256-seed quota. Increase "
            "--feature-mesh-resolution or reduce --obstacle-seeding-margin."
        )
    high_ids = order[-high_candidate_count:]
    low_ids = order[:-high_candidate_count]

    contrast = np.maximum(magnitude - float(np.min(magnitude)), 0.0)
    scale = float(np.percentile(contrast, clip_percentile))
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        feature = np.full(len(candidates), 0.5, dtype=np.float64)
    else:
        feature = np.clip(contrast / scale, 0.0, 1.0)

    high_strength = np.power(feature[high_ids], monitor_power)
    low_strength = np.power(1.0 - feature[low_ids], monitor_power)
    high_target_size = (
        mesh_size_ratio - (mesh_size_ratio - 1.0) * high_strength
    )
    low_target_size = (
        mesh_size_ratio - (mesh_size_ratio - 1.0) * low_strength
    )
    high_seeds = _weighted_cvt_mesh_points(
        candidates[high_ids],
        high_target_size,
        high_count,
        rng,
        mesh_iterations,
    )
    low_seeds = _weighted_cvt_mesh_points(
        candidates[low_ids],
        low_target_size,
        low_count,
        rng,
        mesh_iterations,
    )
    local_seeds = np.concatenate([high_seeds, low_seeds], axis=0)
    rng.shuffle(local_seeds)
    if local_seeds.shape != (SEEDS_PER_PATCH, 2):
        raise RuntimeError(
            f"Adaptive seeding returned {local_seeds.shape}, expected "
            f"{(SEEDS_PER_PATCH, 2)}."
        )
    return _local_to_physical(patch, local_seeds)


# ---------------------------------------------------------------------------
# Pathline integration and seven-channel feature construction
# ---------------------------------------------------------------------------


def _integrate_positions(
    flow: common.FlowWindow,
    patch: common.Patch,
    seeds: np.ndarray,
    start_time: float,
    duration: float,
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate physical positions with the generator's strict boundary rule."""

    times = np.linspace(
        start_time, start_time + duration, PATHLINE_LENGTH, dtype=np.float64
    )
    positions = np.full(
        (PATHLINE_LENGTH, SEEDS_PER_PATCH, 2), np.nan, dtype=np.float64
    )
    valid = np.zeros((PATHLINE_LENGTH, SEEDS_PER_PATCH), dtype=bool)
    active = patch.contains_strictly(seeds)
    active &= common._outside_obstacle(
        seeds, obstacle_center, obstacle_radius
    )
    positions[0, active] = seeds[active]
    valid[0, active] = True

    for step in range(1, PATHLINE_LENGTH):
        ids = np.flatnonzero(active)
        if not ids.size:
            break
        previous_time = float(times[step - 1])
        dt = float(times[step] - times[step - 1])
        new_points = common._rk4_step(
            flow, positions[step - 1, ids], previous_time, dt
        )
        accepted = np.all(np.isfinite(new_points), axis=1)
        accepted &= patch.contains_strictly(new_points)
        accepted &= common._outside_obstacle(
            new_points, obstacle_center, obstacle_radius
        )
        accepted_ids = ids[accepted]
        positions[step, accepted_ids] = new_points[accepted]
        valid[step, accepted_ids] = True
        active[ids[~accepted]] = False
    return positions, valid, times


def integrate_adaptive_patch(
    flow: common.FlowWindow,
    patch: common.Patch,
    seeds: np.ndarray,
    start_time: float,
    duration: float,
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
    rng: np.random.Generator,
    max_reseed_attempts: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Reseed short trajectories and build one ``[16,256,7]`` block."""

    local_seeds = patch.to_local(seeds)
    reseed_passes = 0
    for attempt in range(max_reseed_attempts + 1):
        seeds = _local_to_physical(patch, local_seeds)
        positions, valid, times = _integrate_positions(
            flow,
            patch,
            seeds,
            start_time,
            duration,
            obstacle_center,
            obstacle_radius,
        )
        valid_lengths = valid.sum(axis=0)
        inside_obstacle = ~common._outside_obstacle(
            seeds, obstacle_center, obstacle_radius
        )
        failed = (valid_lengths <= 2) & ~inside_obstacle
        if not np.any(failed) or attempt == max_reseed_attempts:
            break
        shifts = rng.uniform(1e-5, 0.5, int(failed.sum()))
        local_seeds[failed] *= 1.0 - shifts[:, None]
        reseed_passes += 1

    features = np.full(
        (PATHLINE_LENGTH, SEEDS_PER_PATCH, FEATURE_COUNT),
        PADDING_VALUE,
        dtype=np.float32,
    )
    patch_mean_vorticity = flow.patch_mean_vorticity(patch)
    local_seed_positions = patch.to_local(seeds)
    for step, time_value in enumerate(times):
        ids = np.flatnonzero(valid[step])
        if not ids.size:
            continue
        current = positions[step, ids]
        local = patch.to_local(current)
        velocity = flow.velocity(current, float(time_value))
        vorticity = flow.sample(
            flow.vorticity, current, float(time_value)
        )
        mean_vorticity = float(
            np.interp(time_value, flow.times, patch_mean_vorticity)
        )

        features[step, ids, 0:2] = local
        features[step, ids, 2] = (time_value - start_time) / duration
        features[step, ids, 3] = (
            np.abs(vorticity - mean_vorticity) * duration
        )
        features[step, ids, 4] = np.linalg.norm(
            local - local_seed_positions[ids], axis=1
        )
        features[step, ids, 5] = (
            velocity[:, 0] * patch.x_scale * duration
        )
        features[step, ids, 6] = (
            velocity[:, 1] * patch.y_scale * duration
        )

    # Dataset loading extracts the temporary seed label from this location and
    # resets it before forwarding the tensor.  In inference it is simply zero.
    features[0, valid[0], 4] = 0.0
    common.validate_feature_block(features)
    return features, seeds, valid[0], reseed_passes


def prepare_adaptive_inputs(
    flow: common.FlowWindow,
    patches: Sequence[common.Patch],
    args: argparse.Namespace,
    start_time: float,
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
) -> tuple[common.PreparedInputs, int, int]:
    """Generate seeds and pathline features for all square patches."""

    rng = np.random.default_rng(args.seed)
    feature_blocks: list[np.ndarray] = []
    seed_blocks: list[np.ndarray] = []
    valid_blocks: list[np.ndarray] = []
    total_reseed_passes = 0
    invalid_t0_count = 0
    for patch in patches:
        seeds = feature_adaptive_mesh_seeds(
            flow=flow,
            patch=patch,
            start_time=start_time,
            rng=rng,
            high_fraction=args.feature_high_fraction,
            high_vorticity_percentile=(
                args.feature_high_vorticity_percentile
            ),
            mesh_resolution=args.feature_mesh_resolution,
            mesh_size_ratio=args.feature_mesh_size_ratio,
            monitor_power=args.feature_monitor_power,
            mesh_iterations=args.feature_mesh_iterations,
            clip_percentile=args.feature_clip_percentile,
            boundary_margin=args.feature_boundary_margin,
            obstacle_center=obstacle_center,
            obstacle_radius=obstacle_radius,
            obstacle_seeding_margin=args.obstacle_seeding_margin,
        )
        features, final_seeds, initial_valid, reseed_passes = (
            integrate_adaptive_patch(
                flow,
                patch,
                seeds,
                start_time,
                args.duration,
                obstacle_center,
                obstacle_radius,
                rng,
                args.max_reseed_attempts,
            )
        )
        feature_blocks.append(features)
        seed_blocks.append(final_seeds)
        valid_blocks.append(initial_valid)
        total_reseed_passes += reseed_passes
        invalid_t0_count += int(np.count_nonzero(~initial_valid))

    return (
        common.PreparedInputs(
            list(patches), feature_blocks, seed_blocks, valid_blocks
        ),
        total_reseed_passes,
        invalid_t0_count,
    )


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run PathlineTransformerV1 with FEATURE_ADAPTIVE_MESH_SEEDING "
            "on cylinder2d.nc (or another compatible NetCDF flow)."
        )
    )
    parser.add_argument(
        "--input", type=Path, default=Path("dataset/cylinder2d.nc")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("trainedVortexTransformerV1/best_checkpoint.pth.tar"),
    )
    parser.add_argument(
        "--torchscript",
        type=Path,
        default=Path(
            "trainedVortexTransformerV1/BackUp_best_checkpoint.pthtraced.pt"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/segmentation/pathline_transformer_v1.yaml"),
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "torchscript", "eager"],
        default="auto",
        help="auto uses TorchScript on CPU and eager V1 on CUDA/MPS.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/v1_feature_adaptive_detection.png"),
    )
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=Path("outputs/v1_feature_adaptive_seed_scores.csv"),
    )
    parser.add_argument(
        "--detections-output",
        type=Path,
        default=Path("outputs/v1_feature_adaptive_locations.csv"),
    )

    flow_group = parser.add_argument_group("flow patches and integration")
    flow_group.add_argument("--time-index", type=int, default=1000)
    flow_group.add_argument("--duration", type=float, default=1.0)
    flow_group.add_argument(
        "--tile-size",
        type=float,
        default=None,
        help="Square patch size (default: complete y-domain extent).",
    )
    flow_group.add_argument(
        "--tile-stride",
        type=float,
        default=None,
        help="Sliding stride (default: half a patch).",
    )
    flow_group.add_argument(
        "--max-reseed-attempts",
        type=int,
        default=32,
        help="Training-generator default: 32.",
    )

    adaptive = parser.add_argument_group("feature-adaptive mesh seeding")
    adaptive.add_argument("--seed", type=int, default=7068)
    adaptive.add_argument("--feature-high-fraction", type=float, default=0.80)
    adaptive.add_argument(
        "--feature-high-vorticity-percentile", type=float, default=70.0
    )
    adaptive.add_argument("--feature-mesh-resolution", type=int, default=64)
    adaptive.add_argument("--feature-mesh-size-ratio", type=float, default=4.0)
    adaptive.add_argument("--feature-monitor-power", type=float, default=1.5)
    adaptive.add_argument("--feature-mesh-iterations", type=int, default=5)
    adaptive.add_argument("--feature-clip-percentile", type=float, default=99.0)
    adaptive.add_argument("--feature-boundary-margin", type=float, default=1e-6)
    adaptive.add_argument(
        "--obstacle-seeding-margin",
        type=float,
        default=0.0,
        help=(
            "Opt-in fluid-candidate mask in cylinder radii. Zero exactly keeps "
            "the generator's full square mesh; try 1.5-2 to suppress wall seeds."
        ),
    )

    model = parser.add_argument_group("model postprocessing")
    model.add_argument("--threshold", type=float, default=0.5)
    model.add_argument("--smooth-sigma", type=float, default=3.0)
    model.add_argument("--min-component-pixels", type=int, default=8)

    ivd = parser.add_argument_group("instantaneous IVD reference")
    ivd.add_argument("--ivd-threshold", type=float, default=None)
    ivd.add_argument("--ivd-percentile", type=float, default=95.0)
    ivd.add_argument("--ivd-smooth-sigma", type=float, default=1.0)
    ivd.add_argument("--ivd-min-component-pixels", type=int, default=8)
    ivd.add_argument("--ivd-obstacle-margin", type=float, default=1.5)
    ivd.add_argument("--ivd-border-cells", type=int, default=2)

    runtime = parser.add_argument_group("runtime and cylinder geometry")
    runtime.add_argument("--batch-size", type=int, default=1)
    runtime.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto"
    )
    runtime.add_argument("--obstacle-x", type=float, default=0.0)
    runtime.add_argument("--obstacle-y", type=float, default=0.0)
    runtime.add_argument(
        "--obstacle-radius",
        type=float,
        default=None,
        help="Default: read the radius variable from the NetCDF file.",
    )
    runtime.add_argument(
        "--show-patches",
        action="store_true",
        help="Also save an image with the sliding square patch layout.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    required_files = [args.input, args.checkpoint, args.config]
    if args.backend in {"auto", "torchscript"}:
        required_files.append(args.torchscript)
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required file(s) not found: " + ", ".join(missing))
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive.")
    if args.tile_size is not None and args.tile_size <= 0.0:
        raise ValueError("--tile-size must be positive.")
    if args.tile_stride is not None and args.tile_stride <= 0.0:
        raise ValueError("--tile-stride must be positive.")
    if args.max_reseed_attempts < 0:
        raise ValueError("--max-reseed-attempts cannot be negative.")
    if not 0.0 <= args.feature_high_fraction <= 1.0:
        raise ValueError("--feature-high-fraction must be in [0,1].")
    if not 0.0 < args.feature_high_vorticity_percentile < 100.0:
        raise ValueError(
            "--feature-high-vorticity-percentile must be in (0,100)."
        )
    if args.feature_mesh_resolution < 8:
        raise ValueError("--feature-mesh-resolution must be at least 8.")
    if args.feature_mesh_size_ratio < 1.0:
        raise ValueError("--feature-mesh-size-ratio must be at least 1.")
    if args.feature_monitor_power <= 0.0:
        raise ValueError("--feature-monitor-power must be positive.")
    if args.feature_mesh_iterations < 0:
        raise ValueError("--feature-mesh-iterations cannot be negative.")
    if not 0.0 < args.feature_clip_percentile <= 100.0:
        raise ValueError("--feature-clip-percentile must be in (0,100].")
    if not 0.0 <= args.feature_boundary_margin < 1.0:
        raise ValueError("--feature-boundary-margin must be in [0,1).")
    if args.obstacle_seeding_margin < 0.0:
        raise ValueError("--obstacle-seeding-margin cannot be negative.")
    if args.obstacle_radius is not None and args.obstacle_radius < 0.0:
        raise ValueError("--obstacle-radius cannot be negative.")
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
    if args.ivd_threshold is not None and args.ivd_threshold <= 0.0:
        raise ValueError("--ivd-threshold must be positive.")
    if args.ivd_smooth_sigma < 0.0:
        raise ValueError("--ivd-smooth-sigma cannot be negative.")
    if args.ivd_min_component_pixels < 1:
        raise ValueError("--ivd-min-component-pixels must be positive.")
    if args.ivd_obstacle_margin < 1.0:
        raise ValueError("--ivd-obstacle-margin must be at least 1.")
    if args.ivd_border_cells < 0:
        raise ValueError("--ivd-border-cells cannot be negative.")


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    device = _resolve_device(args.device)
    model, epoch, backend = load_v1_model(
        args.checkpoint,
        args.torchscript,
        args.config,
        device,
        args.backend,
    )

    flow, start_time, snapshot_u, snapshot_v, constants = common.load_flow_window(
        args.input, args.time_index, args.duration
    )
    obstacle_center = (args.obstacle_x, args.obstacle_y)
    obstacle_radius = (
        constants.get("radius", 0.0)
        if args.obstacle_radius is None
        else args.obstacle_radius
    )
    tile_size = (
        float(flow.y[-1] - flow.y[0])
        if args.tile_size is None
        else args.tile_size
    )
    tile_stride = (
        0.5 * tile_size if args.tile_stride is None else args.tile_stride
    )
    patches = common.make_cross_patches(
        flow.x, flow.y, tile_size, tile_stride
    )
    prepared, reseed_passes, invalid_t0_count = prepare_adaptive_inputs(
        flow,
        patches,
        args,
        start_time,
        obstacle_center,
        obstacle_radius,
    )

    print(
        f"Reading {args.input}: grid={len(flow.x)}x{len(flow.y)}, "
        f"t=[{start_time:g}, {start_time + args.duration:g}]"
    )
    print(
        "Preprocessing: FEATURE_ADAPTIVE_MESH_SEEDING; "
        f"{int(round(SEEDS_PER_PATCH * args.feature_high_fraction))}/"
        f"{SEEDS_PER_PATCH - int(round(SEEDS_PER_PATCH * args.feature_high_fraction))} "
        "high/low-vorticity CVT seeds per square patch."
    )
    print(
        "NetCDF adapter: finite-difference curl + trilinear sampling; local "
        "position/time/velocity/IVD scaling follows the training generator."
    )
    print(
        f"Prepared {len(patches)} patches as "
        f"[{PATHLINE_LENGTH},{SEEDS_PER_PATCH},{FEATURE_COUNT}]; "
        f"reseed passes={reseed_passes}, invalid t0 seeds={invalid_t0_count}."
    )
    print(
        f"Loaded checkpoint epoch {epoch} on {device} via {backend}; "
        "state_dict matched strictly."
    )

    score_blocks = common.predict_batches(
        model, prepared.features, device, args.batch_size
    )
    points, scores, source_patches = common.merge_duplicate_seeds(
        prepared.seeds, score_blocks, prepared.display_valid
    )
    probability, support = common.build_probability_map(
        flow.x, flow.y, points, scores, args.smooth_sigma
    )
    common.mask_obstacle(
        probability,
        support,
        flow.x,
        flow.y,
        obstacle_center,
        obstacle_radius,
    )
    model_detections = common.detect_probability_regions(
        probability,
        support,
        flow.x,
        flow.y,
        args.threshold,
        args.min_component_pixels,
    )
    ivd, ivd_threshold, mean_vorticity, ivd_detections = (
        common.detect_ivd_regions(
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
    )

    common.save_seed_scores(
        args.scores_output,
        points,
        scores,
        source_patches,
        args.threshold,
    )
    common.save_detections(args.detections_output, model_detections)
    common.plot_detection(
        args.output,
        flow,
        snapshot_u,
        snapshot_v,
        ivd,
        ivd_threshold,
        ivd_detections,
        probability,
        points,
        scores,
        model_detections,
        args.threshold,
        start_time,
        FEATURE_ADAPTIVE_MODE,
        obstacle_center,
        obstacle_radius,
    )
    if args.show_patches:
        common.plot_patch_layout(
            args.output, flow, snapshot_u, snapshot_v, prepared.patches
        )

    positive_count = int(np.count_nonzero(scores >= args.threshold))
    print(
        f"Model-positive seeds: {positive_count}/{len(scores)} at "
        f"p >= {args.threshold:g}; connected regions: {len(model_detections)}"
    )
    for detection_id, detection in enumerate(model_detections, 1):
        print(
            f"  M{detection_id}: x={detection.x:.5f}, y={detection.y:.5f}, "
            f"peak={detection.peak:.3f}, pixels={detection.pixels}"
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
