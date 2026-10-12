#!/usr/bin/env python3
"""Detect vortex regions in a NetCDF flow with the bundled transformer.

The checkpoint consumes pathline features, not a raw velocity image.  Each
model input has shape ``[16, 256, 7]`` and stores
``[x, y, time, IVD, distance, u, v]``.  The model returns one vortex-region
probability for every pathline seed.

Two preprocessing modes are intentionally kept separate:

``legacy-grid`` (default)
    Reproduces the layout observed in ``demoValidationDataset``: 32-by-32
    NetCDF grid patches, a row-major 16-by-16 seed grid, and full-domain
    integration.  Model time and velocity use the recovered legacy time
    reparameterization.  IVD uses the signed-vorticity mean over a configurable
    source-frame window, takes the absolute deviation on the complete grid,
    and only then interpolates to pathline positions.

``cross``
    Experimental compatibility mode for the current synthetic-data
    generator's structural conventions: square physical patches, 8-by-8
    four-arm cross seed groups, normalized time, and trajectories that stop at
    the patch boundary.

The old raw-NetCDF adapter source is absent, so these conventions were
reconstructed and regression-tested against the bundled real-cylinder
validation binaries.  Position, time, and velocity can be reproduced to
floating-point tolerance; small IVD differences remain possible because the
    original path-extraction entry point is unavailable.

The network segments vortex-member seeds; it does not regress vortex centers.
The reported locations are probability-weighted centroids of thresholded
connected regions and should be interpreted as approximate locations.

Example
-------
MPLCONFIGDIR=/tmp/pyflowvis-mpl .venv/bin/python detect.py \
    --preprocessing legacy-grid --time-index 1000 --device cpu
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
from scipy.ndimage import gaussian_filter, label

# Inference is normally run headlessly (CI, SSH, or a remote shell).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


PATHLINE_LENGTH = 16
SEEDS_PER_PATCH = 256
FEATURE_COUNT = 7
PADDING_VALUE = -1000.0
LEGACY_IVD_SCALE = 0.5
LEGACY_DEFAULT_DURATION = 0.72
LEGACY_DEFAULT_MODEL_TIME_START = 11.16666698

LEGACY_GRID_MODE = "legacy-grid"
CROSS_MODE = "cross"
LEGACY_TIME_ANCHORED = "anchored"
LEGACY_TIME_REPLAY = "replay"


# ---------------------------------------------------------------------------
# Data structures and flow interpolation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Patch:
    """A physical patch mapped to a local ``[-1, 1] x [-1, 1]`` frame."""

    index: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    x_start: int | None = None
    y_start: int | None = None
    point_count: int | None = None

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def x_scale(self) -> float:
        return 2.0 / self.width

    @property
    def y_scale(self) -> float:
        return 2.0 / self.height

    def to_local(self, points: np.ndarray) -> np.ndarray:
        """Map physical ``[x, y]`` points into this patch's local frame."""

        return np.column_stack(
            [
                (points[:, 0] - self.xmin) * self.x_scale - 1.0,
                (points[:, 1] - self.ymin) * self.y_scale - 1.0,
            ]
        )

    def contains_strictly(self, points: np.ndarray) -> np.ndarray:
        return (
            (points[:, 0] > self.xmin)
            & (points[:, 0] < self.xmax)
            & (points[:, 1] > self.ymin)
            & (points[:, 1] < self.ymax)
        )


@dataclass(frozen=True)
class RegionDetection:
    """One thresholded connected region and its weighted centroid."""

    x: float
    y: float
    peak: float
    pixels: int
    rotation_sign: int = 0


@dataclass
class PreparedInputs:
    """All model inputs plus their corresponding physical seed positions."""

    patches: list[Patch]
    features: list[np.ndarray]
    seeds: list[np.ndarray]
    display_valid: list[np.ndarray]


class FlowWindow:
    """Velocity and vorticity interpolation over one integration interval."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        times: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        legacy_ivd_reference_mean: float | None = None,
    ) -> None:
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.times = np.asarray(times, dtype=np.float64)
        self.u = np.asarray(u, dtype=np.float32)
        self.v = np.asarray(v, dtype=np.float32)

        expected_shape = (len(self.times), len(self.y), len(self.x))
        if self.u.shape != expected_shape or self.v.shape != expected_shape:
            raise ValueError(
                "Velocity arrays must have shape [T,Y,X]; expected "
                f"{expected_shape}, got u={self.u.shape}, v={self.v.shape}."
            )
        # Second-order spatial derivatives need three x/y samples.  Linear
        # time interpolation only needs two frames.
        if min(len(self.x), len(self.y)) < 3 or len(self.times) < 2:
            raise ValueError(
                "At least three spatial samples and two time frames are required."
            )
        if not (
            np.all(np.diff(self.x) > 0.0)
            and np.all(np.diff(self.y) > 0.0)
            and np.all(np.diff(self.times) > 0.0)
        ):
            raise ValueError("x, y, and time coordinates must be strictly increasing.")

        # Cross-mode preprocessing and the plotted IVD reference retain the
        # existing second-order outer-boundary derivative.
        dvdx = np.gradient(self.v, self.x, axis=2, edge_order=2)
        dudy = np.gradient(self.u, self.y, axis=1, edge_order=2)
        self.vorticity = np.asarray(dvdx - dudy, dtype=np.float32)

        self.legacy_ivd_reference_mean = legacy_ivd_reference_mean
        self.legacy_ivd: np.ndarray | None = None
        if legacy_ivd_reference_mean is not None:
            # Bundled cylinder binaries use NumPy's default edge_order=1 on the
            # full grid, a fixed spatiotemporal reference mean, abs on-grid,
            # and interpolation only afterwards.
            legacy_dvdx = np.gradient(self.v, self.x, axis=2, edge_order=1)
            legacy_dudy = np.gradient(self.u, self.y, axis=1, edge_order=1)
            legacy_curl = legacy_dvdx - legacy_dudy
            self.legacy_ivd = np.asarray(
                LEGACY_IVD_SCALE
                * np.abs(legacy_curl - legacy_ivd_reference_mean),
                dtype=np.float32,
            )

    @staticmethod
    def _bracket(axis: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, ...]:
        upper = np.searchsorted(axis, values, side="right")
        upper = np.clip(upper, 1, len(axis) - 1)
        lower = upper - 1
        weight = (values - axis[lower]) / (axis[upper] - axis[lower])
        return lower, upper, np.clip(weight, 0.0, 1.0)

    def sample(
        self, field: np.ndarray, points: np.ndarray, time: float
    ) -> np.ndarray:
        """Trilinearly sample a ``[T,Y,X]`` field at many physical points."""

        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Expected points shaped [N,2], got {points.shape}.")

        x0, x1, wx = self._bracket(self.x, points[:, 0])
        y0, y1, wy = self._bracket(self.y, points[:, 1])
        t0, t1, wt_array = self._bracket(
            self.times, np.asarray([time], dtype=np.float64)
        )
        t0_index, t1_index = int(t0[0]), int(t1[0])
        wt = float(wt_array[0])

        def bilinear(frame: np.ndarray) -> np.ndarray:
            bottom = frame[y0, x0] * (1.0 - wx) + frame[y0, x1] * wx
            top = frame[y1, x0] * (1.0 - wx) + frame[y1, x1] * wx
            return bottom * (1.0 - wy) + top * wy

        at_t0 = bilinear(field[t0_index])
        at_t1 = bilinear(field[t1_index])
        return at_t0 * (1.0 - wt) + at_t1 * wt

    def velocity(self, points: np.ndarray, time: float) -> np.ndarray:
        return np.stack(
            [self.sample(self.u, points, time), self.sample(self.v, points, time)],
            axis=-1,
        )

    def inside_domain(self, points: np.ndarray) -> np.ndarray:
        """Use the strict boundary convention of the historical integrator."""

        return (
            (points[:, 0] > self.x[0])
            & (points[:, 0] < self.x[-1])
            & (points[:, 1] > self.y[0])
            & (points[:, 1] < self.y[-1])
        )

    def inside_closed_domain(self, points: np.ndarray) -> np.ndarray:
        """Include t0 boundary seeds before applying strict step acceptance."""

        return (
            (points[:, 0] >= self.x[0])
            & (points[:, 0] <= self.x[-1])
            & (points[:, 1] >= self.y[0])
            & (points[:, 1] <= self.y[-1])
        )

    def patch_mean_vorticity(self, patch: Patch) -> np.ndarray:
        """Return one mean-vorticity value per frame for a specific patch."""

        if (
            patch.x_start is not None
            and patch.y_start is not None
            and patch.point_count is not None
        ):
            x_slice = slice(patch.x_start, patch.x_start + patch.point_count)
            y_slice = slice(patch.y_start, patch.y_start + patch.point_count)
            values = self.vorticity[:, y_slice, x_slice]
        else:
            x_mask = (self.x >= patch.xmin) & (self.x <= patch.xmax)
            y_mask = (self.y >= patch.ymin) & (self.y <= patch.ymax)
            if not np.any(x_mask) or not np.any(y_mask):
                raise ValueError(f"Patch {patch.index} contains no NetCDF cells.")
            values = self.vorticity[:, y_mask][:, :, x_mask]

        if values.shape[1] < 2 or values.shape[2] < 2:
            raise ValueError(f"Patch {patch.index} is too small for IVD.")
        return values.mean(axis=(1, 2), dtype=np.float64)


# ---------------------------------------------------------------------------
# Loading and deterministic model execution
# ---------------------------------------------------------------------------


def _spatiotemporal_vorticity_mean(
    dataset: nc.Dataset,
    x: np.ndarray,
    y: np.ndarray,
    start_index: int,
    end_index: int,
    chunk_size: int = 32,
) -> float:
    """Compute the recovered legacy curl reference in bounded memory."""

    total = 0.0
    value_count = 0
    for begin in range(start_index, end_index + 1, chunk_size):
        end = min(begin + chunk_size, end_index + 1)
        u = np.asarray(dataset.variables["u"][begin:end], dtype=np.float32)
        v = np.asarray(dataset.variables["v"][begin:end], dtype=np.float32)
        # NumPy's default edge_order=1 is part of the recovered profile.
        curl = np.gradient(v, x, axis=2) - np.gradient(u, y, axis=1)
        total += float(np.sum(curl, dtype=np.float64))
        value_count += curl.size
    if value_count == 0:
        raise ValueError("The legacy IVD reference frame range is empty.")
    return total / value_count


def load_flow_window(
    nc_path: Path,
    time_index: int,
    duration: float,
    legacy_reference_indices: tuple[int, int] | None = None,
    legacy_reference_mean: float | None = None,
) -> tuple[FlowWindow, float, np.ndarray, np.ndarray, dict[str, float]]:
    """Read the integration frames and, when requested, legacy IVD reference."""

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
        target_time = start_time + duration
        end_index = int(np.searchsorted(all_times, target_time, side="left"))
        if end_index >= len(all_times):
            raise ValueError(
                f"Requested interval ends at t={target_time:g}, after the "
                f"file's last time t={all_times[-1]:g}."
            )

        frames = slice(time_index, end_index + 1)
        times = all_times[frames]
        u = np.asarray(dataset.variables["u"][frames], dtype=np.float32)
        v = np.asarray(dataset.variables["v"][frames], dtype=np.float32)
        snapshot_u = u[0].copy()
        snapshot_v = v[0].copy()

        if legacy_reference_indices is not None and legacy_reference_mean is not None:
            raise ValueError(
                "Specify either legacy reference frames or a precomputed mean, not both."
            )
        resolved_reference_mean = legacy_reference_mean
        if legacy_reference_indices is not None:
            reference_start, reference_end = legacy_reference_indices
            if not 0 <= reference_start <= reference_end < len(all_times):
                raise ValueError(
                    "Legacy IVD reference indices must satisfy "
                    f"0 <= start <= end < {len(all_times)}."
                )
            resolved_reference_mean = _spatiotemporal_vorticity_mean(
                dataset, x, y, reference_start, reference_end
            )
        if resolved_reference_mean is not None and not np.isfinite(
            resolved_reference_mean
        ):
            raise ValueError("The legacy IVD reference mean must be finite.")

        constants: dict[str, float] = {}
        for name in ("radius", "Re", "nu"):
            if name in dataset.variables:
                constants[name] = float(np.asarray(dataset.variables[name][:]).flat[0])

    flow = FlowWindow(
        x,
        y,
        times,
        u,
        v,
        legacy_ivd_reference_mean=resolved_reference_mean,
    )
    return flow, start_time, snapshot_u, snapshot_v, constants


def _load_torch_checkpoint(path: Path) -> dict:
    """Use safe weights-only loading whenever the PyTorch version supports it."""

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"{path} does not contain a 'state_dict'.")
    return checkpoint


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(checkpoint_path: Path, torchscript_path: Path, device: str):
    """Load the deterministic traced forward and strictly apply checkpoint weights.

    The current Python model class samples pathlines randomly even in ``eval``
    mode.  Its companion TorchScript graph instead fixes the sampled indices,
    so that graph is required for reproducible inference.
    """

    checkpoint = _load_torch_checkpoint(checkpoint_path)
    model = torch.jit.load(str(torchscript_path), map_location=device)
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint/model mismatch: {incompatible}")
    model.eval()
    return model, int(checkpoint.get("epoch", -1))


def predict_batches(
    model,
    feature_blocks: Sequence[np.ndarray],
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Run deterministic batched inference and validate the model contract."""

    expected_input = (PATHLINE_LENGTH, SEEDS_PER_PATCH, FEATURE_COUNT)
    predictions: list[np.ndarray] = []
    for begin in range(0, len(feature_blocks), batch_size):
        batch = np.stack(feature_blocks[begin : begin + batch_size])
        if tuple(batch.shape[1:]) != expected_input:
            raise ValueError(
                f"Expected model input [B,16,256,7], got {batch.shape}."
            )
        tensor = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            output = model(tensor)
        expected_output = (len(batch), SEEDS_PER_PATCH)
        if tuple(output.shape) != expected_output:
            raise RuntimeError(
                f"Expected model output {expected_output}, got {tuple(output.shape)}."
            )
        values = output.detach().cpu().numpy()
        if not np.all(np.isfinite(values)):
            raise RuntimeError("Model output contains non-finite probabilities.")
        if np.min(values) < -1e-6 or np.max(values) > 1.0 + 1e-6:
            raise RuntimeError("Model output is outside the expected sigmoid range [0,1].")
        predictions.append(values)

    if not predictions:
        raise ValueError("No model patches were prepared.")
    return np.concatenate(predictions, axis=0)


def validate_feature_block(features: np.ndarray) -> None:
    """Fail early if a prepared pathline tensor violates training invariants."""

    expected = (PATHLINE_LENGTH, SEEDS_PER_PATCH, FEATURE_COUNT)
    if features.shape != expected:
        raise ValueError(f"Expected feature block {expected}, got {features.shape}.")
    if features.dtype != np.float32:
        raise ValueError(f"Feature block must be float32, got {features.dtype}.")
    if not np.all(np.isfinite(features)):
        raise ValueError("Feature block contains NaN or infinity.")

    is_padding_value = features == PADDING_VALUE
    fully_padded = np.all(is_padding_value, axis=2)
    partially_padded = np.any(is_padding_value, axis=2) & ~fully_padded
    if np.any(partially_padded):
        raise ValueError("A padded pathline step must use -1000 in all 7 channels.")
    if np.any(~fully_padded[1:] & fully_padded[:-1]):
        raise ValueError("A pathline cannot become valid again after padding.")

    valid_at_t0 = ~fully_padded[0]
    if np.any(features[0, valid_at_t0, 4] != 0.0):
        raise ValueError("distance[0] must be zero before model inference.")


# ---------------------------------------------------------------------------
# Patch construction and pathline features
# ---------------------------------------------------------------------------


def _grid_starts(axis_size: int, patch_points: int, stride_points: int) -> list[int]:
    if patch_points < 2:
        raise ValueError("--patch-points must be at least 2.")
    if patch_points > axis_size:
        raise ValueError(
            f"A {patch_points}-point patch does not fit an axis of size {axis_size}."
        )
    if stride_points < 1:
        raise ValueError("--patch-stride-points must be positive.")

    last = axis_size - patch_points
    starts = list(range(0, last + 1, stride_points))
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def make_legacy_grid_patches(
    x: np.ndarray,
    y: np.ndarray,
    patch_points: int,
    stride_points: int,
) -> list[Patch]:
    """Build index-aligned patches in the bundled validation convention."""

    patches: list[Patch] = []
    for y_start in _grid_starts(len(y), patch_points, stride_points):
        for x_start in _grid_starts(len(x), patch_points, stride_points):
            patches.append(
                Patch(
                    index=len(patches),
                    x_start=x_start,
                    y_start=y_start,
                    point_count=patch_points,
                    xmin=float(x[x_start]),
                    xmax=float(x[x_start + patch_points - 1]),
                    ymin=float(y[y_start]),
                    ymax=float(y[y_start + patch_points - 1]),
                )
            )
    return patches


def _physical_starts(low: float, high: float, size: float, stride: float) -> list[float]:
    last = high - size
    starts = list(np.arange(low, last + 0.5 * stride, stride, dtype=float))
    starts = [min(value, last) for value in starts if value <= last + 1e-9]
    if not starts or not math.isclose(starts[-1], last, abs_tol=1e-9):
        starts.append(last)
    return list(dict.fromkeys(round(value, 12) for value in starts))


def make_cross_patches(
    x: np.ndarray,
    y: np.ndarray,
    tile_size: float,
    tile_stride: float,
) -> list[Patch]:
    """Build square physical patches for the four-arm cross generator."""

    xmin, xmax = float(x[0]), float(x[-1])
    ymin, ymax = float(y[0]), float(y[-1])
    if tile_size <= 0.0 or tile_stride <= 0.0:
        raise ValueError("--tile-size and --tile-stride must be positive.")
    if tile_size > min(xmax - xmin, ymax - ymin) + 1e-12:
        raise ValueError("--tile-size cannot exceed either spatial domain extent.")

    patches: list[Patch] = []
    for y0 in _physical_starts(ymin, ymax, tile_size, tile_stride):
        for x0 in _physical_starts(xmin, xmax, tile_size, tile_stride):
            patches.append(
                Patch(
                    index=len(patches),
                    xmin=x0,
                    xmax=x0 + tile_size,
                    ymin=y0,
                    ymax=y0 + tile_size,
                )
            )
    return patches


def regular_grid_seeds(patch: Patch) -> np.ndarray:
    """Return the row-major 16-by-16 seed layout seen in legacy validation."""

    seed_x, seed_y = np.meshgrid(
        np.linspace(patch.xmin, patch.xmax, 16),
        np.linspace(patch.ymin, patch.ymax, 16),
    )
    seeds = np.column_stack([seed_x.ravel(), seed_y.ravel()])
    if seeds.shape != (SEEDS_PER_PATCH, 2):
        raise AssertionError(f"Unexpected regular seed shape {seeds.shape}.")
    return seeds


def four_arm_cross_seeds(patch: Patch) -> np.ndarray:
    """Return 8-by-8 groups ordered left, down, right, then up."""

    grid_count = 8
    x_centers = np.linspace(
        patch.xmin + 0.1 * patch.width,
        patch.xmin + 0.9 * patch.width,
        grid_count,
    )
    y_centers = np.linspace(
        patch.ymin + 0.1 * patch.height,
        patch.ymin + 0.9 * patch.height,
        grid_count,
    )
    dx = ((x_centers[-1] - x_centers[0]) / (grid_count - 1)) / 3.0
    dy = ((y_centers[-1] - y_centers[0]) / (grid_count - 1)) / 3.0

    seeds: list[tuple[float, float]] = []
    for center_y in y_centers:
        for center_x in x_centers:
            seeds.extend(
                [
                    (center_x - dx, center_y),
                    (center_x, center_y - dy),
                    (center_x + dx, center_y),
                    (center_x, center_y + dy),
                ]
            )
    result = np.asarray(seeds, dtype=np.float64)
    if result.shape != (SEEDS_PER_PATCH, 2):
        raise AssertionError(f"Unexpected cross seed shape {result.shape}.")
    return result


def _outside_obstacle(
    points: np.ndarray,
    center: tuple[float, float],
    radius: float,
) -> np.ndarray:
    if radius <= 0.0:
        return np.ones(len(points), dtype=bool)
    offset = points - np.asarray(center, dtype=np.float64)[None, :]
    return np.linalg.norm(offset, axis=1) > radius


def _rk4_step(
    flow: FlowWindow,
    points: np.ndarray,
    time: float,
    dt: float,
) -> np.ndarray:
    k1 = flow.velocity(points, time)
    k2 = flow.velocity(points + 0.5 * dt * k1, time + 0.5 * dt)
    k3 = flow.velocity(points + 0.5 * dt * k2, time + 0.5 * dt)
    k4 = flow.velocity(points + dt * k3, time + dt)
    return points + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _pathline_times(start_time: float, duration: float) -> np.ndarray:
    return np.linspace(start_time, start_time + duration, PATHLINE_LENGTH)


def integrate_legacy_grid_patch(
    flow: FlowWindow,
    patch: Patch,
    start_time: float,
    duration: float,
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
    legacy_time_mode: str,
    model_time_start: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one ``legacy-grid`` input shaped ``[16,256,7]``.

    Boundary seeds are allowed to take their first step; subsequent positions
    must be strictly inside the full domain.  Trajectories may leave the seed
    patch and continue until they leave the complete NetCDF domain.

    The bundled binaries reveal the following legacy reparameterization for a
    raw interval of length ``duration``: model time spans exactly one unit,
    local velocity is multiplied by ``duration``, and full-grid IVD is
    interpolated only after its absolute deviation and scale are applied.
    """

    seeds = regular_grid_seeds(patch)
    positions = seeds.copy()
    local_seeds = patch.to_local(seeds)
    times = _pathline_times(start_time, duration)
    features = np.full(
        (PATHLINE_LENGTH, SEEDS_PER_PATCH, FEATURE_COUNT),
        PADDING_VALUE,
        dtype=np.float32,
    )
    if flow.legacy_ivd is None:
        raise ValueError(
            "legacy-grid requires an IVD reference mean or reference frames."
        )
    if legacy_time_mode == LEGACY_TIME_ANCHORED and model_time_start is None:
        raise ValueError("anchored legacy time requires --model-time-start.")

    active = flow.inside_closed_domain(positions)
    for step, time_value in enumerate(times):
        # The legacy files contain a t0 feature for all regular-grid seeds,
        # including seeds exactly on the full-domain boundary.
        ids = np.arange(SEEDS_PER_PATCH) if step == 0 else np.flatnonzero(active)
        if ids.size:
            current = positions[ids]
            local = patch.to_local(current)
            velocity = flow.velocity(current, float(time_value))
            ivd = flow.sample(flow.legacy_ivd, current, float(time_value))

            if legacy_time_mode == LEGACY_TIME_REPLAY:
                model_time = float(time_value) / duration
            elif legacy_time_mode == LEGACY_TIME_ANCHORED:
                model_time = float(model_time_start) + (
                    float(time_value) - start_time
                ) / duration
            else:
                raise ValueError(f"Unknown legacy time mode: {legacy_time_mode}")

            features[step, ids, 0:2] = local
            features[step, ids, 2] = model_time
            features[step, ids, 3] = ivd
            features[step, ids, 4] = np.linalg.norm(
                local - local_seeds[ids], axis=1
            )
            features[step, ids, 5] = (
                velocity[:, 0] * patch.x_scale * duration
            )
            features[step, ids, 6] = (
                velocity[:, 1] * patch.y_scale * duration
            )

        if step == PATHLINE_LENGTH - 1:
            continue
        active_ids = np.flatnonzero(active)
        if not active_ids.size:
            continue
        dt = float(times[step + 1] - time_value)
        new_points = _rk4_step(
            flow, positions[active_ids], float(time_value), dt
        )
        accepted = np.all(np.isfinite(new_points), axis=1)
        accepted &= flow.inside_domain(new_points)
        positions[active_ids[accepted]] = new_points[accepted]
        active[active_ids[~accepted]] = False

    # Training stored the label temporarily in distance[0]; the dataset loader
    # removed it before the forward pass.
    features[0, :, 4] = 0.0
    display_valid = _outside_obstacle(seeds, obstacle_center, obstacle_radius)
    return features, seeds, display_valid


def integrate_cross_patch(
    flow: FlowWindow,
    patch: Patch,
    start_time: float,
    duration: float,
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one current-generator four-arm cross input."""

    seeds = four_arm_cross_seeds(patch)
    positions = seeds.copy()
    local_seeds = patch.to_local(seeds)
    times = _pathline_times(start_time, duration)
    patch_mean = flow.patch_mean_vorticity(patch)
    features = np.full(
        (PATHLINE_LENGTH, SEEDS_PER_PATCH, FEATURE_COUNT),
        PADDING_VALUE,
        dtype=np.float32,
    )

    active = patch.contains_strictly(positions)
    active &= _outside_obstacle(positions, obstacle_center, obstacle_radius)
    initially_valid = active.copy()

    for step, time_value in enumerate(times):
        ids = np.flatnonzero(active)
        if ids.size:
            current = positions[ids]
            local = patch.to_local(current)
            velocity = flow.velocity(current, float(time_value))
            vorticity = flow.sample(flow.vorticity, current, float(time_value))
            mean_vorticity = float(np.interp(time_value, flow.times, patch_mean))

            features[step, ids, 0:2] = local
            features[step, ids, 2] = (time_value - start_time) / duration
            # Under normalized time, vorticity and velocity acquire a duration
            # factor in addition to the spatial-coordinate scale.
            features[step, ids, 3] = (
                np.abs(vorticity - mean_vorticity) * duration
            )
            features[step, ids, 4] = np.linalg.norm(
                local - local_seeds[ids], axis=1
            )
            features[step, ids, 5] = (
                velocity[:, 0] * patch.x_scale * duration
            )
            features[step, ids, 6] = (
                velocity[:, 1] * patch.y_scale * duration
            )

        if step == PATHLINE_LENGTH - 1 or not ids.size:
            continue
        dt = float(times[step + 1] - time_value)
        new_points = _rk4_step(flow, positions[ids], float(time_value), dt)
        accepted = np.all(np.isfinite(new_points), axis=1)
        accepted &= patch.contains_strictly(new_points)
        accepted &= _outside_obstacle(
            new_points, obstacle_center, obstacle_radius
        )
        positions[ids[accepted]] = new_points[accepted]
        active[ids[~accepted]] = False

    features[0, initially_valid, 4] = 0.0
    return features, seeds, initially_valid


def prepare_model_inputs(
    flow: FlowWindow,
    mode: str,
    start_time: float,
    duration: float,
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
    patch_points: int,
    patch_stride_points: int,
    tile_size: float | None,
    tile_stride: float | None,
    legacy_time_mode: str,
    model_time_start: float | None,
) -> PreparedInputs:
    """Construct all patches and dispatch to the selected preprocessing mode."""

    if mode == LEGACY_GRID_MODE:
        patches = make_legacy_grid_patches(
            flow.x, flow.y, patch_points, patch_stride_points
        )
    elif mode == CROSS_MODE:
        resolved_size = (
            float(flow.y[-1] - flow.y[0]) if tile_size is None else tile_size
        )
        resolved_stride = (
            0.5 * resolved_size if tile_stride is None else tile_stride
        )
        patches = make_cross_patches(
            flow.x, flow.y, resolved_size, resolved_stride
        )
    else:  # Kept explicit in case this function is called without argparse.
        raise ValueError(f"Unknown preprocessing mode: {mode}")

    feature_blocks: list[np.ndarray] = []
    seed_blocks: list[np.ndarray] = []
    display_valid_blocks: list[np.ndarray] = []
    for patch in patches:
        if mode == LEGACY_GRID_MODE:
            features, seeds, valid = integrate_legacy_grid_patch(
                flow,
                patch,
                start_time,
                duration,
                obstacle_center,
                obstacle_radius,
                legacy_time_mode,
                model_time_start,
            )
        else:
            features, seeds, valid = integrate_cross_patch(
                flow,
                patch,
                start_time,
                duration,
                obstacle_center,
                obstacle_radius,
            )
        validate_feature_block(features)
        feature_blocks.append(features)
        seed_blocks.append(seeds)
        display_valid_blocks.append(valid)

    return PreparedInputs(
        patches, feature_blocks, seed_blocks, display_valid_blocks
    )


# ---------------------------------------------------------------------------
# Projection, connected regions, and CSV output
# ---------------------------------------------------------------------------


def merge_duplicate_seeds(
    seed_blocks: Sequence[np.ndarray],
    score_blocks: np.ndarray,
    display_valid_blocks: Sequence[np.ndarray],
    precision: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average scores where overlapping patches place the same physical seed."""

    score_accumulator: dict[tuple[float, float], list[float]] = {}
    tile_accumulator: dict[tuple[float, float], list[int]] = {}
    for tile_id, (seeds, scores, valid) in enumerate(
        zip(seed_blocks, score_blocks, display_valid_blocks)
    ):
        for point, score, is_valid in zip(seeds, scores, valid):
            if not is_valid:
                continue
            key = (round(float(point[0]), precision), round(float(point[1]), precision))
            score_accumulator.setdefault(key, []).append(float(score))
            tile_accumulator.setdefault(key, []).append(tile_id)

    if not score_accumulator:
        raise ValueError("No valid seed points remain after projection filtering.")
    points = np.asarray(list(score_accumulator), dtype=np.float64)
    scores = np.asarray(
        [np.mean(score_accumulator[key]) for key in score_accumulator],
        dtype=np.float32,
    )
    source_tiles = np.asarray(
        [tile_accumulator[key][0] for key in score_accumulator],
        dtype=np.int32,
    )
    return points, scores, source_tiles


def _nearest_axis_indices(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    right = np.clip(np.searchsorted(axis, values), 0, len(axis) - 1)
    left = np.clip(right - 1, 0, len(axis) - 1)
    choose_left = np.abs(values - axis[left]) <= np.abs(axis[right] - values)
    return np.where(choose_left, left, right)


def build_probability_map(
    x: np.ndarray,
    y: np.ndarray,
    points: np.ndarray,
    scores: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize sparse probabilities by normalized Gaussian averaging."""

    weighted = np.zeros((len(y), len(x)), dtype=np.float64)
    weights = np.zeros_like(weighted)
    xi = _nearest_axis_indices(x, points[:, 0])
    yi = _nearest_axis_indices(y, points[:, 1])
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


def mask_obstacle(
    probability: np.ndarray,
    support: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    center: tuple[float, float],
    radius: float,
) -> None:
    """Remove the solid cylinder from probability postprocessing in place."""

    if radius <= 0.0:
        return
    grid_x, grid_y = np.meshgrid(x, y)
    inside = np.hypot(grid_x - center[0], grid_y - center[1]) <= radius
    probability[inside] = np.nan
    support[inside] = 0.0


def detect_probability_regions(
    probability: np.ndarray,
    support: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    min_pixels: int,
) -> list[RegionDetection]:
    """Find thresholded regions and compute probability-weighted centroids."""

    mask = (probability >= threshold) & (support > 1e-5)
    components, component_count = label(mask)
    detections: list[RegionDetection] = []
    for component_id in range(1, component_count + 1):
        yy, xx = np.nonzero(components == component_id)
        if len(xx) < min_pixels:
            continue
        component_scores = probability[yy, xx]
        weights = np.maximum(component_scores - threshold, 1e-6)
        detections.append(
            RegionDetection(
                x=float(np.average(x[xx], weights=weights)),
                y=float(np.average(y[yy], weights=weights)),
                peak=float(np.nanmax(component_scores)),
                pixels=len(xx),
            )
        )
    detections.sort(key=lambda item: (item.x, item.y))
    return detections


def detect_ivd_regions(
    flow: FlowWindow,
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
    threshold: float | None,
    percentile: float,
    smooth_sigma: float,
    min_pixels: int,
    obstacle_margin: float,
    border_cells: int,
) -> tuple[np.ndarray, float, float, list[RegionDetection]]:
    """Build an instantaneous IVD reference with rotation-aware components."""

    vorticity = np.asarray(flow.vorticity[0], dtype=np.float64)
    grid_x, grid_y = np.meshgrid(flow.x, flow.y)
    valid = np.isfinite(vorticity)

    if obstacle_radius > 0.0:
        distance = np.hypot(
            grid_x - obstacle_center[0], grid_y - obstacle_center[1]
        )
        valid &= distance > obstacle_margin * obstacle_radius

    if border_cells:
        if 2 * border_cells >= min(vorticity.shape):
            raise ValueError("--ivd-border-cells removes the entire flow grid.")
        valid[:border_cells, :] = False
        valid[-border_cells:, :] = False
        valid[:, :border_cells] = False
        valid[:, -border_cells:] = False
    if not np.any(valid):
        raise ValueError("No valid fluid cells remain for the IVD reference.")

    mean_vorticity = float(np.mean(vorticity[valid]))
    ivd = np.abs(vorticity - mean_vorticity)
    if smooth_sigma > 0.0:
        ivd = gaussian_filter(ivd, sigma=smooth_sigma, mode="nearest")

    resolved_threshold = (
        float(np.percentile(ivd[valid], percentile))
        if threshold is None
        else float(threshold)
    )
    if not np.isfinite(resolved_threshold) or resolved_threshold <= 0.0:
        raise ValueError("The resolved IVD threshold must be finite and positive.")

    detections: list[RegionDetection] = []
    for rotation_sign, sign_mask in (
        (1, vorticity >= mean_vorticity),
        (-1, vorticity < mean_vorticity),
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
                RegionDetection(
                    x=float(np.average(flow.x[xx], weights=weights)),
                    y=float(np.average(flow.y[yy], weights=weights)),
                    peak=float(np.max(component_ivd)),
                    pixels=len(xx),
                    rotation_sign=rotation_sign,
                )
            )

    detections.sort(key=lambda item: (item.x, item.y))
    return np.where(valid, ivd, np.nan), resolved_threshold, mean_vorticity, detections


def save_seed_scores(
    output_path: Path,
    points: np.ndarray,
    scores: np.ndarray,
    tile_ids: np.ndarray,
    threshold: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        # Keep the historical column name for downstream CSV compatibility;
        # "tile" and "patch" refer to the same source window here.
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


def save_detections(
    output_path: Path,
    detections: Sequence[RegionDetection],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "x", "y", "peak_probability", "component_pixels"])
        for detection_id, detection in enumerate(detections, 1):
            writer.writerow(
                [
                    detection_id,
                    f"{detection.x:.8g}",
                    f"{detection.y:.8g}",
                    f"{detection.peak:.8g}",
                    detection.pixels,
                ]
            )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_detection(
    output_path: Path,
    flow: FlowWindow,
    snapshot_u: np.ndarray,
    snapshot_v: np.ndarray,
    ivd: np.ndarray,
    ivd_threshold: float,
    ivd_detections: Sequence[RegionDetection],
    probability: np.ndarray,
    points: np.ndarray,
    scores: np.ndarray,
    model_detections: Sequence[RegionDetection],
    model_threshold: float,
    start_time: float,
    preprocessing: str,
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
) -> None:
    """Draw the IVD reference and transformer result in a two-panel figure."""

    speed = np.hypot(snapshot_u, snapshot_v)
    figure, (ivd_axis, model_axis) = plt.subplots(
        2, 1, figsize=(16, 8.0), sharex=True, constrained_layout=True
    )
    stream_skip = max(1, len(flow.x) // 320)
    # NetCDF may store nominally regular axes as float32.  Exact linspaces keep
    # Matplotlib's strict streamplot spacing check satisfied.
    stream_x = np.linspace(flow.x[0], flow.x[-1], len(flow.x))
    stream_y = np.linspace(flow.y[0], flow.y[-1], len(flow.y))

    def draw_streamlines(axis) -> None:
        axis.streamplot(
            stream_x[::stream_skip],
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
            f"IVD reference at t={start_time:g}: "
            f"|omega - mean(omega)| >= {ivd_threshold:.3g} "
            f"({len(ivd_detections)} connected regions)"
        ),
        aspect="equal",
    )
    if ivd_detections:
        ivd_axis.scatter([], [], marker="x", c="#00e5ff", label="counter-clockwise")
        ivd_axis.scatter([], [], marker="x", c="#ff4fd8", label="clockwise")
        ivd_axis.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ivd_colorbar = figure.colorbar(ivd_image, ax=ivd_axis, pad=0.01)
    ivd_colorbar.set_label("IVD")

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
    for detection_id, detection in enumerate(model_detections, 1):
        model_axis.scatter(
            [detection.x],
            [detection.y],
            marker="x",
            s=70,
            linewidths=2.0,
            c="red",
            zorder=7,
        )
        model_axis.annotate(
            f"M{detection_id}\n{detection.peak:.2f}",
            (detection.x, detection.y),
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
            f"VortexTransformer ({preprocessing}) at t={start_time:g} "
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


def plot_patch_layout(
    output_path: Path,
    flow: FlowWindow,
    snapshot_u: np.ndarray,
    snapshot_v: np.ndarray,
    patches: Sequence[Patch],
) -> None:
    """Write an optional diagnostic image containing patch boundaries."""

    debug_path = output_path.with_name(
        f"{output_path.stem}_patches{output_path.suffix}"
    )
    figure, axis = plt.subplots(figsize=(16, 4.2), constrained_layout=True)
    axis.pcolormesh(
        flow.x,
        flow.y,
        np.hypot(snapshot_u, snapshot_v),
        shading="auto",
    )
    for patch in patches:
        axis.add_patch(
            Rectangle(
                (patch.xmin, patch.ymin),
                patch.width,
                patch.height,
                fill=False,
                edgecolor="white",
                linewidth=0.6,
            )
        )
    axis.set(xlabel="x", ylabel="y", aspect="equal")
    figure.savefig(debug_path, dpi=180)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run best_checkpoint.pth.tar on a NetCDF flow and plot predicted "
            "vortex regions plus approximate connected-region locations."
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
        help="Deterministic traced forward paired with the checkpoint.",
    )
    parser.add_argument(
        "--preprocessing",
        choices=[LEGACY_GRID_MODE, CROSS_MODE],
        default=LEGACY_GRID_MODE,
        help=(
            "legacy-grid follows bundled validation; cross is an experimental "
            "four-arm compatibility profile (default: legacy-grid)."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/vortex_detection.png")
    )
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=Path("outputs/vortex_seed_scores.csv"),
    )
    parser.add_argument(
        "--detections-output",
        type=Path,
        default=Path("outputs/vortex_locations.csv"),
    )
    parser.add_argument("--time-index", type=int, default=1000)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help=(
            "Raw physical integration horizon (default: 0.72 legacy-grid, "
            "1.0 cross)."
        ),
    )

    legacy = parser.add_argument_group("legacy-grid preprocessing")
    legacy.add_argument(
        "--legacy-alpha",
        type=float,
        default=None,
        help="Alias for the legacy raw horizon; cannot be combined with --duration.",
    )
    legacy.add_argument(
        "--patch-points",
        type=int,
        default=None,
        help="NetCDF grid points per patch axis (checkpoint convention: 32).",
    )
    legacy.add_argument(
        "--patch-stride-points",
        type=int,
        default=None,
        help="Patch stride in grid points (bundled validation convention: 32).",
    )
    legacy.add_argument(
        "--legacy-time-mode",
        choices=[LEGACY_TIME_ANCHORED, LEGACY_TIME_REPLAY],
        default=None,
        help=(
            "anchored maps an arbitrary raw start to a representative training "
            "time; replay uses raw_time/duration for bundled-window replay."
        ),
    )
    legacy.add_argument(
        "--model-time-start",
        type=float,
        default=None,
        help=(
            "Model-time start for anchored mode "
            f"(default: {LEGACY_DEFAULT_MODEL_TIME_START:.8g})."
        ),
    )
    legacy.add_argument(
        "--legacy-ivd-reference-start-index",
        type=int,
        default=None,
        help="First source frame for the legacy curl mean (default: 800).",
    )
    legacy.add_argument(
        "--legacy-ivd-reference-end-index",
        type=int,
        default=None,
        help="Last source frame, inclusive, for the curl mean (default: 1000).",
    )
    legacy.add_argument(
        "--legacy-ivd-reference-mean",
        type=float,
        default=None,
        help="Precomputed signed-curl mean; overrides the reference frame range.",
    )

    cross = parser.add_argument_group("cross preprocessing")
    cross.add_argument(
        "--tile-size",
        type=float,
        default=None,
        help="Square physical patch size (default: full y extent).",
    )
    cross.add_argument(
        "--tile-stride",
        type=float,
        default=None,
        help="Physical sliding stride (default: half a tile).",
    )

    model = parser.add_argument_group("model postprocessing")
    model.add_argument("--threshold", type=float, default=0.5)
    model.add_argument(
        "--smooth-sigma",
        type=float,
        default=None,
        help="Probability-map Gaussian sigma (default: 1 legacy-grid, 3 cross).",
    )
    model.add_argument(
        "--min-component-pixels",
        type=int,
        default=None,
        help="Minimum connected-region size (default: 50 legacy-grid, 8 cross).",
    )

    ivd = parser.add_argument_group("instantaneous IVD reference")
    ivd.add_argument(
        "--ivd-threshold",
        type=float,
        default=None,
        help="Absolute threshold; by default --ivd-percentile is used.",
    )
    ivd.add_argument("--ivd-percentile", type=float, default=95.0)
    ivd.add_argument("--ivd-smooth-sigma", type=float, default=1.0)
    ivd.add_argument("--ivd-min-component-pixels", type=int, default=8)
    ivd.add_argument("--ivd-obstacle-margin", type=float, default=1.5)
    ivd.add_argument("--ivd-border-cells", type=int, default=2)

    runtime = parser.add_argument_group("runtime and geometry")
    runtime.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference batch size (default: 4 legacy-grid, 1 cross).",
    )
    runtime.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto"
    )
    runtime.add_argument("--obstacle-x", type=float, default=0.0)
    runtime.add_argument("--obstacle-y", type=float, default=0.0)
    runtime.add_argument(
        "--obstacle-radius",
        type=float,
        default=None,
        help="Default: use the NetCDF radius variable when present.",
    )
    runtime.add_argument(
        "--show-patches",
        "--show-tiles",
        dest="show_patches",
        action="store_true",
        help="Write an additional image showing patch boundaries.",
    )
    return parser


def _resolve_and_validate_args(args: argparse.Namespace) -> None:
    if args.preprocessing == LEGACY_GRID_MODE:
        if args.tile_size is not None or args.tile_stride is not None:
            raise ValueError(
                "--tile-size/--tile-stride apply only to --preprocessing cross."
            )
        args.patch_points = 32 if args.patch_points is None else args.patch_points
        args.patch_stride_points = (
            32 if args.patch_stride_points is None else args.patch_stride_points
        )
        if args.legacy_alpha is not None and args.duration is not None:
            raise ValueError("Use either --legacy-alpha or --duration, not both.")
        if args.legacy_alpha is not None:
            args.duration = args.legacy_alpha
        elif args.duration is None:
            args.duration = LEGACY_DEFAULT_DURATION
        args.legacy_time_mode = (
            LEGACY_TIME_ANCHORED
            if args.legacy_time_mode is None
            else args.legacy_time_mode
        )
        if args.legacy_time_mode == LEGACY_TIME_ANCHORED:
            args.model_time_start = (
                LEGACY_DEFAULT_MODEL_TIME_START
                if args.model_time_start is None
                else args.model_time_start
            )
        elif args.model_time_start is not None:
            raise ValueError(
                "--model-time-start applies only to --legacy-time-mode anchored."
            )

        reference_indices_given = (
            args.legacy_ivd_reference_start_index is not None
            or args.legacy_ivd_reference_end_index is not None
        )
        if args.legacy_ivd_reference_mean is not None and reference_indices_given:
            raise ValueError(
                "Use either --legacy-ivd-reference-mean or reference indices."
            )
        if args.legacy_ivd_reference_mean is None:
            args.legacy_ivd_reference_start_index = (
                800
                if args.legacy_ivd_reference_start_index is None
                else args.legacy_ivd_reference_start_index
            )
            args.legacy_ivd_reference_end_index = (
                1000
                if args.legacy_ivd_reference_end_index is None
                else args.legacy_ivd_reference_end_index
            )
    else:
        if args.patch_points is not None or args.patch_stride_points is not None:
            raise ValueError(
                "--patch-points/--patch-stride-points apply only to "
                "--preprocessing legacy-grid."
            )
        # These values are unused by the cross dispatcher, but keeping them
        # concrete simplifies its type contract.
        args.patch_points = 0
        args.patch_stride_points = 0
        legacy_only_values = (
            args.legacy_alpha,
            args.legacy_time_mode,
            args.model_time_start,
            args.legacy_ivd_reference_start_index,
            args.legacy_ivd_reference_end_index,
            args.legacy_ivd_reference_mean,
        )
        if any(value is not None for value in legacy_only_values):
            raise ValueError(
                "Legacy time/IVD options apply only to --preprocessing legacy-grid."
            )
        args.duration = 1.0 if args.duration is None else args.duration
        # Unused by cross mode; keep the dispatcher arguments concrete.
        args.legacy_time_mode = LEGACY_TIME_REPLAY
        args.model_time_start = None

    if args.duration <= 0.0:
        raise ValueError("--duration must be positive.")
    if args.model_time_start is not None and not np.isfinite(args.model_time_start):
        raise ValueError("--model-time-start must be finite.")
    if (
        args.legacy_ivd_reference_mean is not None
        and not np.isfinite(args.legacy_ivd_reference_mean)
    ):
        raise ValueError("--legacy-ivd-reference-mean must be finite.")
    if args.batch_size is None:
        args.batch_size = 4 if args.preprocessing == LEGACY_GRID_MODE else 1
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if args.smooth_sigma is None:
        args.smooth_sigma = 1.0 if args.preprocessing == LEGACY_GRID_MODE else 3.0
    if args.min_component_pixels is None:
        args.min_component_pixels = 50 if args.preprocessing == LEGACY_GRID_MODE else 8
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
    if args.obstacle_radius is not None and args.obstacle_radius < 0.0:
        raise ValueError("--obstacle-radius cannot be negative.")


def _print_mode_contract(args: argparse.Namespace, patch_count: int) -> None:
    if args.preprocessing == LEGACY_GRID_MODE:
        print(
            "Preprocessing: legacy-grid; 16x16 regular seeds, recovered time/"
            "velocity scaling, full-grid IVD, full-domain integration."
        )
        if args.legacy_time_mode == LEGACY_TIME_ANCHORED:
            print(
                f"Legacy time map: anchored raw duration={args.duration:g} to "
                f"model time [{args.model_time_start:g}, "
                f"{args.model_time_start + 1.0:g}]."
            )
        else:
            print(
                f"Legacy time map: replay tau=raw_time/{args.duration:g}."
            )
        print(
            "Calibration note: the legacy adapter was reconstructed from bundled "
            "validation binaries; validate on labels before using a new flow "
            "for quantitative conclusions."
        )
    else:
        print(
            "Preprocessing: experimental cross compatibility; 8x8 four-arm "
            "groups, normalized time, patch-local IVD, patch-boundary integration."
        )
    print(
        f"Prepared {patch_count} patches; every model input is "
        f"[{PATHLINE_LENGTH}, {SEEDS_PER_PATCH}, {FEATURE_COUNT}]."
    )


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _resolve_and_validate_args(args)

    reference_indices = None
    if (
        args.preprocessing == LEGACY_GRID_MODE
        and args.legacy_ivd_reference_mean is None
    ):
        reference_indices = (
            args.legacy_ivd_reference_start_index,
            args.legacy_ivd_reference_end_index,
        )
    flow, start_time, snapshot_u, snapshot_v, constants = load_flow_window(
        args.input,
        args.time_index,
        args.duration,
        legacy_reference_indices=reference_indices,
        legacy_reference_mean=(
            args.legacy_ivd_reference_mean
            if args.preprocessing == LEGACY_GRID_MODE
            else None
        ),
    )
    obstacle_center = (args.obstacle_x, args.obstacle_y)
    obstacle_radius = (
        constants.get("radius", 0.0)
        if args.obstacle_radius is None
        else args.obstacle_radius
    )
    prepared = prepare_model_inputs(
        flow=flow,
        mode=args.preprocessing,
        start_time=start_time,
        duration=args.duration,
        obstacle_center=obstacle_center,
        obstacle_radius=obstacle_radius,
        patch_points=args.patch_points,
        patch_stride_points=args.patch_stride_points,
        tile_size=args.tile_size,
        tile_stride=args.tile_stride,
        legacy_time_mode=args.legacy_time_mode,
        model_time_start=args.model_time_start,
    )

    print(
        f"Reading {args.input}: grid={len(flow.x)}x{len(flow.y)}, "
        f"t=[{start_time:g}, {start_time + args.duration:g}]"
    )
    _print_mode_contract(args, len(prepared.patches))
    if flow.legacy_ivd_reference_mean is not None:
        print(
            "Legacy IVD signed-curl reference mean: "
            f"{flow.legacy_ivd_reference_mean:.12g}."
        )

    device = choose_device(args.device)
    model, epoch = load_model(args.checkpoint, args.torchscript, device)
    print(f"Loaded checkpoint epoch {epoch} on {device}; state_dict matched strictly.")
    score_blocks = predict_batches(
        model, prepared.features, device, args.batch_size
    )
    points, scores, source_patches = merge_duplicate_seeds(
        prepared.seeds, score_blocks, prepared.display_valid
    )
    probability, support = build_probability_map(
        flow.x, flow.y, points, scores, args.smooth_sigma
    )
    mask_obstacle(
        probability,
        support,
        flow.x,
        flow.y,
        obstacle_center,
        obstacle_radius,
    )
    model_detections = detect_probability_regions(
        probability,
        support,
        flow.x,
        flow.y,
        args.threshold,
        args.min_component_pixels,
    )
    ivd, ivd_threshold, mean_vorticity, ivd_detections = detect_ivd_regions(
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

    save_seed_scores(
        args.scores_output, points, scores, source_patches, args.threshold
    )
    save_detections(args.detections_output, model_detections)
    plot_detection(
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
        args.preprocessing,
        obstacle_center,
        obstacle_radius,
    )
    if args.show_patches:
        plot_patch_layout(
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
