#!/usr/bin/env python3
"""Visual comparison of six vortex detectors on jungtelziemniak2d.nc.

Implemented detectors are IVD, Q, the 2-D Delta discriminant, lambda2,
VortexTransformer, and FGMA-VT.  The neural inputs contain 16 time samples,
256 pathlines, and seven features ordered as [x, y, t, IVD, distance, u, v].

The rectangular flow domain is covered by overlapping square inference tiles.
Every tile is mapped to the model domain [-1, 1]^2; predictions in overlaps
are smoothly blended back onto the NetCDF grid.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import torch
import torch.nn.functional as torch_f
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

import comparison_plot_style as plot_style
from VatistasFlowDatasetGenerator import (
    FEATURE_ADAPTIVE_MESH_SEEDING,
    GRID_CROSS_SEEDING,
    _pathline_seedings,
)


LINE_LENGTH = 16
COUNT_K = 16
LINE_COUNT = 256
CHANNEL_COUNT = 7
PAD = -1000.0


@dataclass(frozen=True)
class TimeWindow:
    x: np.ndarray
    y: np.ndarray
    times: np.ndarray
    velocity: np.ndarray  # [T,Y,X,2], only frames surrounding the requested window
    snapshot: np.ndarray  # [Y,X,2] at the requested time index
    snapshot_time: float


@dataclass(frozen=True)
class Tile:
    number: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin


@dataclass(frozen=True)
class Result:
    label: str
    values: np.ndarray
    detected: np.ndarray
    rule: str
    samples_xy: np.ndarray | None = None
    sample_probability: np.ndarray | None = None


def _axis_name(names: Iterable[str], choices: set[str]) -> str:
    match = next((name for name in names if name.lower() in choices), None)
    if match is None:
        raise ValueError(f"Missing a coordinate dimension matching {sorted(choices)}.")
    return match


def _read_velocity_slice(
    dataset: netCDF4.Dataset,
    variable_name: str,
    selection: slice | int,
    time_name: str,
    y_name: str,
    x_name: str,
) -> np.ndarray:
    variable = dataset.variables[variable_name]
    expected = (time_name, y_name, x_name)
    if tuple(variable.dimensions) != expected:
        raise ValueError(
            f"{variable_name!r} dimensions are {variable.dimensions}; expected {expected}."
        )
    data = np.ma.filled(variable[selection], np.nan).astype(np.float32)
    return data


def load_time_window(path: Path, time_index: int, duration: float) -> TimeWindow:
    """Read only the frames required for one learned-model time window."""

    if not path.is_file():
        raise FileNotFoundError(path)
    if duration <= 0.0:
        raise ValueError("--duration must be positive.")
    with netCDF4.Dataset(path, "r") as dataset:
        dimension_names = tuple(dataset.dimensions)
        x_name = _axis_name(dimension_names, {"x", "xdim"})
        y_name = _axis_name(dimension_names, {"y", "ydim"})
        time_name = _axis_name(dimension_names, {"t", "time", "tdim"})
        if "u" not in dataset.variables or "v" not in dataset.variables:
            raise ValueError("NetCDF input must contain 'u' and 'v' variables.")

        x = np.asarray(dataset.variables[x_name][:], dtype=np.float64)
        y = np.asarray(dataset.variables[y_name][:], dtype=np.float64)
        full_times = np.asarray(dataset.variables[time_name][:], dtype=np.float64)
        if time_index < 0 or time_index >= len(full_times):
            raise IndexError(f"--time-index must be in [0,{len(full_times) - 1}].")
        start = float(full_times[time_index])
        finish = start + duration
        if finish > full_times[-1]:
            raise ValueError(
                f"Requested window ends at {finish:g}, after the final data time "
                f"{full_times[-1]:g}."
            )
        lower = max(0, int(np.searchsorted(full_times, start, side="right")) - 1)
        upper = min(len(full_times), int(np.searchsorted(full_times, finish, side="left")) + 1)
        frame_slice = slice(lower, upper)
        u = _read_velocity_slice(dataset, "u", frame_slice, time_name, y_name, x_name)
        v = _read_velocity_slice(dataset, "v", frame_slice, time_name, y_name, x_name)
        snapshot_u = _read_velocity_slice(dataset, "u", time_index, time_name, y_name, x_name)
        snapshot_v = _read_velocity_slice(dataset, "v", time_index, time_name, y_name, x_name)

    if x[0] > x[-1]:
        x, u, v, snapshot_u, snapshot_v = (
            x[::-1], u[..., ::-1], v[..., ::-1], snapshot_u[..., ::-1], snapshot_v[..., ::-1]
        )
    if y[0] > y[-1]:
        y, u, v, snapshot_u, snapshot_v = (
            y[::-1], u[:, ::-1], v[:, ::-1], snapshot_u[::-1], snapshot_v[::-1]
        )
    times = full_times[frame_slice]
    if times[0] > times[-1]:
        times, u, v = times[::-1], u[::-1], v[::-1]

    for axis_name, coordinates in (("x", x), ("y", y), ("time", times)):
        if coordinates.ndim != 1 or np.any(np.diff(coordinates) <= 0.0):
            raise ValueError(f"The {axis_name} coordinates must be ascending and unique.")
    velocity = np.stack((u, v), axis=-1)
    snapshot = np.stack((snapshot_u, snapshot_v), axis=-1)
    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(snapshot)):
        raise ValueError("Velocity data contain NaN or infinite values.")
    return TimeWindow(x, y, times, velocity, snapshot, start)


def instantaneous_criteria(
    window: TimeWindow,
    smooth_sigma: float,
    ivd_percentile: float = 97.0,
    core_percentile: float = 85.0,
) -> list[Result]:
    """Evaluate robust vortex-core masks on the physical NetCDF grid.

    A bare zero crossing is retained as the sign requirement for Q, Delta and
    lambda2, but is not sufficient on this strongly sheared field.  Among
    values with the correct sign, only the strongest ``core_percentile`` tail
    is classified as a vortex core.
    """

    u = window.snapshot[..., 0].astype(np.float64)
    v = window.snapshot[..., 1].astype(np.float64)
    if smooth_sigma > 0.0:
        u = gaussian_filter(u, smooth_sigma, mode="nearest")
        v = gaussian_filter(v, smooth_sigma, mode="nearest")
    du_dy, du_dx = np.gradient(u, window.y, window.x, edge_order=2)
    dv_dy, dv_dx = np.gradient(v, window.y, window.x, edge_order=2)

    curl = dv_dx - du_dy
    ivd = np.abs(curl - float(np.mean(curl)))
    sxx = du_dx
    syy = dv_dy
    sxy = 0.5 * (du_dy + dv_dx)
    rotation = 0.5 * (du_dy - dv_dx)
    q = rotation**2 - 0.5 * (sxx**2 + syy**2 + 2.0 * sxy**2)

    gradient_trace = du_dx + dv_dy
    gradient_det = du_dx * dv_dy - du_dy * dv_dx
    delta = 4.0 * gradient_det - gradient_trace**2

    # Largest eigenvalue of S^2 + Omega^2: the 2-D lambda2 analogue.
    tensor_11 = sxx**2 + sxy**2 - rotation**2
    tensor_22 = syy**2 + sxy**2 - rotation**2
    tensor_12 = sxy * (sxx + syy)
    radius = np.sqrt((0.5 * (tensor_11 - tensor_22)) ** 2 + tensor_12**2)
    lambda2 = 0.5 * (tensor_11 + tensor_22) + radius

    ivd_cut = float(np.percentile(ivd, ivd_percentile))
    numerical_cut = 1e-5 * max(1.0, float(np.max(np.abs(curl))))
    ivd_cut = max(ivd_cut, numerical_cut)
    positive_q = q[q > 0.0]
    positive_delta = delta[delta > 0.0]
    negative_lambda_magnitude = -lambda2[lambda2 < 0.0]
    if not len(positive_q) or not len(positive_delta) or not len(negative_lambda_magnitude):
        raise RuntimeError("A classical criterion has no correctly signed response values.")
    q_cut = float(np.percentile(positive_q, core_percentile))
    delta_cut = float(np.percentile(positive_delta, core_percentile))
    lambda_magnitude_cut = float(
        np.percentile(negative_lambda_magnitude, core_percentile)
    )
    return [
        Result("IVD", ivd, ivd >= ivd_cut, f">= P{ivd_percentile:g} ({ivd_cut:.3g})"),
        Result(
            "Q",
            q,
            q >= q_cut,
            f">= positive P{core_percentile:g} ({q_cut:.3g})",
        ),
        Result(
            "Delta",
            delta,
            delta >= delta_cut,
            f">= positive P{core_percentile:g} ({delta_cut:.3g})",
        ),
        Result(
            "lambda2",
            lambda2,
            lambda2 <= -lambda_magnitude_cut,
            f"<= {-lambda_magnitude_cut:.3g}",
        ),
    ]


def _covering_starts(low: float, high: float, size: float, overlap: float) -> list[float]:
    extent = high - low
    if size > extent + 1e-10:
        raise ValueError(f"Tile size {size:g} exceeds domain extent {extent:g}.")
    last = high - size
    if np.isclose(last, low):
        return [low]
    stride = size * (1.0 - overlap)
    starts = list(np.arange(low, last + 0.5 * stride, stride, dtype=float))
    starts = [value for value in starts if value < last - 1e-10]
    starts.append(last)
    return starts


def make_tiles(window: TimeWindow, size: float, overlap: float) -> list[Tile]:
    if size <= 0.0:
        raise ValueError("--patch-size must be positive.")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("--patch-overlap must lie in [0,1).")
    x_starts = _covering_starts(window.x[0], window.x[-1], size, overlap)
    y_starts = _covering_starts(window.y[0], window.y[-1], size, overlap)
    tiles: list[Tile] = []
    for ymin in y_starts:
        for xmin in x_starts:
            tiles.append(Tile(len(tiles), xmin, xmin + size, ymin, ymin + size))
    return tiles


class TileVelocity:
    """Space/time-normalized differentiable interpolator for one square tile."""

    def __init__(self, window: TimeWindow, tile: Tile, duration: float):
        if not np.isclose(tile.width, tile.height):
            raise ValueError("Learned inference tiles must be square.")
        for axis_name, coordinates in (("x", window.x), ("y", window.y)):
            spacing = np.diff(coordinates)
            # Coordinates are stored as float32, so a mathematically regular
            # linspace has small last-bit spacing variations after loading.
            if not np.allclose(spacing, spacing[0], rtol=1e-4, atol=1e-10):
                raise ValueError(f"Learned inference requires a regular {axis_name}-axis.")
        self.window = window
        self.tile = tile
        self.duration = float(duration)
        self._frames = torch.from_numpy(window.velocity).permute(0, 3, 1, 2)

    def _frame(self, tau: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        physical_time = self.window.snapshot_time + float(tau) * self.duration
        upper = int(np.searchsorted(self.window.times, physical_time, side="right"))
        upper = min(max(upper, 1), len(self.window.times) - 1)
        lower = upper - 1
        alpha = (physical_time - self.window.times[lower]) / (
            self.window.times[upper] - self.window.times[lower]
        )
        pair = self._frames[[lower, upper]].to(device=device, dtype=dtype)
        return (1.0 - float(alpha)) * pair[0:1] + float(alpha) * pair[1:2]

    def torch_eval(self, local_points: torch.Tensor, tau: float) -> torch.Tensor:
        if local_points.shape[-1] != 2:
            raise ValueError("Pathline points must be two-dimensional.")
        physical_x = self.tile.xmin + 0.5 * (local_points[..., 0] + 1.0) * self.tile.width
        physical_y = self.tile.ymin + 0.5 * (local_points[..., 1] + 1.0) * self.tile.height
        global_x = 2.0 * (physical_x - self.window.x[0]) / (
            self.window.x[-1] - self.window.x[0]
        ) - 1.0
        global_y = 2.0 * (physical_y - self.window.y[0]) / (
            self.window.y[-1] - self.window.y[0]
        ) - 1.0
        sampling_grid = torch.stack((global_x, global_y), dim=-1).reshape(1, -1, 1, 2)
        sampled = torch_f.grid_sample(
            self._frame(tau, local_points.device, local_points.dtype),
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        physical_velocity = sampled[0, :, :, 0].transpose(0, 1).reshape(local_points.shape)
        normalized_scale = 2.0 * self.duration / self.tile.width
        return physical_velocity * normalized_scale

    def __call__(self, points: np.ndarray, tau: float) -> np.ndarray:
        with torch.no_grad():
            output = self.torch_eval(torch.as_tensor(points, dtype=torch.float32), tau)
        return output.numpy().astype(np.float64)

    def curl(self, points: np.ndarray, tau: float) -> np.ndarray:
        coordinates = torch.as_tensor(points, dtype=torch.float32).detach().requires_grad_(True)
        velocity = self.torch_eval(coordinates, tau)
        grad_u = torch.autograd.grad(velocity[..., 0].sum(), coordinates, retain_graph=True)[0]
        grad_v = torch.autograd.grad(velocity[..., 1].sum(), coordinates)[0]
        return (grad_v[..., 0] - grad_u[..., 1]).detach().numpy().astype(np.float64)

    def average_curl(self, tau: float, count: int = 64) -> float:
        coordinates = torch.linspace(-1.0, 1.0, count)
        bottom = torch.stack((coordinates, torch.full_like(coordinates, -1.0)), dim=-1)
        right = torch.stack((torch.full_like(coordinates, 1.0), coordinates), dim=-1)
        top = torch.stack((coordinates, torch.full_like(coordinates, 1.0)), dim=-1)
        left = torch.stack((torch.full_like(coordinates, -1.0), coordinates), dim=-1)
        with torch.no_grad():
            circulation = (
                torch.trapezoid(self.torch_eval(bottom, tau)[:, 0], coordinates)
                + torch.trapezoid(self.torch_eval(right, tau)[:, 1], coordinates)
                - torch.trapezoid(self.torch_eval(top, tau)[:, 0], coordinates)
                - torch.trapezoid(self.torch_eval(left, tau)[:, 1], coordinates)
            )
        return float(circulation / 4.0)


def _valid_local(points: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(points), axis=1) & np.all(
        (points > -1.0) & (points < 1.0), axis=1
    )


def build_model_input(
    evaluator: TileVelocity,
    sampling_method: str,
    seed_value: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate seeds, integrate with RK4, and assemble seven input channels."""

    seeding_options = {
        "sampling": sampling_method,
        "count_k": COUNT_K,
        "jitter_std": 0.05,
        "feature_high_fraction": 0.80,
        "feature_high_vorticity_percentile": 70.0,
        "feature_mesh_resolution": 64,
        "feature_mesh_size_ratio": 4.0,
        "feature_monitor_power": 1.5,
        "feature_mesh_iterations": 5,
        "feature_clip_percentile": 99.0,
        "feature_boundary_margin": 1e-6,
    }
    random = np.random.default_rng(seed_value)
    seeds = _pathline_seedings(
        seeding_options, random, velocity_eval=evaluator, time=0.0
    )
    if seeds.shape != (LINE_COUNT, 2):
        raise RuntimeError(f"Seeding produced {seeds.shape}, expected ({LINE_COUNT},2).")

    tau = np.linspace(0.0, 1.0, LINE_LENGTH)
    tracks = np.full((LINE_LENGTH, LINE_COUNT, 2), np.nan, dtype=np.float64)
    tracks[0] = seeds
    active = _valid_local(seeds)
    for step in range(LINE_LENGTH - 1):
        line_ids = np.flatnonzero(active)
        if not len(line_ids):
            break
        point = tracks[step, line_ids]
        time = float(tau[step])
        dt = float(tau[step + 1] - tau[step])
        k1 = evaluator(point, time)
        k2 = evaluator(point + 0.5 * dt * k1, time + 0.5 * dt)
        k3 = evaluator(point + 0.5 * dt * k2, time + 0.5 * dt)
        k4 = evaluator(point + dt * k3, time + dt)
        candidate = point + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        accepted = _valid_local(candidate)
        tracks[step + 1, line_ids[accepted]] = candidate[accepted]
        active[line_ids[~accepted]] = False

    features = np.full((LINE_LENGTH, LINE_COUNT, CHANNEL_COUNT), PAD, dtype=np.float32)
    for step, time in enumerate(tau):
        valid = np.isfinite(tracks[step, :, 0])
        if not np.any(valid):
            continue
        positions = tracks[step, valid]
        velocity = evaluator(positions, float(time))
        local_curl = evaluator.curl(positions, float(time))
        features[step, valid, 0:2] = positions
        features[step, valid, 2] = time
        features[step, valid, 3] = np.abs(
            local_curl - evaluator.average_curl(float(time))
        )
        features[step, valid, 4] = np.linalg.norm(positions - seeds[valid], axis=1)
        features[step, valid, 5:7] = velocity
    features[0, :, 4] = 0.0
    return features, seeds


def find_checkpoint(requested: Path, alternatives: Iterable[Path], kind: str) -> Path:
    for candidate in (requested, *alternatives):
        if candidate.is_file():
            if candidate != requested:
                warnings.warn(f"{requested} not found; using {candidate} for {kind}.", stacklevel=2)
            return candidate
    attempted = "\n".join(f"  - {candidate}" for candidate in (requested, *alternatives))
    raise FileNotFoundError(f"No {kind} found. Attempted:\n{attempted}")


def prepare_network(
    archive_path: Path,
    traced_path: Path,
    device: torch.device,
) -> tuple[torch.jit.ScriptModule, int | str]:
    """Strictly combine a checkpoint archive with its traced architecture."""

    archive = torch.load(archive_path, map_location="cpu", weights_only=False)
    if not isinstance(archive, dict) or not isinstance(archive.get("state_dict"), dict):
        raise TypeError(f"{archive_path} does not contain a state_dict checkpoint.")
    weights = {
        (name.removeprefix("module.")): tensor
        for name, tensor in archive["state_dict"].items()
    }
    network = torch.jit.load(str(traced_path), map_location="cpu")
    targets = [network]
    if getattr(network, "model", None) is not None:
        targets.append(network.model)

    mismatch: list[str] = []
    for target in targets:
        expected = set(target.state_dict())
        supplied = set(weights)
        if expected == supplied:
            target.load_state_dict(weights, strict=True)
            network.eval().to(device)
            return network, archive.get("epoch", "unknown")
        mismatch.append(
            f"missing={sorted(expected - supplied)[:3]}, "
            f"unexpected={sorted(supplied - expected)[:3]}"
        )
    raise RuntimeError(
        f"Checkpoint {archive_path} is incompatible with {traced_path}: "
        + "; ".join(mismatch)
    )


def predict_batch(
    model: torch.jit.ScriptModule,
    batch: list[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    tensor = torch.from_numpy(np.stack(batch)).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        probability = model(tensor).detach().cpu().numpy()
    expected = (len(batch), LINE_COUNT)
    if probability.shape != expected:
        raise RuntimeError(f"Model output is {probability.shape}, expected {expected}.")
    if not np.all(np.isfinite(probability)):
        raise RuntimeError("Model returned non-finite probabilities.")
    return probability


def seeds_to_physical(tile: Tile, seeds: np.ndarray) -> np.ndarray:
    points = np.empty_like(seeds)
    points[:, 0] = tile.xmin + 0.5 * (seeds[:, 0] + 1.0) * tile.width
    points[:, 1] = tile.ymin + 0.5 * (seeds[:, 1] + 1.0) * tile.height
    return points


def merge_tile_predictions(
    window: TimeWindow,
    tiles: list[Tile],
    local_seeds: list[np.ndarray],
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate scattered outputs per tile and blend overlapping tiles."""

    accumulated = np.zeros((len(window.y), len(window.x)), dtype=np.float64)
    total_weight = np.zeros_like(accumulated)
    all_points: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    for tile, seeds, tile_scores in zip(tiles, local_seeds, scores):
        physical_seeds = seeds_to_physical(tile, seeds)
        all_points.append(physical_seeds)
        all_scores.append(tile_scores)
        x_ids = np.flatnonzero((window.x >= tile.xmin) & (window.x <= tile.xmax))
        y_ids = np.flatnonzero((window.y >= tile.ymin) & (window.y <= tile.ymax))
        local_x, local_y = np.meshgrid(window.x[x_ids], window.y[y_ids])
        interpolated = griddata(
            physical_seeds, tile_scores, (local_x, local_y), method="linear"
        )
        nearest = griddata(
            physical_seeds, tile_scores, (local_x, local_y), method="nearest"
        )
        interpolated = np.where(np.isfinite(interpolated), interpolated, nearest)

        norm_x = 2.0 * (local_x - tile.xmin) / tile.width - 1.0
        norm_y = 2.0 * (local_y - tile.ymin) / tile.height - 1.0
        # A nonzero edge floor covers the global boundary; cosine tapering
        # gives central predictions more influence in tile overlaps.
        blend = 0.05 + 0.95 * np.cos(0.5 * np.pi * np.abs(norm_x)) * np.cos(
            0.5 * np.pi * np.abs(norm_y)
        )
        target = np.ix_(y_ids, x_ids)
        accumulated[target] += interpolated * blend
        total_weight[target] += blend

    if np.any(total_weight <= 0.0):
        raise RuntimeError("Inference tiles did not cover every output grid point.")
    merged = np.clip(accumulated / total_weight, 0.0, 1.0)
    return merged, np.concatenate(all_points), np.concatenate(all_scores)


def neural_result(
    label: str,
    sampling: str,
    window: TimeWindow,
    tiles: list[Tile],
    duration: float,
    archive: Path,
    traced: Path,
    device: torch.device,
    threshold: float,
    random_seed: int,
) -> tuple[Result, int | str, float]:
    inputs: list[np.ndarray] = []
    seed_sets: list[np.ndarray] = []
    valid_fractions: list[float] = []
    for tile in tiles:
        features, seeds = build_model_input(
            TileVelocity(window, tile, duration), sampling, random_seed + tile.number
        )
        inputs.append(features)
        seed_sets.append(seeds)
        valid_fractions.append(float(np.mean(features[..., 0] > -999.0)))
    network, epoch = prepare_network(archive, traced, device)
    scores = predict_batch(network, inputs, device)
    field, points, point_scores = merge_tile_predictions(window, tiles, seed_sets, scores)
    return (
        Result(label, field, field >= threshold, f">= {threshold:g}", points, point_scores),
        epoch,
        float(np.mean(valid_fractions)),
    )


def _positive_norm(values: np.ndarray) -> Normalize:
    upper = float(np.percentile(values, 99.0))
    return Normalize(vmin=0.0, vmax=max(upper, np.finfo(float).eps))


def draw_comparison(
    window: TimeWindow,
    results: list[Result],
    tiles: list[Tile],
    output: Path,
    show: bool,
    dataset_label: str = "jungtelziemniak2d",
) -> None:
    xx, yy = np.meshgrid(window.x, window.y)
    # Matplotlib checks equal spacing more strictly than the float32 NetCDF
    # coordinates permit; these linspaces represent the same regular grid.
    stream_x = np.linspace(window.x[0], window.x[-1], len(window.x))
    stream_y = np.linspace(window.y[0], window.y[-1], len(window.y))
    columns = 4 if len(results) > 6 else 3
    figure, axes = plt.subplots(
        2, columns, figsize=(6.3 * columns, 9), constrained_layout=True
    )
    learned = {"VortexTransformer", "FGMA-VT", "MVUNet", "VortexUNet"}
    field_cmap = plot_style.FIELD_CMAP
    background_color = plot_style.BACKGROUND_COLOR
    streamline_sets = []

    for axis, result in zip(axes.flat, results):
        axis.set_facecolor(background_color)
        if result.label in learned:
            display = np.maximum(result.values, 0.0)
            image = axis.pcolormesh(
                xx, yy, display, cmap=field_cmap, vmin=0.0, vmax=1.0, shading="auto"
            )
        else:
            score = -result.values if result.label == "lambda2" else result.values
            display = np.maximum(score, 0.0)
            image = axis.pcolormesh(
                xx,
                yy,
                display,
                cmap=field_cmap,
                norm=_positive_norm(display),
                shading="auto",
            )
        if np.any(result.detected):
            axis.contourf(
                xx,
                yy,
                result.detected.astype(float),
                levels=[0.5, 1.5],
                colors=["none"],
                hatches=["////"],
                alpha=0.0,
            )
            if np.any(~result.detected):
                axis.contour(
                    xx,
                    yy,
                    result.detected.astype(float),
                    levels=[0.5],
                    colors=plot_style.THRESHOLD_COLOR,
                    linewidths=0.7,
                )
        if result.samples_xy is not None and result.sample_probability is not None:
            axis.scatter(
                result.samples_xy[:, 0],
                result.samples_xy[:, 1],
                c=result.sample_probability,
                cmap=field_cmap,
                vmin=0.0,
                vmax=1.0,
                s=2.5,
                alpha=0.35,
                linewidths=0.0,
            )
            for tile in tiles:
                axis.plot(
                    [tile.xmin, tile.xmax, tile.xmax, tile.xmin, tile.xmin],
                    [tile.ymin, tile.ymin, tile.ymax, tile.ymax, tile.ymin],
                    color="white",
                    linewidth=0.45,
                    linestyle="--",
                    alpha=0.45,
                )
        streamline_sets.append(axis.streamplot(
            stream_x,
            stream_y,
            window.snapshot[..., 0],
            window.snapshot[..., 1],
            color=plot_style.STREAMLINE_COLOR,
            density=0.55,
            linewidth=0.3,
            arrowsize=0.4,
            minlength=0.12,
        ))
        fraction = 100.0 * float(np.mean(result.detected))
        axis.set_title(f"{result.label} ({result.rule}; detected {fraction:.1f}%)")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")
        figure.colorbar(image, ax=axis, shrink=0.85, pad=0.015)
    for axis in axes.flat[len(results):]:
        axis.set_visible(False)

    figure.suptitle(
        f"{dataset_label} vortex detection at t={window.snapshot_time:.6g} "
        "(hatched = detected)",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".pdf":
        plot_style.thin_streamplots_for_pdf(streamline_sets)
    figure.savefig(output, dpi=180)
    print(f"Saved visualization: {output}")
    if show:
        plt.show()
    plt.close(figure)


def choose_device(text: str) -> torch.device:
    if text == "auto":
        text = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def parser() -> argparse.ArgumentParser:
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument(
        "--input", type=Path, default=Path("dataset/jungtelziemniak2d.nc")
    )
    arguments.add_argument(
        "--output", type=Path, default=Path("comparison_jungtelziemniak2d.png")
    )
    arguments.add_argument("--time-index", type=int, default=0)
    arguments.add_argument("--duration", type=float, default=0.05)
    arguments.add_argument("--patch-size", type=float, default=4.0)
    arguments.add_argument("--patch-overlap", type=float, default=0.5)
    arguments.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional legacy override that sets both neural thresholds.",
    )
    arguments.add_argument("--vt-threshold", type=float, default=0.55)
    arguments.add_argument("--fgma-threshold", type=float, default=0.35)
    arguments.add_argument("--ivd-percentile", type=float, default=97.0)
    arguments.add_argument(
        "--classical-core-percentile",
        type=float,
        default=85.0,
        help="Tail percentile within correctly signed Q/Delta/lambda2 responses.",
    )
    arguments.add_argument("--smoothing-sigma", type=float, default=1.0)
    arguments.add_argument("--seed", type=int, default=7068)
    arguments.add_argument("--device", default="auto")
    arguments.add_argument("--show", action="store_true")
    arguments.add_argument(
        "--vt-checkpoint",
        type=Path,
        default=Path("trainedVortexTransformer/best_checkpoint.pth.tar"),
    )
    arguments.add_argument(
        "--vt-script",
        type=Path,
        default=Path("trainedVortexTransformer/Backup_best_checkpoint.pthtraced.pt"),
    )
    arguments.add_argument(
        "--fgma-checkpoint",
        type=Path,
        default=Path("trainedVortexTransformerBetter/best_checkpoint.pth.tar"),
    )
    arguments.add_argument(
        "--fgma-script",
        type=Path,
        default=Path("trainedVortexTransformerBetter/Backup_best_checkpoint.pthtraced.pt"),
    )
    return arguments


def main() -> None:
    args = parser().parse_args()
    for option, value in (
        ("--vt-threshold", args.vt_threshold),
        ("--fgma-threshold", args.fgma_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{option} must lie in [0,1].")
    if args.threshold is not None:
        if not 0.0 <= args.threshold <= 1.0:
            raise ValueError("--threshold must lie in [0,1].")
        warnings.warn(
            "--threshold overrides both model-specific thresholds; prefer "
            "--vt-threshold and --fgma-threshold.",
            stacklevel=2,
        )
        vt_threshold = fgma_threshold = args.threshold
    else:
        vt_threshold = args.vt_threshold
        fgma_threshold = args.fgma_threshold
    if not 0.0 < args.ivd_percentile < 100.0:
        raise ValueError("--ivd-percentile must lie in (0,100).")
    if not 0.0 < args.classical_core_percentile < 100.0:
        raise ValueError("--classical-core-percentile must lie in (0,100).")
    device = choose_device(args.device)
    window = load_time_window(args.input, args.time_index, args.duration)
    tiles = make_tiles(window, args.patch_size, args.patch_overlap)

    vt_checkpoint = find_checkpoint(args.vt_checkpoint, (), "VortexTransformer checkpoint")
    vt_script = find_checkpoint(
        args.vt_script,
        (Path("trainedVortexTransformer/Backup_best_checkpoint.pthtraced.pt"),),
        "VortexTransformer TorchScript",
    )
    fgma_checkpoint = find_checkpoint(
        args.fgma_checkpoint,
        (Path("trainedVortexTransformerBetter/best_checkpoint.pth.tar"),),
        "FGMA-VT checkpoint",
    )
    fgma_script = find_checkpoint(
        args.fgma_script,
        (Path("trainedVortexTransformerBetter/Backup_best_checkpoint.pthtraced.pt"),),
        "FGMA-VT TorchScript",
    )

    results = instantaneous_criteria(
        window,
        args.smoothing_sigma,
        args.ivd_percentile,
        args.classical_core_percentile,
    )
    print(f"Using {len(tiles)} overlapping square inference tiles.")
    print(f"Running VortexTransformer with {GRID_CROSS_SEEDING}...")
    vt, vt_epoch, vt_valid = neural_result(
        "VortexTransformer",
        GRID_CROSS_SEEDING,
        window,
        tiles,
        args.duration,
        vt_checkpoint,
        vt_script,
        device,
        vt_threshold,
        args.seed,
    )
    print(f"Running FGMA-VT with {FEATURE_ADAPTIVE_MESH_SEEDING}...")
    fgma, fgma_epoch, fgma_valid = neural_result(
        "FGMA-VT",
        FEATURE_ADAPTIVE_MESH_SEEDING,
        window,
        tiles,
        args.duration,
        fgma_checkpoint,
        fgma_script,
        device,
        fgma_threshold,
        args.seed + 1000,
    )
    results.extend((vt, fgma))
    print(
        f"VortexTransformer epoch={vt_epoch}, valid pathline samples={vt_valid:.1%}; "
        f"FGMA-VT epoch={fgma_epoch}, valid pathline samples={fgma_valid:.1%}."
    )
    draw_comparison(window, results, tiles, args.output, args.show)


if __name__ == "__main__":
    main()
