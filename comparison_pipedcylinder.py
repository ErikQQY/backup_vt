#!/usr/bin/env python3
"""Eight-way vortex detection comparison for ``pipedcylinder2d.nc``.

Implemented methods are IVD, Q, Delta, lambda2, MVUNet, VortexUNet,
VortexTransformer, and FGMA-VT. The transformer inputs contain 16 time samples,
and seven channels: [x, y, t, IVD, distance, u, v].  Each checkpoint is run
with the preprocessing convention used for training: VortexTransformer uses
the recovered legacy regular grid and FGMA-VT uses
FEATURE_ADAPTIVE_MESH_SEEDING from VatistasFlowDatasetGenerator.
MVUNet receives the generator-specific dense [u,v,curl,IVD] t0 grid and keeps
GRID_CROSS_SEEDING as its training-data provenance.
VortexUNet receives the generator-specific dense [u,v] t0 grid with the same
GRID_CROSS_SEEDING training-data provenance.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib
import netCDF4 as nc
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, gaussian_filter, label
from scipy.optimize import linear_sum_assignment

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import comparison_plot_style as plot_style
import detect as legacy_vt
import mvunet_comparison as mvunet
import vortexunet_comparison as vortexunet
from VatistasFlowDatasetGenerator import (
    FEATURE_ADAPTIVE_MESH_SEEDING,
    _pathline_seedings,
)

LENGTH, COUNT_K, SEED_COUNT, CHANNELS = 16, 16, 256, 7
PADDING = -1000.0


@dataclass(frozen=True)
class Patch:
    index: int
    xs: int
    xe: int
    ys: int
    ye: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    def to_physical(self, points):
        points = np.asarray(points, dtype=np.float64)
        return np.column_stack((
            self.xmin + (points[:, 0] + 1.0) * self.width / 2.0,
            self.ymin + (points[:, 1] + 1.0) * self.height / 2.0,
        ))


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    peak: float
    pixels: int
    rotation_sign: int


@dataclass
class Result:
    name: str
    score: np.ndarray
    threshold: float
    mask: np.ndarray
    detections: list[Detection]
    seeding: str = "-"


class Flow:
    """Velocity time window, gradients, interpolation, and pipe mask."""

    def __init__(self, x, y, times, u, v, radius):
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.times = np.asarray(times, dtype=np.float64)
        self.u = np.asarray(u, dtype=np.float32)
        self.v = np.asarray(v, dtype=np.float32)
        self.radius = float(radius)
        expected = (len(times), len(y), len(x))
        if self.u.shape != expected or self.v.shape != expected:
            raise ValueError(f"Velocity shape must be {expected}.")
        if min(len(x), len(y)) < 3 or len(times) < 2:
            raise ValueError("At least 3x3 spatial samples and two frames are required.")

        self.ux = np.gradient(self.u, self.x, axis=2, edge_order=2)
        self.uy = np.gradient(self.u, self.y, axis=1, edge_order=2)
        self.vx = np.gradient(self.v, self.x, axis=2, edge_order=2)
        self.vy = np.gradient(self.v, self.y, axis=1, edge_order=2)
        self.curl = self.vx - self.uy

        # Solids are exactly zero for the complete time window. Erosion removes
        # finite-difference contamination at walls and cylinder boundaries.
        self.fluid = np.any(np.hypot(self.u, self.v) > 1e-9, axis=0)
        self.analysis_fluid = binary_erosion(self.fluid, iterations=2)
        self.grid_x, self.grid_y = np.meshgrid(self.x, self.y)
        self.cylinders = ((0.0, 0.0), (3.0, 1.0))

    def time_bracket(self, time):
        hi = min(max(int(np.searchsorted(self.times, time, side="right")), 1),
                 len(self.times) - 1)
        lo = hi - 1
        weight = (float(time) - self.times[lo]) / (self.times[hi] - self.times[lo])
        return lo, hi, float(np.clip(weight, 0.0, 1.0))

    @staticmethod
    def axis_bracket(axis, values):
        hi = np.clip(np.searchsorted(axis, values, side="right"), 1, len(axis)-1)
        lo = hi - 1
        weight = (values - axis[lo]) / (axis[hi] - axis[lo])
        return lo, hi, np.clip(weight, 0.0, 1.0)

    def sample(self, field, points, time):
        points = np.asarray(points, dtype=np.float64)
        x0, x1, wx = self.axis_bracket(self.x, points[:, 0])
        y0, y1, wy = self.axis_bracket(self.y, points[:, 1])
        t0, t1, wt = self.time_bracket(time)
        def bilinear(frame):
            lower = frame[y0, x0]*(1-wx) + frame[y0, x1]*wx
            upper = frame[y1, x0]*(1-wx) + frame[y1, x1]*wx
            return lower*(1-wy) + upper*wy
        return bilinear(field[t0])*(1-wt) + bilinear(field[t1])*wt

    def sample_static(self, field, points):
        points = np.asarray(points, dtype=np.float64)
        x0, x1, wx = self.axis_bracket(self.x, points[:, 0])
        y0, y1, wy = self.axis_bracket(self.y, points[:, 1])
        lower = field[y0, x0]*(1-wx) + field[y0, x1]*wx
        upper = field[y1, x0]*(1-wx) + field[y1, x1]*wx
        return lower*(1-wy) + upper*wy

    def velocity(self, points, time):
        return np.column_stack((self.sample(self.u, points, time),
                                self.sample(self.v, points, time)))

    def points_are_fluid(self, points):
        points = np.asarray(points, dtype=np.float64)
        inside = ((points[:, 0] > self.x[0]) & (points[:, 0] < self.x[-1]) &
                  (points[:, 1] > self.y[0]) & (points[:, 1] < self.y[-1]))
        result = np.zeros(len(points), dtype=bool)
        if np.any(inside):
            result[inside] = self.sample_static(
                self.fluid.astype(np.float32), points[inside]) > 0.7
        return result

    def local_curl(self, patch, local_points, time):
        physical = patch.to_physical(local_points)
        return ((patch.width/patch.height)*self.sample(self.vx, physical, time)
                - (patch.height/patch.width)*self.sample(self.uy, physical, time))

    def local_mean_curl(self, patch, time):
        valid = self.analysis_fluid[patch.ys:patch.ye, patch.xs:patch.xe]
        if not np.any(valid):
            return 0.0
        lo, hi, weight = self.time_bracket(time)
        def mean(frame):
            vx = self.vx[frame, patch.ys:patch.ye, patch.xs:patch.xe]
            uy = self.uy[frame, patch.ys:patch.ye, patch.xs:patch.xe]
            curl = (patch.width/patch.height)*vx - (patch.height/patch.width)*uy
            return float(np.mean(curl[valid], dtype=np.float64))
        return mean(lo)*(1-weight) + mean(hi)*weight


class DifferentiableVelocity:
    """Bilinear local velocity used by feature-adaptive mesh seeding."""
    device, dtype = torch.device("cpu"), torch.float32

    def __init__(self, parent):
        self.parent = parent
        self.u = torch.from_numpy(parent.flow.u)
        self.v = torch.from_numpy(parent.flow.v)

    def __call__(self, local_points, time):
        flow, patch = self.parent.flow, self.parent.patch
        original_shape = local_points.shape
        points = local_points.reshape(-1, 2)
        px = patch.xmin + (points[:, 0]+1)*patch.width/2
        py = patch.ymin + (points[:, 1]+1)*patch.height/2
        gx = 2*(px-flow.x[0])/(flow.x[-1]-flow.x[0])-1
        gy = 2*(py-flow.y[0])/(flow.y[-1]-flow.y[0])-1
        grid = torch.stack((gx, gy), dim=-1).reshape(1, 1, -1, 2)
        lo, hi, weight = flow.time_bracket(float(time))
        def sample(frame):
            image = torch.stack((self.u[frame], self.v[frame]))[None]
            return F.grid_sample(image, grid, align_corners=True,
                                 padding_mode="border")[0, :, 0].T
        velocity = sample(lo)*(1-weight) + sample(hi)*weight
        scale = velocity.new_tensor((2/patch.width, 2/patch.height))
        return (velocity*scale).reshape(original_shape)


class LocalVelocity:
    def __init__(self, flow, patch):
        self.flow, self.patch = flow, patch
        self.torch_eval = DifferentiableVelocity(self)

    def __call__(self, local_points, time):
        physical = self.patch.to_physical(local_points)
        velocity = self.flow.velocity(physical, time)
        return velocity*np.asarray((2/self.patch.width, 2/self.patch.height))


def load_flow(path, time_index, duration):
    with nc.Dataset(path) as dataset:
        required = ("u", "v", "xdim", "ydim", "tdim")
        missing = [name for name in required if name not in dataset.variables]
        if missing:
            raise ValueError(f"Missing variables in {path}: {missing}")
        x, y = dataset["xdim"][:], dataset["ydim"][:]
        all_times = np.asarray(dataset["tdim"][:], dtype=np.float64)
        if not 0 <= time_index < len(all_times)-1:
            raise ValueError(f"--time-index must be in [0,{len(all_times)-2}].")
        start, end = float(all_times[time_index]), float(all_times[time_index]+duration)
        if end > all_times[-1]:
            raise ValueError(f"Integration end {end:g} exceeds dataset end {all_times[-1]:g}.")
        last = min(int(np.searchsorted(all_times, end))+1, len(all_times)-1)
        frames = slice(time_index, last+1)
        radius = float(np.asarray(dataset["radius"][:]).reshape(-1)[0]) \
            if "radius" in dataset.variables else 0.0
        flow = Flow(x, y, all_times[frames], dataset["u"][frames],
                    dataset["v"][frames], radius)
    return flow, start


def axis_starts(size, points, stride):
    if not 3 <= points <= size or stride < 1:
        raise ValueError(f"Invalid patch={points}, stride={stride}, axis={size}.")
    final = size-points
    starts = list(range(0, final+1, stride))
    if starts[-1] != final:
        starts.append(final)
    return starts


def make_patches(flow, points, stride, minimum_fluid_fraction):
    patches = []
    for ys in axis_starts(len(flow.y), points, stride):
        for xs in axis_starts(len(flow.x), points, stride):
            xe, ye = xs+points, ys+points
            fraction = float(np.mean(flow.fluid[ys:ye, xs:xe]))
            if fraction < minimum_fluid_fraction:
                continue
            patches.append(Patch(len(patches), xs, xe, ys, ye,
                float(flow.x[xs]), float(flow.x[xe-1]),
                float(flow.y[ys]), float(flow.y[ye-1])))
    if not patches:
        raise ValueError("No patches contain enough fluid cells.")
    return patches


def make_seeds(evaluator, method, start_time, rng, args):
    config = dict(count_k=COUNT_K, sampling=method, jitter_std=args.cross_jitter,
        feature_high_fraction=args.feature_high_fraction,
        feature_high_vorticity_percentile=args.feature_high_percentile,
        feature_mesh_resolution=args.feature_mesh_resolution,
        feature_mesh_size_ratio=args.feature_mesh_size_ratio,
        feature_monitor_power=args.feature_monitor_power,
        feature_mesh_iterations=args.feature_mesh_iterations,
        feature_clip_percentile=args.feature_clip_percentile,
        feature_boundary_margin=args.feature_boundary_margin)
    seeds = _pathline_seedings(config, rng, velocity_eval=evaluator, time=start_time)
    if seeds.shape != (SEED_COUNT, 2):
        raise RuntimeError(f"{method} returned shape {seeds.shape}.")
    return seeds


def pathline_features(flow, patch, seeds, start_time, duration):
    evaluator = LocalVelocity(flow, patch)
    physical_times = np.linspace(start_time, start_time+duration, LENGTH)
    model_times = np.linspace(0, 1, LENGTH)
    positions = seeds.copy()
    active = np.all((positions > -1) & (positions < 1), axis=1)
    active &= flow.points_are_fluid(patch.to_physical(positions))
    features = np.full((LENGTH, SEED_COUNT, CHANNELS), PADDING, dtype=np.float32)
    for step, time in enumerate(physical_times):
        ids = np.flatnonzero(active)
        if len(ids):
            points = positions[ids]
            features[step, ids, :2] = points
            features[step, ids, 2] = model_times[step]
            features[step, ids, 3] = np.abs(
                flow.local_curl(patch, points, time)-flow.local_mean_curl(patch, time))
            features[step, ids, 4] = np.linalg.norm(points-seeds[ids], axis=1)
            features[step, ids, 5:] = evaluator(points, time)
        if step == LENGTH-1 or not len(ids):
            continue
        dt, points = physical_times[step+1]-time, positions[ids]
        k1 = evaluator(points, time)
        k2 = evaluator(points+dt*k1/2, time+dt/2)
        k3 = evaluator(points+dt*k2/2, time+dt/2)
        k4 = evaluator(points+dt*k3, time+dt)
        new_points = points+dt*(k1+2*k2+2*k3+k4)/6
        accepted = np.all(np.isfinite(new_points), axis=1)
        accepted &= np.all((new_points > -1) & (new_points < 1), axis=1)
        accepted &= flow.points_are_fluid(patch.to_physical(new_points))
        positions[ids[accepted]] = new_points[accepted]
        active[ids[~accepted]] = False
    features[0, :, 4] = np.where(features[0, :, 0] > -999, 0.0, PADDING)
    return features


def load_model(checkpoint_path, script_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), Mapping):
        raise ValueError(f"Invalid checkpoint: {checkpoint_path}")
    state = checkpoint["state_dict"]
    model = torch.jit.load(str(script_path), map_location="cpu")
    wrapped = getattr(model, "model", None)
    if wrapped is not None and set(wrapped.state_dict()) == set(state):
        incompatible = wrapped.load_state_dict(state, strict=True)
    elif set(model.state_dict()) == set(state):
        incompatible = model.load_state_dict(state, strict=True)
    elif set(model.state_dict()) == {"model."+key for key in state}:
        incompatible = model.load_state_dict({"model."+key: value for key, value in state.items()},
                                             strict=True)
    else:
        raise RuntimeError("TorchScript and checkpoint parameter names differ.")
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(str(incompatible))
    model.eval()
    return model, int(checkpoint.get("epoch", -1))


def infer(model, features, batch_size):
    outputs = []
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(np.stack(features[start:start+batch_size]))
        with torch.inference_mode():
            output = model(batch)
        if output.shape != (len(batch), SEED_COUNT):
            raise RuntimeError(f"Unexpected network output {tuple(output.shape)}.")
        outputs.append(output.cpu().numpy())
    return np.concatenate(outputs)


def rasterize(flow, points, scores, sigma):
    def nearest(axis, values):
        right = np.clip(np.searchsorted(axis, values), 0, len(axis)-1)
        left = np.clip(right-1, 0, len(axis)-1)
        return np.where(abs(values-axis[left]) <= abs(values-axis[right]), left, right)
    xi, yi = nearest(flow.x, points[:, 0]), nearest(flow.y, points[:, 1])
    weighted = np.zeros(flow.fluid.shape, dtype=np.float64)
    support = np.zeros_like(weighted)
    np.add.at(weighted, (yi, xi), scores)
    np.add.at(support, (yi, xi), 1.0)
    weighted = gaussian_filter(weighted, sigma=sigma, mode="nearest")
    support = gaussian_filter(support, sigma=sigma, mode="nearest")
    cutoff = max(float(support.max())*1e-3, 1e-12)
    probability = np.divide(weighted, support, out=np.full_like(weighted, np.nan),
                            where=support > cutoff)
    probability[~flow.analysis_fluid] = np.nan
    return probability, support


def classical_fields(flow):
    ux, uy, vx, vy = [np.asarray(value[0], dtype=np.float64)
                      for value in (flow.ux, flow.uy, flow.vx, flow.vy)]
    curl = vx-uy
    mean_curl = float(np.mean(curl[flow.analysis_fluid]))
    ivd = np.abs(curl-mean_curl)
    sxy, oxy = (uy+vx)/2, (uy-vx)/2
    q = (2*oxy**2-(ux**2+vy**2+2*sxy**2))/2
    delta = 4*(ux*vy-uy*vx)-(ux+vy)**2
    strain = np.empty(ux.shape+(2, 2)); rotation = np.empty_like(strain)
    strain[..., 0, 0], strain[..., 0, 1] = ux, sxy
    strain[..., 1, 0], strain[..., 1, 1] = sxy, vy
    rotation[..., 0, 0], rotation[..., 1, 1] = 0, 0
    rotation[..., 0, 1], rotation[..., 1, 0] = oxy, -oxy
    lambda2 = np.linalg.eigvalsh(strain@strain+rotation@rotation)[..., 1]
    return {"IVD": ivd, "Q": q, "Delta": delta, "lambda2": lambda2}


def detect_components(score, threshold, flow, min_pixels, support=None):
    valid = flow.analysis_fluid & np.isfinite(score)
    if support is not None:
        valid &= support > max(float(np.nanmax(support))*1e-3, 1e-12)
    raw_mask = valid & (score >= threshold)
    mask = np.zeros_like(raw_mask)
    detections = []
    for sign, signed in ((1, flow.curl[0] >= 0), (-1, flow.curl[0] < 0)):
        regions, count = label(raw_mask & signed)
        for region in range(1, count+1):
            yy, xx = np.nonzero(regions == region)
            if len(xx) < min_pixels:
                continue
            mask[yy, xx] = True
            values = score[yy, xx]
            weights = np.maximum(values-threshold, np.finfo(float).eps)
            detections.append(Detection(
                float(np.average(flow.x[xx], weights=weights)),
                float(np.average(flow.y[yy], weights=weights)),
                float(np.max(values)), len(xx), sign))
    detections.sort(key=lambda item: (item.x, item.y))
    return mask, detections


def classical_results(flow, args):
    fields, results = classical_fields(flow), []
    for name in ("IVD", "Q", "Delta", "lambda2"):
        score = -fields[name] if name == "lambda2" else fields[name]
        candidates = score[flow.analysis_fluid & np.isfinite(score) & (score > 0)]
        threshold = float(np.percentile(candidates, args.criterion_percentile)) \
            if len(candidates) else float("inf")
        mask, detections = detect_components(score, threshold, flow,
                                             args.min_component_pixels)
        display = np.where(flow.analysis_fluid, score, np.nan)
        results.append(Result(name, display, threshold, mask, detections))
    return results


def neural_result(name, seeding, flow, start_time, patches, checkpoint, script,
                  args, random_seed, threshold, smooth_sigma, min_pixels):
    feature_blocks, seed_blocks, valid_seed_blocks = [], [], []
    for patch in patches:
        evaluator = LocalVelocity(flow, patch)
        local_seeds = make_seeds(evaluator, seeding, start_time,
            np.random.default_rng(random_seed+patch.index), args)
        physical = patch.to_physical(local_seeds)
        feature_blocks.append(pathline_features(flow, patch, local_seeds,
                                                start_time, args.duration))
        seed_blocks.append(physical)
        valid_seed_blocks.append(flow.points_are_fluid(physical))
    model, epoch = load_model(checkpoint, script)
    block_scores = infer(model, feature_blocks, args.batch_size)
    points = np.concatenate(seed_blocks)
    scores = block_scores.reshape(-1)
    valid = np.concatenate(valid_seed_blocks)
    probability, support = rasterize(flow, points[valid], scores[valid], smooth_sigma)
    mask, detections = detect_components(probability, threshold, flow,
                                         min_pixels, support)
    return Result(name, probability, threshold, mask, detections, seeding), \
        epoch, points, scores, valid


def legacy_vt_result(flow, start_time, checkpoint, script, args):
    """Run VortexTransformer with the exact preprocessing used in training.

    The original checkpoint is not a GRID_CROSS_SEEDING model.  It was trained
    with a regular 16-by-16 seed grid, a 0.72 physical-time horizon, anchored
    model time, and a fixed historical IVD reference.  Applying the newer
    cross adapter makes high probabilities follow tile boundaries and pipe
    walls instead of coherent vortices.
    """

    legacy_flow, legacy_start, _, _, constants = legacy_vt.load_flow_window(
        args.input,
        args.time_index,
        args.vt_duration,
        legacy_reference_indices=(
            args.vt_ivd_reference_start_index,
            args.vt_ivd_reference_end_index,
        ),
    )
    if not np.isclose(legacy_start, start_time):
        raise RuntimeError("Legacy and shared flow windows start at different times.")

    prepared = legacy_vt.prepare_model_inputs(
        flow=legacy_flow,
        mode=legacy_vt.LEGACY_GRID_MODE,
        start_time=legacy_start,
        duration=args.vt_duration,
        obstacle_center=flow.cylinders[0],
        obstacle_radius=float(constants.get("radius", flow.radius)),
        patch_points=args.vt_patch_points,
        patch_stride_points=args.vt_patch_stride,
        tile_size=None,
        tile_stride=None,
        legacy_time_mode=legacy_vt.LEGACY_TIME_ANCHORED,
        model_time_start=args.vt_model_time_start,
    )
    model, epoch = legacy_vt.load_model(checkpoint, script, "cpu")
    block_scores = legacy_vt.predict_batches(
        model, prepared.features, "cpu", args.batch_size
    )
    points, scores, _ = legacy_vt.merge_duplicate_seeds(
        prepared.seeds, block_scores, prepared.display_valid
    )

    # The legacy adapter only knows about the first cylinder.  Apply the
    # measured fluid mask here so neither cylinder nor a pipe wall can be
    # reported as a vortex in the comparison figure or CSV.
    valid = flow.points_are_fluid(points)
    probability, support = rasterize(
        flow, points[valid], scores[valid], args.vt_smooth_sigma
    )
    mask, detections = detect_components(
        probability,
        args.vt_threshold,
        flow,
        args.vt_min_component_pixels,
        support,
    )
    return (
        Result(
            "VortexTransformer",
            probability,
            args.vt_threshold,
            mask,
            detections,
            "LEGACY_GRID_SEEDING",
        ),
        epoch,
        points,
        scores,
        valid,
    )


def existing(path, fallbacks=()):
    if path.is_file():
        return path
    for fallback in fallbacks:
        if fallback.is_file():
            print(f"Path {path} not found; using repository file {fallback}.")
            return fallback
    raise FileNotFoundError(f"No model file found for {path}.")


def save_detections(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("method", "seeding", "id", "x", "y", "peak",
                         "pixels", "rotation_sign", "threshold"))
        for result in results:
            for index, item in enumerate(result.detections, 1):
                writer.writerow((result.name, result.seeding, index,
                    f"{item.x:.9g}", f"{item.y:.9g}", f"{item.peak:.9g}",
                    item.pixels, item.rotation_sign, f"{result.threshold:.9g}"))


def save_seed_scores(path, outputs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("method", "seeding", "id", "x", "y", "score", "fluid"))
        for name, seeding, points, scores, valid in outputs:
            for index, (point, score, is_valid) in enumerate(zip(points, scores, valid), 1):
                writer.writerow((name, seeding, index, f"{point[0]:.9g}",
                    f"{point[1]:.9g}", f"{float(score):.9g}", int(is_valid)))


def agreement(reference, candidate):
    """Return mask IoU/F1 and optimal mean center distance."""
    intersection = int(np.count_nonzero(reference.mask & candidate.mask))
    union = int(np.count_nonzero(reference.mask | candidate.mask))
    reference_pixels = int(np.count_nonzero(reference.mask))
    candidate_pixels = int(np.count_nonzero(candidate.mask))
    precision = intersection/max(candidate_pixels, 1)
    recall = intersection/max(reference_pixels, 1)
    f1 = 2*precision*recall/max(precision+recall, np.finfo(float).eps)
    iou = intersection/max(union, 1)
    reference_centers = np.asarray([(d.x, d.y) for d in reference.detections])
    candidate_centers = np.asarray([(d.x, d.y) for d in candidate.detections])
    if len(reference_centers) and len(candidate_centers):
        distances = np.linalg.norm(
            reference_centers[:, None]-candidate_centers[None, :], axis=2)
        rows, columns = linear_sum_assignment(distances)
        mean_center_distance = float(np.mean(distances[rows, columns]))
    else:
        mean_center_distance = float("nan")
    return iou, f1, mean_center_distance


def plot_results(path, flow, start_time, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 4, figsize=(22, 7.8), constrained_layout=True)
    skip = 3
    field_cmap = plot_style.FIELD_CMAP
    streamline_sets = []
    for axis, result in zip(axes.flat, results):
        is_neural = result.name in (
            "MVUNet", "VortexUNet", "VortexTransformer", "FGMA-VT"
        )
        # Sparse neural support is zero-valued background, not solid. Keeping
        # NaN only outside the eroded fluid mask avoids white sampling holes.
        display = result.score
        if is_neural:
            display = np.where(
                flow.analysis_fluid, np.nan_to_num(display, nan=0.0), np.nan)
        display = np.maximum(display, 0.0)
        positive = display[np.isfinite(display) & (display > 0)]
        vmax = 1.0 if is_neural else (float(np.percentile(positive, 99.5))
                                      if len(positive) else 1.0)
        axis.set_facecolor(plot_style.BACKGROUND_COLOR)
        image = axis.imshow(display, extent=(flow.x[0], flow.x[-1],
            flow.y[0], flow.y[-1]), origin="lower", aspect="equal",
            cmap=field_cmap, vmin=0, vmax=max(vmax, 1e-12))
        stream_x = np.linspace(flow.x[0], flow.x[-1], len(flow.x[::skip]))
        stream_y = np.linspace(flow.y[0], flow.y[-1], len(flow.y[::skip]))
        streamline_sets.append(axis.streamplot(stream_x, stream_y,
            flow.u[0, ::skip, ::skip], flow.v[0, ::skip, ::skip],
            color=plot_style.STREAMLINE_COLOR,
            density=0.65, linewidth=0.3, arrowsize=0.4))
        if np.any(result.mask):
            axis.contour(flow.grid_x, flow.grid_y, result.mask.astype(float),
                         levels=[0.5], colors=plot_style.THRESHOLD_COLOR,
                         linewidths=0.8)
        for center in flow.cylinders:
            axis.add_patch(Circle(
                center,
                flow.radius,
                facecolor=plot_style.OBSTACLE_FACE_COLOR,
                edgecolor=plot_style.OBSTACLE_EDGE_COLOR,
                linewidth=0.7,
                zorder=6,
            ))
        title = f"{result.name} ({len(result.detections)} regions)"
        if result.seeding != "-":
            title += "\n"+result.seeding
        axis.set(title=title, xlim=(flow.x[0], flow.x[-1]),
                 ylim=(flow.y[0], flow.y[-1]))
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, shrink=0.75, pad=0.015)
    for axis in axes.flat[len(results):]:
        axis.set_visible(False)
    figure.suptitle(f"Piped-cylinder vortex detection at t={start_time:.4g}\n"
        "dark: solid; yellow: detected region", fontsize=14)
    if path.suffix.lower() == ".pdf":
        plot_style.thin_streamplots_for_pdf(streamline_sets)
        figure.savefig(path, dpi=220)
    else:
        figure.savefig(path, dpi=220)
        plot_style.thin_streamplots_for_pdf(streamline_sets)
        figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/pipedcylinder2d.nc"))
    parser.add_argument("--time-index", type=int, default=1000)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--patch-points", type=int, default=75)
    parser.add_argument("--patch-stride", type=int, default=75)
    parser.add_argument("--minimum-fluid-fraction", type=float, default=0.15)
    parser.add_argument("--criterion-percentile", type=float, default=70.0)
    parser.add_argument("--min-component-pixels", type=int, default=6)
    parser.add_argument("--vt-duration", type=float,
        default=legacy_vt.LEGACY_DEFAULT_DURATION)
    parser.add_argument("--vt-patch-points", type=int, default=32)
    parser.add_argument("--vt-patch-stride", type=int, default=32)
    parser.add_argument("--vt-model-time-start", type=float,
        default=legacy_vt.LEGACY_DEFAULT_MODEL_TIME_START)
    parser.add_argument("--vt-ivd-reference-start-index", type=int, default=800)
    parser.add_argument("--vt-ivd-reference-end-index", type=int, default=1000)
    parser.add_argument("--vt-threshold", type=float, default=0.5)
    parser.add_argument("--vt-smooth-sigma", type=float, default=1.0)
    parser.add_argument("--vt-min-component-pixels", type=int, default=50)
    # Jointly calibrated against the fluid-masked IVD response at t=10.  The
    # sparse probability raster needs a small cutoff and limited smoothing;
    # the former 5e-4 / 2.5 combination merged unrelated support regions.
    parser.add_argument("--fgma-threshold", type=float, default=0.0004)
    parser.add_argument("--fgma-smooth-sigma", type=float, default=2.5)
    parser.add_argument("--fgma-min-component-pixels", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=4259)
    parser.add_argument("--cross-jitter", type=float, default=0.05)
    parser.add_argument("--feature-high-fraction", type=float, default=0.8)
    parser.add_argument("--feature-high-percentile", type=float, default=70.0)
    parser.add_argument("--feature-mesh-resolution", type=int, default=64)
    parser.add_argument("--feature-mesh-size-ratio", type=float, default=4.0)
    parser.add_argument("--feature-monitor-power", type=float, default=1.5)
    parser.add_argument("--feature-mesh-iterations", type=int, default=5)
    parser.add_argument("--feature-clip-percentile", type=float, default=99.0)
    parser.add_argument("--feature-boundary-margin", type=float, default=1e-6)
    parser.add_argument("--vt-checkpoint", type=Path,
        default=Path("trainedVortexTransformer/best_checkpoint.pth.tar"))
    parser.add_argument("--vt-torchscript", type=Path,
        default=Path("trainedVortexTransformer/Backup_best_checkpoint.pthtraced.pt"))
    parser.add_argument("--fgma-checkpoint", type=Path,
        default=Path("trainedVortexTransformerBetter/best_checkpoint.pth.tar"))
    parser.add_argument("--fgma-torchscript", type=Path,
        default=Path("trainedVortexTransformerBetter/Backup_best_checkpoint.pthtraced.pt"))
    parser.add_argument("--output", type=Path,
        default=Path("outputs/pipedcylinder2d_comparison_new.png"))
    parser.add_argument("--csv-output", type=Path,
        default=Path("outputs/pipedcylinder2d_comparison_new.csv"))
    parser.add_argument("--seed-scores-output", type=Path,
        default=Path("outputs/pipedcylinder2d_seed_scores_new.csv"))
    mvunet.add_arguments(parser, min_component_pixels=60)
    vortexunet.add_arguments(parser, min_component_pixels=60)
    return parser


def validate_args(args):
    mvunet.validate_arguments(args)
    vortexunet.validate_arguments(args)
    if (min(args.duration, args.vt_duration) <= 0
            or min(args.patch_stride, args.vt_patch_stride) < 1):
        raise ValueError("Durations and patch strides must be positive.")
    if not 0 < args.minimum_fluid_fraction <= 1:
        raise ValueError("--minimum-fluid-fraction must be in (0,1].")
    if (not 0 < args.criterion_percentile < 100
            or not 0 < args.vt_threshold < 1
            or not 0 < args.fgma_threshold < 1):
        raise ValueError("Percentile or probability threshold is invalid.")
    if (min(args.vt_smooth_sigma, args.fgma_smooth_sigma) <= 0
            or min(args.min_component_pixels, args.vt_min_component_pixels,
                   args.fgma_min_component_pixels, args.batch_size) < 1):
        raise ValueError("Sigma and counts must be positive.")
    if (args.vt_ivd_reference_start_index < 0
            or args.vt_ivd_reference_end_index
            < args.vt_ivd_reference_start_index):
        raise ValueError("Invalid VortexTransformer IVD reference frame range.")


def main(argv: Iterable[str] | None = None):
    args = build_parser().parse_args(argv)
    validate_args(args)
    flow, start = load_flow(args.input, args.time_index, args.duration)
    patches = make_patches(flow, args.patch_points, args.patch_stride,
                           args.minimum_fluid_fraction)
    print(f"Loaded {args.input}: {len(flow.x)}x{len(flow.y)}, "
          f"t=[{start:g},{start+args.duration:g}], fluid patches={len(patches)}")

    vt_checkpoint = existing(args.vt_checkpoint)
    vt_script = existing(args.vt_torchscript,
        (Path("trainedVortexTransformer/Backup_best_checkpoint.pthtraced.pt"),))
    fgma_checkpoint = existing(args.fgma_checkpoint,
        (Path("trainedVortexTransformerBetter/best_checkpoint.pth.tar"),))
    fgma_script = existing(args.fgma_torchscript,
        (Path("trainedVortexTransformerBetter/Backup_best_checkpoint.pthtraced.pt"),))

    results = classical_results(flow, args)
    print("Running VortexTransformer with LEGACY_GRID_SEEDING...")
    vt, vt_epoch, vt_points, vt_scores, vt_valid = legacy_vt_result(
        flow, start, vt_checkpoint, vt_script, args)
    print("Running FGMA-VT with FEATURE_ADAPTIVE_MESH_SEEDING...")
    fgma, fgma_epoch, fgma_points, fgma_scores, fgma_valid = neural_result(
        "FGMA-VT", FEATURE_ADAPTIVE_MESH_SEEDING, flow, start, patches,
        fgma_checkpoint, fgma_script, args, args.random_seed+100000,
        args.fgma_threshold, args.fgma_smooth_sigma,
        args.fgma_min_component_pixels)
    print(f"Running MVUNet with {mvunet.TRAINING_SEEDING} training data...")
    mv = mvunet.run(
        flow.u[0], flow.v[0], flow.x, flow.y, args,
        valid_mask=flow.analysis_fluid,
    )
    mv_result = Result(
        "MVUNet", mv.probability, args.mvunet_threshold, mv.mask,
        mv.detections, mvunet.TRAINING_SEEDING,
    )
    print(
        f"Running VortexUNet with {vortexunet.TRAINING_SEEDING} "
        "training data..."
    )
    vu = vortexunet.run(
        flow.u[0], flow.v[0], flow.x, flow.y, args,
        valid_mask=flow.analysis_fluid,
    )
    vu_result = Result(
        "VortexUNet", vu.probability, args.vortexunet_threshold, vu.mask,
        vu.detections, vortexunet.TRAINING_SEEDING,
    )
    results.extend((mv_result, vu_result, vt, fgma))

    iou, f1, center_distance = agreement(results[0], fgma)
    print(f"FGMA-VT vs IVD: IoU={iou:.3f}, F1={f1:.3f}, "
          f"mean matched-center distance={center_distance:.3f}")

    save_detections(args.csv_output, results)
    save_seed_scores(args.seed_scores_output, (
        (vt.name, vt.seeding, vt_points, vt_scores, vt_valid),
        (fgma.name, fgma.seeding, fgma_points, fgma_scores, fgma_valid),
    ))
    plot_results(args.output, flow, start, results)
    print(
        f"VortexTransformer epoch={vt_epoch}; FGMA-VT epoch={fgma_epoch}; "
        f"MVUNet epoch={mv.epoch}; VortexUNet epoch={vu.epoch}"
    )
    for result in results:
        print(f"{result.name:20s}: {len(result.detections):3d} regions, "
              f"threshold={result.threshold:.6g}, seeding={result.seeding}")
    print(f"Visualization: {args.output} and {args.output.with_suffix('.pdf')}")
    print(f"Detections: {args.csv_output}\nSeed scores: {args.seed_scores_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
