#!/usr/bin/env python3
"""Compare eight classical and learned vortex detectors on the double gyre.

Both transformer networks receive [B,16,256,7] pathline tensors with channels
[x,y,t,IVD,distance,u,v].  VortexTransformer uses GRID_CROSS_SEEDING and
FGMA-VT uses FEATURE_ADAPTIVE_MESH_SEEDING from VatistasFlowDatasetGenerator.
MVUNet uses the generator's dense [u,v,curl,IVD] t0 representation; its
training-data provenance is GRID_CROSS_SEEDING.
VortexUNet instead uses the generator's dense [u,v] t0 representation, with
the same GRID_CROSS_SEEDING training-data provenance.
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
from scipy.ndimage import gaussian_filter, label

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import comparison_plot_style as plot_style
import mvunet_comparison as mvunet
import vortexunet_comparison as vortexunet
from VatistasFlowDatasetGenerator import (
    FEATURE_ADAPTIVE_MESH_SEEDING,
    GRID_CROSS_SEEDING,
    _pathline_seedings,
)

LENGTH, COUNT_K, SEED_COUNT, FEATURE_COUNT = 16, 16, 256, 7
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

    def physical(self, p):
        p = np.asarray(p, dtype=np.float64)
        return np.column_stack((self.xmin + (p[:, 0] + 1) * self.width / 2,
                                self.ymin + (p[:, 1] + 1) * self.height / 2))


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    peak: float
    pixels: int
    sign: int


@dataclass
class Result:
    name: str
    field: np.ndarray
    score: np.ndarray
    threshold: float
    mask: np.ndarray
    detections: list[Detection]
    seeding: str = "-"


class Flow:
    """Loaded NetCDF time window and its velocity-gradient components."""
    def __init__(self, x, y, times, u, v):
        self.x, self.y, self.times = map(lambda a: np.asarray(a, np.float64),
                                         (x, y, times))
        self.u, self.v = np.asarray(u, np.float32), np.asarray(v, np.float32)
        shape = (len(times), len(y), len(x))
        if self.u.shape != shape or self.v.shape != shape:
            raise ValueError(f"Velocity arrays must be {shape}.")
        if min(len(x), len(y)) < 3 or len(times) < 2:
            raise ValueError("At least 3x3x2 samples are required.")
        self.ux = np.gradient(self.u, self.x, axis=2, edge_order=2)
        self.uy = np.gradient(self.u, self.y, axis=1, edge_order=2)
        self.vx = np.gradient(self.v, self.x, axis=2, edge_order=2)
        self.vy = np.gradient(self.v, self.y, axis=1, edge_order=2)
        self.curl = self.vx - self.uy

    def time_bracket(self, t):
        hi = min(max(int(np.searchsorted(self.times, t, side="right")), 1),
                 len(self.times) - 1)
        lo = hi - 1
        w = (float(t) - self.times[lo]) / (self.times[hi] - self.times[lo])
        return lo, hi, float(np.clip(w, 0, 1))

    @staticmethod
    def axis_bracket(axis, values):
        hi = np.clip(np.searchsorted(axis, values, side="right"), 1, len(axis)-1)
        lo = hi - 1
        w = (values-axis[lo])/(axis[hi]-axis[lo])
        return lo, hi, np.clip(w, 0, 1)

    def sample(self, field, points, t):
        p = np.asarray(points, np.float64)
        x0, x1, wx = self.axis_bracket(self.x, p[:, 0])
        y0, y1, wy = self.axis_bracket(self.y, p[:, 1])
        t0, t1, wt = self.time_bracket(t)
        def bilinear(frame):
            a = frame[y0, x0]*(1-wx) + frame[y0, x1]*wx
            b = frame[y1, x0]*(1-wx) + frame[y1, x1]*wx
            return a*(1-wy) + b*wy
        return bilinear(field[t0])*(1-wt) + bilinear(field[t1])*wt

    def velocity(self, points, t):
        return np.column_stack((self.sample(self.u, points, t),
                                self.sample(self.v, points, t)))

    def local_curl(self, patch, points, t):
        physical = patch.physical(points)
        return ((patch.width/patch.height)*self.sample(self.vx, physical, t)
                - (patch.height/patch.width)*self.sample(self.uy, physical, t))

    def local_mean_curl(self, patch, t):
        lo, hi, w = self.time_bracket(t)
        def mean(k):
            vx = self.vx[k, patch.ys:patch.ye, patch.xs:patch.xe]
            uy = self.uy[k, patch.ys:patch.ye, patch.xs:patch.xe]
            return float(np.mean((patch.width/patch.height)*vx
                                 - (patch.height/patch.width)*uy))
        return mean(lo)*(1-w) + mean(hi)*w


class TorchVelocity:
    """Differentiable local velocity required by adaptive mesh seeding."""
    device, dtype = torch.device("cpu"), torch.float32
    def __init__(self, parent):
        self.parent = parent
        self.u = torch.from_numpy(parent.flow.u)
        self.v = torch.from_numpy(parent.flow.v)

    def __call__(self, points, t):
        flow, patch = self.parent.flow, self.parent.patch
        old_shape = points.shape
        p = points.reshape(-1, 2)
        px = patch.xmin + (p[:, 0]+1)*patch.width/2
        py = patch.ymin + (p[:, 1]+1)*patch.height/2
        gx = 2*(px-flow.x[0])/(flow.x[-1]-flow.x[0])-1
        gy = 2*(py-flow.y[0])/(flow.y[-1]-flow.y[0])-1
        grid = torch.stack((gx, gy), -1).reshape(1, 1, -1, 2)
        lo, hi, w = flow.time_bracket(float(t))
        def sample(k):
            image = torch.stack((self.u[k], self.v[k]))[None]
            return F.grid_sample(image, grid, align_corners=True,
                                 padding_mode="border")[0, :, 0].T
        velocity = sample(lo)*(1-w) + sample(hi)*w
        scale = velocity.new_tensor((2/patch.width, 2/patch.height))
        return (velocity*scale).reshape(old_shape)


class LocalVelocity:
    def __init__(self, flow, patch):
        self.flow, self.patch = flow, patch
        self.torch_eval = TorchVelocity(self)

    def __call__(self, points, t):
        velocity = self.flow.velocity(self.patch.physical(points), t)
        return velocity*np.asarray((2/self.patch.width, 2/self.patch.height))


def load_flow(path, time_index, duration):
    with nc.Dataset(path) as ds:
        required = ("u", "v", "xdim", "ydim", "tdim")
        missing = [n for n in required if n not in ds.variables]
        if missing:
            raise ValueError(f"Missing NetCDF variables: {missing}")
        x, y = np.asarray(ds["xdim"][:]), np.asarray(ds["ydim"][:])
        all_t = np.asarray(ds["tdim"][:], np.float64)
        if not 0 <= time_index < len(all_t)-1:
            raise ValueError(f"--time-index must be in [0,{len(all_t)-2}].")
        start, end = float(all_t[time_index]), float(all_t[time_index]+duration)
        if end > all_t[-1]:
            raise ValueError(f"Requested end time {end:g} exceeds {all_t[-1]:g}.")
        last = min(int(np.searchsorted(all_t, end))+1, len(all_t)-1)
        frames = slice(time_index, last+1)
        flow = Flow(x, y, all_t[frames], ds["u"][frames], ds["v"][frames])
    return flow, start


def axis_starts(size, points, stride):
    if not 3 <= points <= size or stride < 1:
        raise ValueError(f"Invalid patch={points}, stride={stride}, axis={size}.")
    final = size-points
    starts = list(range(0, final+1, stride))
    if starts[-1] != final:
        starts.append(final)
    return starts


def make_patches(flow, points, stride):
    patches = []
    for ys in axis_starts(len(flow.y), points, stride):
        for xs in axis_starts(len(flow.x), points, stride):
            xe, ye = xs+points, ys+points
            patches.append(Patch(len(patches), xs, xe, ys, ye,
                                 float(flow.x[xs]), float(flow.x[xe-1]),
                                 float(flow.y[ys]), float(flow.y[ye-1])))
    return patches


def seeds_for(evaluator, method, start, rng, args):
    config = dict(count_k=COUNT_K, sampling=method, jitter_std=args.cross_jitter,
        feature_high_fraction=args.feature_high_fraction,
        feature_high_vorticity_percentile=args.feature_high_percentile,
        feature_mesh_resolution=args.feature_mesh_resolution,
        feature_mesh_size_ratio=args.feature_mesh_size_ratio,
        feature_monitor_power=args.feature_monitor_power,
        feature_mesh_iterations=args.feature_mesh_iterations,
        feature_clip_percentile=args.feature_clip_percentile,
        feature_boundary_margin=args.feature_boundary_margin)
    seeds = _pathline_seedings(config, rng, velocity_eval=evaluator, time=start)
    if seeds.shape != (SEED_COUNT, 2):
        raise RuntimeError(f"{method} returned unexpected shape {seeds.shape}.")
    return seeds


def pathline_features(flow, patch, seeds, start, duration):
    evaluator = LocalVelocity(flow, patch)
    times = np.linspace(start, start+duration, LENGTH)
    model_times = np.linspace(0, 1, LENGTH)
    positions = seeds.copy()
    active = np.all((positions > -1) & (positions < 1), axis=1)
    features = np.full((LENGTH, SEED_COUNT, FEATURE_COUNT), PADDING, np.float32)
    for step, time in enumerate(times):
        ids = np.flatnonzero(active)
        if len(ids):
            p = positions[ids]
            features[step, ids, :2] = p
            features[step, ids, 2] = model_times[step]
            features[step, ids, 3] = np.abs(flow.local_curl(patch, p, time)
                - flow.local_mean_curl(patch, time))
            features[step, ids, 4] = np.linalg.norm(p-seeds[ids], axis=1)
            features[step, ids, 5:] = evaluator(p, time)
        if step == LENGTH-1 or not len(ids):
            continue
        dt, p = times[step+1]-time, positions[ids]
        k1 = evaluator(p, time)
        k2 = evaluator(p+dt*k1/2, time+dt/2)
        k3 = evaluator(p+dt*k2/2, time+dt/2)
        k4 = evaluator(p+dt*k3, time+dt)
        new = p+dt*(k1+2*k2+2*k3+k4)/6
        ok = np.all(np.isfinite(new), axis=1) & np.all((new>-1)&(new<1), axis=1)
        positions[ids[ok]] = new[ok]
        active[ids[~ok]] = False
    features[0, :, 4] = 0
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
    elif set(model.state_dict()) == {"model."+k for k in state}:
        incompatible = model.load_state_dict({"model."+k: v for k, v in state.items()},
                                             strict=True)
    else:
        raise RuntimeError("TorchScript/checkpoint parameter names do not match.")
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(str(incompatible))
    model.eval()
    return model, int(checkpoint.get("epoch", -1))


def infer(model, blocks, batch_size):
    outputs = []
    for first in range(0, len(blocks), batch_size):
        batch = torch.from_numpy(np.stack(blocks[first:first+batch_size]))
        with torch.inference_mode():
            out = model(batch)
        if out.shape != (len(batch), SEED_COUNT):
            raise RuntimeError(f"Unexpected model output {tuple(out.shape)}.")
        outputs.append(out.cpu().numpy())
    return np.concatenate(outputs)


def rasterize(x, y, points, scores, sigma):
    def nearest(axis, values):
        right = np.clip(np.searchsorted(axis, values), 0, len(axis)-1)
        left = np.clip(right-1, 0, len(axis)-1)
        return np.where(abs(values-axis[left]) <= abs(values-axis[right]), left, right)
    xi, yi = nearest(x, points[:, 0]), nearest(y, points[:, 1])
    weighted, support = np.zeros((len(y), len(x))), np.zeros((len(y), len(x)))
    np.add.at(weighted, (yi, xi), scores)
    np.add.at(support, (yi, xi), 1)
    weighted = gaussian_filter(weighted, sigma, mode="nearest")
    support = gaussian_filter(support, sigma, mode="nearest")
    cutoff = max(float(support.max())*1e-3, 1e-12)
    probability = np.divide(weighted, support, out=np.full_like(weighted, np.nan),
                            where=support > cutoff)
    return probability, support


def criterion_fields(flow):
    ux, uy, vx, vy = map(lambda a: np.asarray(a[0], np.float64),
                         (flow.ux, flow.uy, flow.vx, flow.vy))
    curl = vx-uy
    ivd = abs(curl-curl.mean())
    sxy, oxy = (uy+vx)/2, (uy-vx)/2
    q = ((2*oxy**2) - (ux**2+vy**2+2*sxy**2))/2
    delta = 4*(ux*vy-uy*vx) - (ux+vy)**2
    s = np.empty(ux.shape+(2, 2)); o = np.empty_like(s)
    s[..., 0, 0], s[..., 0, 1] = ux, sxy
    s[..., 1, 0], s[..., 1, 1] = sxy, vy
    o[..., 0, 0], o[..., 1, 1] = 0, 0
    o[..., 0, 1], o[..., 1, 0] = oxy, -oxy
    lambda2 = np.linalg.eigvalsh(s@s + o@o)[..., 1]
    return {"IVD": ivd, "Q": q, "Delta": delta, "lambda2": lambda2}


def components(score, threshold, flow, min_pixels, border, support=None):
    valid = np.isfinite(score)
    if support is not None:
        valid &= support > max(float(np.nanmax(support))*1e-3, 1e-12)
    if border:
        valid[:border] = valid[-border:] = False
        valid[:, :border] = valid[:, -border:] = False
    mask, detections = valid & (score >= threshold), []
    for sign, signed in ((1, flow.curl[0] >= 0), (-1, flow.curl[0] < 0)):
        regions, count = label(mask & signed)
        for region in range(1, count+1):
            yy, xx = np.nonzero(regions == region)
            if len(xx) < min_pixels:
                continue
            values = score[yy, xx]
            weights = np.maximum(values-threshold, np.finfo(float).eps)
            detections.append(Detection(float(np.average(flow.x[xx], weights=weights)),
                float(np.average(flow.y[yy], weights=weights)), float(values.max()),
                len(xx), sign))
    detections.sort(key=lambda d: (d.x, d.y))
    return mask, detections


def classical(flow, args):
    fields, results = criterion_fields(flow), []
    for name in ("IVD", "Q", "Delta", "lambda2"):
        score = -fields[name] if name == "lambda2" else fields[name]
        positive = score[np.isfinite(score) & (score > 0)]
        threshold = float(np.percentile(positive, args.criterion_percentile)) \
            if len(positive) else float("inf")
        mask, found = components(score, threshold, flow,
                                 args.min_component_pixels, args.border_cells)
        results.append(Result(name, fields[name], score, threshold, mask, found))
    return results


def neural(name, seeding, flow, start, patches, checkpoint, script, args, seed):
    feature_blocks, seed_blocks = [], []
    for patch in patches:
        evaluator = LocalVelocity(flow, patch)
        local = seeds_for(evaluator, seeding, start,
                          np.random.default_rng(seed+patch.index), args)
        feature_blocks.append(pathline_features(flow, patch, local, start, args.duration))
        seed_blocks.append(patch.physical(local))
    model, epoch = load_model(checkpoint, script)
    block_scores = infer(model, feature_blocks, args.batch_size)
    points, scores = np.concatenate(seed_blocks), block_scores.reshape(-1)
    probability, support = rasterize(flow.x, flow.y, points, scores,
                                     args.model_smooth_sigma)
    mask, found = components(probability, args.model_threshold, flow,
                             args.min_component_pixels, args.border_cells, support)
    return Result(name, probability, probability, args.model_threshold, mask, found,
                  seeding), epoch, points, scores


def existing(path, fallbacks=()):
    if path.is_file():
        return path
    for fallback in fallbacks:
        if fallback.is_file():
            print(f"Path {path} not found; using repository file {fallback}.")
            return fallback
    raise FileNotFoundError(f"No model file found for {path}.")


def save_csv(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(("method", "seeding", "id", "x", "y",
            "peak", "pixels", "rotation_sign", "threshold"))
        for result in results:
            for i, d in enumerate(result.detections, 1):
                w.writerow((result.name, result.seeding, i, f"{d.x:.9g}",
                    f"{d.y:.9g}", f"{d.peak:.9g}", d.pixels, d.sign,
                    f"{result.threshold:.9g}"))


def save_seeds(path, outputs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(("method", "seeding", "id", "x", "y", "score"))
        for name, seeding, points, scores in outputs:
            for i, (p, score) in enumerate(zip(points, scores), 1):
                w.writerow((name, seeding, i, f"{p[0]:.9g}", f"{p[1]:.9g}",
                            f"{float(score):.9g}"))


def plot(path, flow, start, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.6), constrained_layout=True)
    xx, yy = np.meshgrid(flow.x, flow.y)
    skip = max(1, min(len(flow.x), len(flow.y))//64)
    field_cmap = plot_style.FIELD_CMAP
    streamline_sets = []
    for ax, result in zip(axes.flat, results):
        positive = result.score[np.isfinite(result.score) & (result.score > 0)]
        neural_method = result.name in (
            "MVUNet", "VortexUNet", "VortexTransformer", "FGMA-VT"
        )
        vmax = 1 if neural_method else (np.percentile(positive, 99.5)
                                        if len(positive) else 1)
        display = np.maximum(result.score, 0.0)
        ax.set_facecolor(plot_style.BACKGROUND_COLOR)
        image = ax.imshow(display, extent=(flow.x[0], flow.x[-1],
            flow.y[0], flow.y[-1]), origin="lower", aspect="equal",
            cmap=field_cmap, vmin=0, vmax=max(vmax, 1e-12))
        streamline_sets.append(ax.streamplot(
            xx[::skip, ::skip], yy[::skip, ::skip],
            flow.u[0, ::skip, ::skip], flow.v[0, ::skip, ::skip],
            color=plot_style.STREAMLINE_COLOR, density=.65,
            linewidth=.35, arrowsize=.45))
        if result.mask.any():
            ax.contour(xx, yy, result.mask.astype(float), levels=[.5],
                       colors=plot_style.THRESHOLD_COLOR, linewidths=1)
        title = f"{result.name} ({len(result.detections)} regions)"
        if result.seeding != "-": title += "\n"+result.seeding
        ax.set(title=title, xlim=(flow.x[0], flow.x[-1]),
               ylim=(flow.y[0], flow.y[-1]))
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, shrink=.82, pad=.02)
    for ax in axes.flat[len(results):]:
        ax.set_visible(False)
    fig.suptitle(f"Double gyre vortex detection at t={start:.4g}\n"
                 "yellow: detected region", fontsize=14)
    if path.suffix.lower() == ".pdf":
        plot_style.thin_streamplots_for_pdf(streamline_sets)
        fig.savefig(path, dpi=220)
    else:
        fig.savefig(path, dpi=220)
        plot_style.thin_streamplots_for_pdf(streamline_sets)
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("dataset/doublegyre2d.nc"))
    p.add_argument("--time-index", type=int, default=0)
    p.add_argument("--duration", type=float, default=1.)
    p.add_argument("--patch-points", type=int, default=128)
    p.add_argument("--patch-stride", type=int, default=128)
    p.add_argument("--criterion-percentile", type=float, default=90.)
    p.add_argument("--model-threshold", type=float, default=0.0005)
    p.add_argument("--model-smooth-sigma", type=float, default=4.)
    p.add_argument("--min-component-pixels", type=int, default=8)
    p.add_argument("--border-cells", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--random-seed", type=int, default=4259)
    p.add_argument("--cross-jitter", type=float, default=.05)
    p.add_argument("--feature-high-fraction", type=float, default=.8)
    p.add_argument("--feature-high-percentile", type=float, default=70.)
    p.add_argument("--feature-mesh-resolution", type=int, default=64)
    p.add_argument("--feature-mesh-size-ratio", type=float, default=4.)
    p.add_argument("--feature-monitor-power", type=float, default=1.5)
    p.add_argument("--feature-mesh-iterations", type=int, default=5)
    p.add_argument("--feature-clip-percentile", type=float, default=99.)
    p.add_argument("--feature-boundary-margin", type=float, default=1e-6)
    p.add_argument("--vt-checkpoint", type=Path,
                   default=Path("trainedVortexTransformer/best_checkpoint.pth.tar"))
    p.add_argument("--vt-torchscript", type=Path,
                   default=Path("trainVortexTransformer/Backup_best_chekcpoint.pthtraced.pt"))
    p.add_argument("--fgma-checkpoint", type=Path,
                   default=Path("trainVortexTransformerBetter/best_checkpoint.pth.tar"))
    p.add_argument("--fgma-torchscript", type=Path,
                   default=Path("trainVortexTransformerBetter/Backup_best_chekcpoint.pthtraced.pt"))
    p.add_argument("--output", type=Path,
                   default=Path("outputs/doublegyre2d_comparison_new.png"))
    p.add_argument("--csv-output", type=Path,
                   default=Path("outputs/doublegyre2d_comparison_new.csv"))
    p.add_argument("--seed-scores-output", type=Path,
                   default=Path("outputs/doublegyre2d_seed_scores_new.csv"))
    mvunet.add_arguments(p)
    vortexunet.add_arguments(p)
    return p


def validate(args):
    mvunet.validate_arguments(args)
    vortexunet.validate_arguments(args)
    if args.duration <= 0 or args.patch_stride < 1:
        raise ValueError("Duration and patch stride must be positive.")
    if not 0 < args.criterion_percentile < 100 or not 0 < args.model_threshold < 1:
        raise ValueError("Percentile/threshold values are out of range.")
    if args.model_smooth_sigma <= 0 or min(args.min_component_pixels,
                                           args.batch_size) < 1:
        raise ValueError("Sigma and counts must be positive.")


def main(argv: Iterable[str] | None = None):
    args = parser().parse_args(argv); validate(args)
    flow, start = load_flow(args.input, args.time_index, args.duration)
    patches = make_patches(flow, args.patch_points, args.patch_stride)
    print(f"Loaded {args.input}: {len(flow.x)}x{len(flow.y)}, "
          f"t=[{start:g},{start+args.duration:g}], patches={len(patches)}")
    vt_ckpt = existing(args.vt_checkpoint)
    vt_script = existing(args.vt_torchscript,
        (Path("trainedVortexTransformer/BackUp_best_checkpoint.pthtraced.pt"),))
    fg_ckpt = existing(args.fgma_checkpoint,
        (Path("trainedVortexTransformerBetter/best_checkpoint.pth.tar"),))
    fg_script = existing(args.fgma_torchscript,
        (Path("trainedVortexTransformerBetter/BackUp_best_checkpoint.pthtraced.pt"),))
    results = classical(flow, args)
    print("Running VortexTransformer with GRID_CROSS_SEEDING...")
    vt, vt_epoch, vt_points, vt_scores = neural("VortexTransformer",
        GRID_CROSS_SEEDING, flow, start, patches, vt_ckpt, vt_script,
        args, args.random_seed)
    print("Running FGMA-VT with FEATURE_ADAPTIVE_MESH_SEEDING...")
    fg, fg_epoch, fg_points, fg_scores = neural("FGMA-VT",
        FEATURE_ADAPTIVE_MESH_SEEDING, flow, start, patches, fg_ckpt, fg_script,
        args, args.random_seed+100000)
    print(f"Running MVUNet with {mvunet.TRAINING_SEEDING} training data...")
    mv = mvunet.run(
        flow.u[0], flow.v[0], flow.x, flow.y, args,
        border_cells=args.border_cells,
    )
    mv_result = Result(
        "MVUNet", mv.probability, mv.probability,
        args.mvunet_threshold, mv.mask, mv.detections,
        mvunet.TRAINING_SEEDING,
    )
    print(
        f"Running VortexUNet with {vortexunet.TRAINING_SEEDING} "
        "training data..."
    )
    vu = vortexunet.run(
        flow.u[0], flow.v[0], flow.x, flow.y, args,
        border_cells=args.border_cells,
    )
    vu_result = Result(
        "VortexUNet", vu.probability, vu.probability,
        args.vortexunet_threshold, vu.mask, vu.detections,
        vortexunet.TRAINING_SEEDING,
    )
    results.extend((mv_result, vu_result, vt, fg))
    save_csv(args.csv_output, results)
    save_seeds(args.seed_scores_output, ((vt.name, vt.seeding, vt_points, vt_scores),
                                         (fg.name, fg.seeding, fg_points, fg_scores)))
    plot(args.output, flow, start, results)
    print(
        f"VortexTransformer epoch={vt_epoch}; FGMA-VT epoch={fg_epoch}; "
        f"MVUNet epoch={mv.epoch}; VortexUNet epoch={vu.epoch}"
    )
    for result in results:
        print(f"{result.name:20s}: {len(result.detections):2d} regions, "
              f"threshold={result.threshold:.6g}, seeding={result.seeding}")
    print(f"Visualization: {args.output} and {args.output.with_suffix('.pdf')}")
    print(f"Detections: {args.csv_output}\nSeed scores: {args.seed_scores_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
