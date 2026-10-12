"""Fit Vatistas vortex-profile parameters to real 2D flow patches, record their
statistical distribution, and re-sample it to synthesize new steady fields.
This reproduces the "Parameter Space Fitting" step of the VortexTransformer paper. See docs/vatistas_profile.md for the math.
Pipeline
--------
1. Load real unsteady flows via the same path used by FTLE_experiment.py
   (relocate_flow2d_dataset_folder + load_UnsteadyVectorFields_general) and slice
   them into spatial 32x32 velocity patches (sliding like a convolution kernel).
2. Fit each patch with a mixture of two steady Vatistas profiles (Eq. 10). Every
   profile has 7 continuous parameters (theta, sx, sy, rc, n, tx, ty) plus a
   discrete base shape S_i, so a patch has 14 continuous parameters. We minimize
   velocity MSE with 200 iterations of simulated annealing followed by 200
   iterations of gradient descent (following [KG19]).
3. Reject geometrically invalid fits, retain marginal statistics for reporting,
   and preserve valid parameter rows as a joint empirical distribution for
   re-sampling.

Usage
-----
    python FittingVatistasParam.py --config config/FittingVatistas.yaml
"""

from __future__ import annotations

import os
import json
import argparse
import logging

import numpy as np
import torch

from DeepUtils.utils import EasyConfig
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import (
    relocate_flow2d_dataset_folder,
    load_UnsteadyVectorFields_general,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Vatistas model constants
# ---------------------------------------------------------------------------
# 7 continuous parameters per profile, in this fixed order.
PARAM_NAMES = ["theta", "sx", "sy", "rc", "n", "tx", "ty"]
TH, SX, SY, RC, N, TX, TY = range(7)

# Eq. 4: the three constant base-shape matrices.
#   0 -> S1 saddle,  1 -> S2 center clockwise,  2 -> S3 center counter-clockwise
S_MATRICES = torch.tensor(
    [
        [[1.0, 0.0], [0.0, -1.0]],   # S1 saddle
        [[0.0, 1.0], [-1.0, 0.0]],   # S2 center (cw)
        [[0.0, -1.0], [1.0, 0.0]],   # S3 center (ccw)
    ],
    dtype=torch.float32,
)
SHAPE_LABELS = {0: "S1_saddle", 1: "S2_center_cw", 2: "S3_center_ccw"}

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Forward model  (batched + differentiable)
# ---------------------------------------------------------------------------
def vatistas_mixture_velocity(coords, params, shape_idx, bounds, profile_mask=None):
    """Evaluate the mixture-of-profiles velocity field (Eq. 9 & 10), batched.

    Args:
        coords:      [H, W, 2] query grid (patch-local frame, normalized to [-1,1]^2).
        params:      [P, M, 7] continuous parameters (theta, sx, sy, rc, n, tx, ty).
        shape_idx:   [P, M] long, base-shape index in {0,1,2}.
        bounds:      dict with 'rc','n','s' = (min,max) used to keep the math valid.
        profile_mask:[P, M] optional {0,1} multiplier to switch profiles off.

    Returns:
        velocity: [P, H, W, 2] summed over the M profiles.
    """
    P, M, _ = params.shape
    H, W = coords.shape[0], coords.shape[1]
    eps = 1e-9

    theta = params[..., TH]                                   # [P, M]
    # Clamp the parameters that would otherwise break the math (positive rc, n, s).
    rc = params[..., RC].clamp(bounds["rc"][0], bounds["rc"][1])
    n = params[..., N].clamp(bounds["n"][0], bounds["n"][1])
    sx = params[..., SX].clamp(bounds["s"][0], bounds["s"][1])
    sy = params[..., SY].clamp(bounds["s"][0], bounds["s"][1])
    tx = params[..., TX]
    ty = params[..., TY]

    cos, sin = torch.cos(theta), torch.sin(theta)             # [P, M]

    # Inverse deformation D^{-1} = 1/(sx*sy) * [[sy c, sy s], [-sx s, sx c]]  (det D = sx*sy)
    inv_det = 1.0 / (sx * sy)
    Di00 = inv_det * sy * cos
    Di01 = inv_det * sy * sin
    Di10 = inv_det * (-sx * sin)
    Di11 = inv_det * sx * cos

    Xp = coords[..., 0].view(1, 1, H, W)                      # [1,1,H,W]
    Yp = coords[..., 1].view(1, 1, H, W)

    def e(a):  # [P,M] -> [P,M,1,1] for broadcasting against the grid
        return a.view(P, M, 1, 1)

    # Pull the query point back into the base (undeformed) frame: x = D^{-1}(x'-T).
    dx = Xp - e(tx)
    dy = Yp - e(ty)
    bx = e(Di00) * dx + e(Di01) * dy                          # [P,M,H,W]
    by = e(Di10) * dx + e(Di11) * dy

    r = torch.sqrt(bx * bx + by * by + eps)                   # [P,M,H,W]

    # Eq. 3 factor g(r) = v0(r)/r = 1/(2*pi*rc^2) * (1 + (r/rc)^{2n})^{-1/n},
    # evaluated in log-space for numerical stability (no divide-by-r, no overflow).
    log_ratio = torch.log(r) - torch.log(e(rc))
    p2n = (2.0 * e(n)) * log_ratio
    p2n = p2n.clamp(max=30.0)
    log_bracket = torch.log1p(torch.exp(p2n))                 # log(1 + (r/rc)^{2n})
    g = torch.exp(-(1.0 / e(n)) * log_bracket) / (TWO_PI * e(rc) * e(rc))

    # Base velocity vb = g * (S . x)   (Eq. 2)
    S = S_MATRICES.to(params.device)[shape_idx]               # [P,M,2,2]
    Sx = S[..., 0, 0].unsqueeze(-1).unsqueeze(-1) * bx + S[..., 0, 1].unsqueeze(-1).unsqueeze(-1) * by
    Sy = S[..., 1, 0].unsqueeze(-1).unsqueeze(-1) * bx + S[..., 1, 1].unsqueeze(-1).unsqueeze(-1) * by
    vbx = g * Sx
    vby = g * Sy

    # Push the velocity forward: v' = D . vb   (Eq. 9), D = [[sx c, -sy s],[sx s, sy c]]
    vpx = e(sx * cos) * vbx + e(-sy * sin) * vby
    vpy = e(sx * sin) * vbx + e(sy * cos) * vby

    if profile_mask is not None:
        vpx = vpx * e(profile_mask)
        vpy = vpy * e(profile_mask)

    vx = vpx.sum(dim=1)                                        # [P,H,W]
    vy = vpy.sum(dim=1)
    return torch.stack([vx, vy], dim=-1)                      # [P,H,W,2]


def per_patch_mse(pred, target):
    """MSE per patch: [P,H,W,2] vs [P,H,W,2] -> [P]."""
    return ((pred - target) ** 2).reshape(pred.shape[0], -1).mean(dim=1)


def _cfg_value(section, name, default):
    """Read an optional EasyConfig/dict value without making it mandatory."""
    if section is None:
        return default
    if hasattr(section, "get"):
        return section.get(name, default)
    return getattr(section, name, default)


def _geometry_settings(cfg):
    """Geometry constraints shared by fitting, filtering, and re-sampling.

    The individual Vatistas parameters are not sufficient to bound the visible
    vortex core: its physical semi-axes are ``rc*sx`` and ``rc*sy``.  Defaults
    below are deliberately expressed in the normalized patch frame [-1,1]^2
    and can be overridden by ``model.geometry`` / ``optim`` in the YAML file.
    """
    geometry = _cfg_value(cfg.model, "geometry", {}) or {}
    optim = _cfg_value(cfg, "optim", {}) or {}
    return {
        "min_core_axis": float(_cfg_value(geometry, "min_core_axis", 0.05)),
        "max_core_axis": float(_cfg_value(geometry, "max_core_axis", 0.50)),
        "domain_limit": float(_cfg_value(geometry, "domain_limit", 0.95)),
        "center_limit": float(_cfg_value(geometry, "center_limit", 0.90)),
        "spacing_factor": float(_cfg_value(geometry, "spacing_factor", 1.50)),
        "spacing_margin": float(_cfg_value(geometry, "spacing_margin", 0.05)),
        "max_interaction_ratio": float(
            _cfg_value(geometry, "max_interaction_ratio", 0.15)),
        "max_center_shift_fraction": float(
            _cfg_value(geometry, "max_center_shift_fraction", 0.20)),
        "interaction_ring_samples": int(
            _cfg_value(geometry, "interaction_ring_samples", 32)),
        "max_fit_mse": float(_cfg_value(geometry, "max_fit_mse", 0.30)),
        "size_weight": float(_cfg_value(optim, "geometry_size_weight", 0.50)),
        "containment_weight": float(
            _cfg_value(optim, "geometry_containment_weight", 0.75)),
        "center_weight": float(_cfg_value(optim, "geometry_center_weight", 0.25)),
        "spacing_weight": float(_cfg_value(optim, "geometry_spacing_weight", 0.50)),
    }


def _physical_core_geometry(params):
    """Return physical semi-axes and axis-aligned extents for [P,M,7]."""
    theta = params[..., TH]
    axis_x = params[..., RC] * params[..., SX]
    axis_y = params[..., RC] * params[..., SY]
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    extent_x = torch.sqrt(
        (axis_x * cos_t) ** 2 + (axis_y * sin_t) ** 2 + 1e-12)
    extent_y = torch.sqrt(
        (axis_x * sin_t) ** 2 + (axis_y * cos_t) ** 2 + 1e-12)
    return axis_x, axis_y, extent_x, extent_y


def fit_objective(pred, target, params, cfg):
    """Per-patch fitting score = velocity MSE + vortex-geometry penalties.

    Returning both the score and the unregularized MSE lets optimization select
    physically valid solutions without hiding reconstruction quality.
    """
    settings = _geometry_settings(cfg)
    data_mse = per_patch_mse(pred, target)
    axis_x, axis_y, extent_x, extent_y = _physical_core_geometry(params)

    min_axis = settings["min_core_axis"]
    max_axis = settings["max_core_axis"]
    size_penalty = (
        torch.relu(min_axis - axis_x) ** 2
        + torch.relu(min_axis - axis_y) ** 2
        + torch.relu(axis_x - max_axis) ** 2
        + torch.relu(axis_y - max_axis) ** 2
    ).mean(dim=1)

    tx, ty = params[..., TX], params[..., TY]
    domain_limit = settings["domain_limit"]
    containment_penalty = (
        torch.relu(tx.abs() + extent_x - domain_limit) ** 2
        + torch.relu(ty.abs() + extent_y - domain_limit) ** 2
    ).mean(dim=1)
    center_penalty = (
        torch.relu(tx.abs() - settings["center_limit"]) ** 2
        + torch.relu(ty.abs() - settings["center_limit"]) ** 2
    ).mean(dim=1)

    spacing_penalty = torch.zeros_like(data_mse)
    profile_count = params.shape[1]
    if profile_count > 1:
        # A conservative non-overlap test using each ellipse's circumscribed
        # circle.  It is stricter than comparing the undeformed rc values.
        outer_radius = torch.maximum(axis_x, axis_y)
        pair_penalties = []
        for first in range(profile_count):
            for second in range(first + 1, profile_count):
                distance = torch.sqrt(
                    (tx[:, first] - tx[:, second]) ** 2
                    + (ty[:, first] - ty[:, second]) ** 2
                    + 1e-12)
                required = (
                    settings["spacing_factor"]
                    * (outer_radius[:, first] + outer_radius[:, second])
                    + settings["spacing_margin"])
                pair_penalties.append(torch.relu(required - distance) ** 2)
        spacing_penalty = torch.stack(pair_penalties, dim=1).mean(dim=1)

    score = (
        data_mse
        + settings["size_weight"] * size_penalty
        + settings["containment_weight"] * containment_penalty
        + settings["center_weight"] * center_penalty
        + settings["spacing_weight"] * spacing_penalty
    )
    return score, data_mse


def _swirling_strength_peak(velocity, margin=3):
    """Maximum 2-D swirling strength of one normalized [H,W,2] patch."""
    height, width = velocity.shape[:2]
    dy = 2.0 / max(1, height - 1)
    dx = 2.0 / max(1, width - 1)
    du_dy, du_dx = np.gradient(velocity[..., 0], dy, dx)
    dv_dy, dv_dx = np.gradient(velocity[..., 1], dy, dx)
    trace = du_dx + dv_dy
    determinant = du_dx * dv_dy - du_dy * dv_dx
    swirl = 0.5 * np.sqrt(np.maximum(4.0 * determinant - trace * trace, 0.0))
    margin = max(0, int(margin))
    if margin > 0 and height > 2 * margin and width > 2 * margin:
        swirl = swirl[margin:-margin, margin:-margin]
    return float(np.max(swirl)) if swirl.size else 0.0


# ---------------------------------------------------------------------------
# Data: load real flows and slice into velocity patches
# ---------------------------------------------------------------------------
def _clamp_time_range(t_start, t_end, n_steps):
    a = 0 if t_start is None else int(t_start)
    b = n_steps if t_end is None else int(t_end)
    a = max(0, min(a, n_steps))
    b = max(0, min(b, n_steps))
    if b <= a:
        a, b = 0, n_steps
    return a, b


def extract_patches(cfg, device):
    """Load the configured real flows and tile every requested time slice into
    (patch_size x patch_size) velocity patches slid with `patch_stride`.

    Returns:
        targets: [P, H, W, 2] float32 (RMS-normalized if cfg.dataset.normalize_patch)
        scales:  [P] the per-patch RMS speed removed by normalization
        meta:    list of dicts describing each patch's origin
    """
    dcfg = cfg.dataset
    ps = int(dcfg.patch_size)
    stride = int(dcfg.patch_stride)
    min_rms = float(dcfg.min_patch_rms)
    normalize = bool(dcfg.normalize_patch)
    remove_mean = bool(dcfg.get("remove_mean_velocity", True))
    min_residual_fraction = float(dcfg.get("min_residual_fraction", 0.05))
    min_swirl_strength = float(dcfg.get("min_swirl_strength", 0.05))
    swirl_margin = int(dcfg.get("swirl_margin", 3))

    field_specs = list(dcfg.fields)
    names = [str(f["name"]) for f in field_specs]
    logging.info(f"[patches] loading real flows: {names}")
    fields = load_UnsteadyVectorFields_general(dcfg.dat_dir, names)

    patches, scales, meta = [], [], []
    skipped_low_speed = 0
    skipped_low_residual = 0
    skipped_no_swirl = 0
    for spec, vf in zip(field_specs, fields):
        if vf is None:
            logging.warning(f"[patches] field '{spec['name']}' failed to load; skip.")
            continue
        field = vf.field
        field = field.cpu().numpy() if isinstance(field, torch.Tensor) else np.asarray(field)
        field = field.astype(np.float32)                      # [T, Y, X, 2]
        T, Y, X = field.shape[0], field.shape[1], field.shape[2]
        a, b = _clamp_time_range(spec.get("t_start"), spec.get("t_end"), T)
        t_stride = max(1, int(spec.get("t_stride", 1)))
        t_indices = list(range(a, b, t_stride))
        if Y < ps or X < ps:
            logging.warning(f"[patches] field '{spec['name']}' ({Y}x{X}) smaller than patch {ps}; skip.")
            continue

        row_starts = list(range(0, Y - ps + 1, stride))
        col_starts = list(range(0, X - ps + 1, stride))
        n_before = len(patches)
        for t in t_indices:
            slice_t = field[t]                                # [Y, X, 2]
            for y0 in row_starts:
                for x0 in col_starts:
                    p = slice_t[y0:y0 + ps, x0:x0 + ps, :].copy()  # [ps, ps,2]
                    raw_rms = float(np.sqrt(np.mean(
                        p[..., 0] ** 2 + p[..., 1] ** 2)))
                    if raw_rms < min_rms:
                        skipped_low_speed += 1
                        continue

                    # A constant background flow is a nuisance component for
                    # vortex-profile fitting.  Without removing it, the lowest-MSE
                    # solution is often a huge vortex centered outside the patch,
                    # whose far field approximates uniform translation.
                    background = p.mean(axis=(0, 1), keepdims=True)
                    working = p - background if remove_mean else p
                    residual_rms = float(np.sqrt(np.mean(
                        working[..., 0] ** 2 + working[..., 1] ** 2)))
                    if (
                        residual_rms < min_rms
                        or residual_rms / max(raw_rms, 1e-12) < min_residual_fraction
                    ):
                        skipped_low_residual += 1
                        continue

                    scale = residual_rms if normalize else 1.0
                    normalized = (
                        working / (scale + 1e-12) if normalize else working)
                    swirl_peak = _swirling_strength_peak(normalized, swirl_margin)
                    if swirl_peak < min_swirl_strength:
                        skipped_no_swirl += 1
                        continue

                    patches.append(normalized.astype(np.float32, copy=False))
                    scales.append(scale)
                    meta.append({
                        "field": str(spec["name"]), "t": int(t),
                        "y0": int(y0), "x0": int(x0),
                        "background_u": float(background[..., 0].item()),
                        "background_v": float(background[..., 1].item()),
                        "raw_rms": raw_rms,
                        "residual_rms": residual_rms,
                        "swirl_peak": swirl_peak,
                    })
        logging.info(f"[patches] '{spec['name']}': {len(t_indices)} slices -> {len(patches) - n_before} patches "
                     f"(time steps {a}:{b}:{t_stride}, grid {Y}x{X}).")

    if not patches:
        raise RuntimeError("No patches extracted. Check dataset paths / field names / min_patch_rms.")

    targets = np.stack(patches, axis=0).astype(np.float32)    # [P, ps, ps, 2]
    scales = np.asarray(scales, dtype=np.float32)

    # Cap the patch count (paper fits ~1500) with a reproducible random subsample.
    max_patches = int(dcfg.max_patches)
    if targets.shape[0] > max_patches:
        idx = np.random.default_rng(int(cfg.get("seed", 0))).permutation(targets.shape[0])[:max_patches]
        idx.sort()
        targets, scales, meta = targets[idx], scales[idx], [meta[i] for i in idx]
        logging.info(f"[patches] subsampled to max_patches={max_patches}.")

    logging.info(f"[patches] total {targets.shape[0]} patches of size {ps}x{ps}.")
    logging.info(
        "[patches] preprocessing remove_mean=%s; skipped low-speed=%d, "
        "low-residual=%d, no-swirl=%d (min residual fraction %.3f, "
        "min swirl %.3f).",
        remove_mean, skipped_low_speed, skipped_low_residual, skipped_no_swirl,
        min_residual_fraction, min_swirl_strength)
    return torch.from_numpy(targets).to(device), torch.from_numpy(scales).to(device), meta


def make_local_grid(ps, device):
    """Patch-local query grid on [-1,1]^2: coords[i,j] = (x_j, y_i), shape [ps,ps,2]."""
    lin = torch.linspace(-1.0, 1.0, ps, device=device)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    return torch.stack([xx, yy], dim=-1)                      # [ps, ps, 2]


# ---------------------------------------------------------------------------
# Optimization: simulated annealing + gradient descent
# ---------------------------------------------------------------------------
def _project(params, shape_idx, bounds, rng=None):
    """Clamp continuous params into valid ranges (theta is left free; wrapped later)."""
    with torch.no_grad():
        params[..., RC].clamp_(bounds["rc"][0], bounds["rc"][1])
        params[..., N].clamp_(bounds["n"][0], bounds["n"][1])
        params[..., SX].clamp_(bounds["s"][0], bounds["s"][1])
        params[..., SY].clamp_(bounds["s"][0], bounds["s"][1])
        params[..., TX].clamp_(-bounds["t"], bounds["t"])
        params[..., TY].clamp_(-bounds["t"], bounds["t"])
    return params


def datadriven_seeds(target, coords, cfg, M):
    """Detect up to M vortex cores per patch from the vorticity field and return
    per-profile seed (tx, ty, base-shape). Peaks of |omega| localize the cores; the
    sign of omega picks ccw (S3) vs cw (S2). Successive peaks are suppressed in a disk
    so the two profiles seed different vortices. This gives SA/GD a physically-informed
    starting basin instead of a random one."""
    P, H, W, _ = target.shape
    u, v = target[..., 0], target[..., 1]                        # [P,H,W]
    dvdx = torch.gradient(v, spacing=2.0 / (W - 1), dim=2)[0]
    dudy = torch.gradient(u, spacing=2.0 / (H - 1), dim=1)[0]
    omega = dvdx - dudy                                          # vorticity [P,H,W]
    oabs = omega.abs().clone()

    # One-sided derivatives and cropped coherent structures make patch borders
    # unreliable seed locations.  Keep initial cores in the interior; geometry
    # regularization later prevents them from drifting back outside the patch.
    seed_margin = int(cfg.optim.get("seed_margin", 3))
    if seed_margin > 0 and H > 2 * seed_margin and W > 2 * seed_margin:
        oabs[:, :seed_margin, :] = 0
        oabs[:, -seed_margin:, :] = 0
        oabs[:, :, :seed_margin] = 0
        oabs[:, :, -seed_margin:] = 0

    cx, cy = coords[..., 0], coords[..., 1]                      # [H,W]
    cand = [int(c) for c in cfg.model.shape_candidates]
    has_cw, has_ccw = (1 in cand), (2 in cand)
    dev = target.device
    ii, jj = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    rad = max(2, int(0.25 * min(H, W)))                          # suppression radius (grid cells)

    tx0 = torch.zeros(P, M, device=dev)
    ty0 = torch.zeros(P, M, device=dev)
    shape0 = torch.zeros(P, M, dtype=torch.long, device=dev)
    ar = torch.arange(P, device=dev)
    for m in range(M):
        idx = oabs.reshape(P, -1).argmax(dim=1)                  # [P] flat peak
        pi, pj = idx // W, idx % W
        tx0[:, m], ty0[:, m] = cx[pi, pj], cy[pi, pj]
        w_at = omega[ar, pi, pj]
        s = torch.where(w_at >= 0, torch.full_like(idx, 2), torch.full_like(idx, 1))  # ccw/cw
        if not has_ccw:
            s = torch.where(s == 2, torch.full_like(s, 1), s)
        if not has_cw:
            s = torch.where(s == 1, torch.full_like(s, 2), s)
        shape0[:, m] = s
        d2 = (ii.view(1, H, W) - pi.view(P, 1, 1)) ** 2 + (jj.view(1, H, W) - pj.view(P, 1, 1)) ** 2
        oabs = torch.where(d2 <= rad * rad, torch.zeros_like(oabs), oabs)
    return tx0, ty0, shape0


def init_params(P, M, cfg, bounds, device, restart=0, seeds=None):
    """Random-but-plausible initial parameters and base shapes for every profile.
    `restart` varies the RNG so multi-restart fitting explores different basins.
    If `seeds`=(tx0,ty0,shape0) is given (data-driven mode), core positions/shapes are
    initialized from vortex detection; restart 0 uses them exactly, later restarts add
    growing jitter + random shape flips to retain exploration."""
    g = torch.Generator(device="cpu").manual_seed(int(cfg.get("seed", 0)) + 9973 * int(restart))

    def u(lo, hi, shape):
        return (lo + (hi - lo) * torch.rand(shape, generator=g)).to(device)

    params = torch.empty(P, M, 7, device=device)
    params[..., TH] = u(-np.pi, np.pi, (P, M))
    params[..., SX] = u(0.5, 1.5, (P, M))
    params[..., SY] = u(0.5, 1.5, (P, M))
    params[..., RC] = u(0.2, 0.6, (P, M))
    params[..., N] = u(1.0, 3.0, (P, M))
    candidates = torch.tensor([int(c) for c in cfg.model.shape_candidates], dtype=torch.long)

    if seeds is not None:
        tx0, ty0, shape0 = seeds
        jit = 0.0 if restart == 0 else min(0.5, 0.12 * restart)
        params[..., TX] = tx0 + jit * torch.randn((P, M), generator=g).to(device)
        params[..., TY] = ty0 + jit * torch.randn((P, M), generator=g).to(device)
        if restart == 0:
            shape_idx = shape0.clone()
        else:
            flip = (torch.rand((P, M), generator=g).to(device) < 0.3)
            rnd = candidates[torch.randint(0, len(candidates), (P, M), generator=g)].to(device)
            shape_idx = torch.where(flip, rnd, shape0)
    else:
        params[..., TX] = u(-0.5, 0.5, (P, M))
        params[..., TY] = u(-0.5, 0.5, (P, M))
        pick = torch.randint(0, len(candidates), (P, M), generator=g)
        shape_idx = candidates[pick].to(device)
    return _project(params, shape_idx, bounds), shape_idx


def simulated_annealing(coords, target, params, shape_idx, cfg, bounds):
    """Batched Metropolis simulated annealing over all patches simultaneously."""
    ocfg = cfg.optim
    iters = int(ocfg.sa_iters)
    T0, Tend = float(ocfg.sa_T0), float(ocfg.sa_Tend)
    flip_p = float(ocfg.sa_shape_flip_p)
    cooling = (Tend / T0) ** (1.0 / max(1, iters))
    P, M, _ = params.shape
    device = params.device
    candidates = torch.tensor([int(c) for c in cfg.model.shape_candidates], dtype=torch.long, device=device)

    # per-parameter proposal scale (physical units), annealed with temperature
    sigma = torch.tensor([0.5, 0.3, 0.3, 0.15, 0.5, 0.2, 0.2], device=device)

    with torch.no_grad():
        cur = params.clone()
        cur_shape = shape_idx.clone()
        cur_score, cur_mse = fit_objective(
            vatistas_mixture_velocity(coords, cur, cur_shape, bounds),
            target, cur, cfg)
        best, best_shape = cur.clone(), cur_shape.clone()
        best_score, best_mse = cur_score.clone(), cur_mse.clone()

        T = T0
        for it in range(iters):
            frac = T / T0
            prop = cur + torch.randn_like(cur) * (sigma.view(1, 1, 7) * frac)
            prop = _project(prop, cur_shape, bounds)

            prop_shape = cur_shape
            if flip_p > 0.0:
                flip = torch.rand(P, M, device=device) < flip_p
                rnd_shape = candidates[torch.randint(0, len(candidates), (P, M), device=device)]
                prop_shape = torch.where(flip, rnd_shape, cur_shape)

            prop_score, prop_mse = fit_objective(
                vatistas_mixture_velocity(coords, prop, prop_shape, bounds),
                target, prop, cfg)

            delta = prop_score - cur_score
            accept = (delta < 0) | (torch.rand(P, device=device) < torch.exp(-delta / max(T, 1e-8)))
            a = accept.view(P, 1, 1)
            cur = torch.where(a, prop, cur)
            cur_shape = torch.where(accept.view(P, 1), prop_shape, cur_shape)
            cur_score = torch.where(accept, prop_score, cur_score)
            cur_mse = torch.where(accept, prop_mse, cur_mse)

            improved = cur_score < best_score
            b = improved.view(P, 1, 1)
            best = torch.where(b, cur, best)
            best_shape = torch.where(improved.view(P, 1), cur_shape, best_shape)
            best_score = torch.where(improved, cur_score, best_score)
            best_mse = torch.where(improved, cur_mse, best_mse)

            T *= cooling
            if (it + 1) % 50 == 0 or it == iters - 1:
                logging.info(
                    f"[SA] iter {it+1}/{iters}  T={T:.4g}  "
                    f"mean score={best_score.mean().item():.6f}  "
                    f"mean mse={best_mse.mean().item():.6f}")

    return best, best_shape, best_score


def gradient_descent(coords, target, params, shape_idx, cfg, bounds):
    """Adam refinement of the continuous parameters (base shapes held fixed).

    The learning rate follows a cosine decay from gd_lr to gd_lr_final; with more GD
    iterations the decay lets the fit settle into a lower-MSE minimum than a fixed lr,
    which plateaus. Set gd_lr_final == gd_lr for a constant rate."""
    iters = int(cfg.optim.gd_iters)
    lr0 = float(cfg.optim.gd_lr)
    lr1 = float(cfg.optim.get("gd_lr_final", lr0))
    p = params.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr0)

    for it in range(iters):
        cur_lr = lr1 + 0.5 * (lr0 - lr1) * (1.0 + np.cos(np.pi * it / max(1, iters - 1)))
        for grp in opt.param_groups:
            grp["lr"] = cur_lr
        opt.zero_grad()
        pred = vatistas_mixture_velocity(coords, p, shape_idx, bounds)
        score_per_patch, mse_per_patch = fit_objective(pred, target, p, cfg)
        loss = score_per_patch.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p], max_norm=10.0)
        opt.step()
        _project(p, shape_idx, bounds)
        if (it + 1) % 50 == 0 or it == iters - 1:
            logging.info(
                f"[GD] iter {it+1}/{iters}  lr={cur_lr:.4g}  "
                f"mean score={loss.item():.6f}  "
                f"mean mse={mse_per_patch.mean().item():.6f}")

    with torch.no_grad():
        final = p.detach()
        final_score, final_mse = fit_objective(
            vatistas_mixture_velocity(coords, final, shape_idx, bounds),
            target, final, cfg)
    return final, final_mse, final_score


def lbfgs_polish(coords, target, params, shape_idx, cfg, bounds):
    """Second-order (L-BFGS + strong-Wolfe line search) polish after Adam. L-BFGS
    converges to a tighter minimum of this smooth MSE surface than first-order Adam,
    squeezing extra PSNR. rc/n/sx/sy stay valid via the forward's internal clamps;
    tx/ty are projected once at the end. No-op when optim.lbfgs_iters <= 0."""
    iters = int(cfg.optim.get("lbfgs_iters", 0))
    if iters <= 0:
        with torch.no_grad():
            score, mse = fit_objective(
                vatistas_mixture_velocity(coords, params, shape_idx, bounds),
                target, params, cfg)
            return params, mse, score
    p = params.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([p], max_iter=iters, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        score, _ = fit_objective(
            vatistas_mixture_velocity(coords, p, shape_idx, bounds),
            target, p, cfg)
        loss = score.sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        _project(p, shape_idx, bounds)
        final = p.detach()
        final_score, final_mse = fit_objective(
            vatistas_mixture_velocity(coords, final, shape_idx, bounds),
            target, final, cfg)
    # guard: never let the polish worsen a patch (line-search overshoot) or emit NaN
    orig_score, orig_mse = fit_objective(
        vatistas_mixture_velocity(coords, params, shape_idx, bounds),
        target, params, cfg)
    bad = (final_score > orig_score) | ~torch.isfinite(final_score)
    if bad.any():
        b = bad.view(-1, 1, 1)
        final = torch.where(b, params, final)
        final_score = torch.where(bad, orig_score, final_score)
        final_mse = torch.where(bad, orig_mse, final_mse)
    return final, final_mse, final_score


def fit_patches(coords, target, cfg, bounds, M, device):
    """Full fit for every patch: simulated annealing -> gradient descent, repeated
    from `cfg.optim.n_restarts` random inits with the best-per-patch result kept
    (multi-restart is the main lever for escaping SA/GD local minima)."""
    P = target.shape[0]
    n_restarts = int(cfg.optim.get("n_restarts", 1))
    seeds = None
    if str(cfg.optim.get("init_mode", "random")) == "datadriven":
        seeds = datadriven_seeds(target, coords, cfg, M)
    best_params = best_shape = best_loss = best_score = None
    for r in range(n_restarts):
        params, shape_idx = init_params(P, M, cfg, bounds, device, restart=r, seeds=seeds)
        params, shape_idx, _ = simulated_annealing(coords, target, params, shape_idx, cfg, bounds)
        params, loss, score = gradient_descent(
            coords, target, params, shape_idx, cfg, bounds)
        if int(cfg.optim.get("lbfgs_iters", 0)) > 0:
            params, loss, score = lbfgs_polish(
                coords, target, params, shape_idx, cfg, bounds)
        if best_score is None:
            best_params, best_shape = params, shape_idx
            best_loss, best_score = loss, score
        else:
            take = score < best_score                     # [P]
            b = take.view(P, 1, 1)
            best_params = torch.where(b, params, best_params)
            best_shape = torch.where(take.view(P, 1), shape_idx, best_shape)
            best_loss = torch.where(take, loss, best_loss)
            best_score = torch.where(take, score, best_score)
        if n_restarts > 1:
            logging.info(
                f"[fit] restart {r+1}/{n_restarts}  "
                f"this score={score.mean().item():.6f}, mse={loss.mean().item():.6f}  "
                f"best score={best_score.mean().item():.6f}, "
                f"mse={best_loss.mean().item():.6f}")
    return best_params, best_shape, best_loss


def fit_quality_report(coords, target, params, shape_idx, bounds, tag="fit"):
    """Compute and log MSE + PSNR of the fitted mixture vs the (normalized) patches.

    PSNR uses each patch's own peak-to-peak range as the reference (the convention in
    FTLE_experiment._sr_metrics), then averages over patches. Returns a metrics dict.
    """
    with torch.no_grad():
        pred = vatistas_mixture_velocity(coords, params, shape_idx, bounds)
        P = target.shape[0]
        se = (pred - target) ** 2
        mse_pp = se.reshape(P, -1).mean(dim=1)                          # [P]
        tgt = target.reshape(P, -1)
        rng = (tgt.max(dim=1).values - tgt.min(dim=1).values).clamp_min(1e-8)
        psnr_pp = 20.0 * torch.log10(rng) - 10.0 * torch.log10(mse_pp.clamp_min(1e-12))
        global_mse = se.mean()
        # global PSNR against the overall data range (single number for the whole set)
        grng = (target.max() - target.min()).clamp_min(1e-8)
        global_psnr = 20.0 * torch.log10(grng) - 10.0 * torch.log10(global_mse.clamp_min(1e-12))

    m = {
        "mse_mean": float(mse_pp.mean()), "mse_median": float(mse_pp.median()),
        "psnr_mean": float(psnr_pp.mean()), "psnr_median": float(psnr_pp.median()),
        "global_mse": float(global_mse), "global_psnr": float(global_psnr),
    }
    logging.info(f"[{tag}] velocity MSE  mean={m['mse_mean']:.6f}  median={m['mse_median']:.6f}  "
                 f"global={m['global_mse']:.6f}")
    logging.info(f"[{tag}] PSNR (dB)     mean={m['psnr_mean']:.2f}  median={m['psnr_median']:.2f}  "
                 f"global={m['global_psnr']:.2f}")
    return m


# ---------------------------------------------------------------------------
# Valid-profile distribution fitting + geometry-aware generation
# ---------------------------------------------------------------------------
def profile_contributions(coords, params, shape_idx, bounds):
    """Per-profile RMS energy fraction within a patch -> [P, M] in [0,1].

    Used to drop degenerate (near-zero) profiles before building the distribution.
    """
    P, M, _ = params.shape
    with torch.no_grad():
        rms = torch.zeros(P, M, device=params.device)
        for m in range(M):
            mask = torch.zeros(P, M, device=params.device)
            mask[:, m] = 1.0
            v = vatistas_mixture_velocity(coords, params, shape_idx, bounds, profile_mask=mask)
            rms[:, m] = torch.sqrt((v ** 2).reshape(P, -1).mean(dim=1) + 1e-12)
        total = rms.sum(dim=1, keepdim=True) + 1e-12
        return rms / total


def valid_profile_mask(params, shape_idx, fit_loss, contribution, cfg, bounds):
    """Select profiles suitable for a clean synthetic-label distribution.

    Energy contribution alone is insufficient (and is identically one when
    fitting a single profile).  Reject off-domain, oversized, boundary-clamped,
    or poorly fitted profiles before computing any parameter statistics.
    """
    settings = _geometry_settings(cfg)
    axis_x, axis_y, extent_x, extent_y = _physical_core_geometry(params)
    tx, ty = params[..., TX], params[..., TY]

    contribution_ok = contribution >= float(cfg.generate.min_contribution)
    size_ok = (
        (axis_x >= settings["min_core_axis"])
        & (axis_x <= settings["max_core_axis"])
        & (axis_y >= settings["min_core_axis"])
        & (axis_y <= settings["max_core_axis"])
    )
    contained = (
        (tx.abs() + extent_x <= settings["domain_limit"])
        & (ty.abs() + extent_y <= settings["domain_limit"])
        & (tx.abs() <= settings["center_limit"])
        & (ty.abs() <= settings["center_limit"])
    )
    fit_ok = fit_loss.view(-1, 1) <= settings["max_fit_mse"]

    exclude_bound_hits = bool(_cfg_value(
        _cfg_value(cfg.model, "geometry", {}) or {},
        "exclude_bound_hits", True))
    if exclude_bound_hits:
        eps = 1e-4
        not_clamped = (
            (params[..., RC] > bounds["rc"][0] + eps)
            & (params[..., RC] < bounds["rc"][1] - eps)
            & (params[..., SX] > bounds["s"][0] + eps)
            & (params[..., SX] < bounds["s"][1] - eps)
            & (params[..., SY] > bounds["s"][0] + eps)
            & (params[..., SY] < bounds["s"][1] - eps)
            & (tx.abs() < bounds["t"] - eps)
            & (ty.abs() < bounds["t"] - eps)
        )
    else:
        not_clamped = torch.ones_like(contribution_ok)

    preliminary = contribution_ok & size_ok & contained & fit_ok & not_clamped
    row_spacing_ok = np.ones(params.shape[0], dtype=bool)
    row_interaction_ok = np.ones(params.shape[0], dtype=bool)
    params_np = params.detach().cpu().numpy()
    shapes_np = shape_idx.detach().cpu().numpy()
    preliminary_np = preliminary.detach().cpu().numpy()
    for patch_index in range(params.shape[0]):
        profiles = [
            (params_np[patch_index, profile_index],
             int(shapes_np[patch_index, profile_index]))
            for profile_index in range(params.shape[1])
            if preliminary_np[patch_index, profile_index]
        ]
        accepted = []
        for profile, shape in profiles:
            if accepted and not _profiles_are_spaced(profile, accepted, cfg):
                row_spacing_ok[patch_index] = False
                break
            accepted.append((profile, shape))
        if len(profiles) > 1 and not _profiles_have_low_interaction(
                profiles, cfg, bounds):
            row_interaction_ok[patch_index] = False
    row_spacing_t = torch.from_numpy(row_spacing_ok).to(params.device)
    row_interaction_t = torch.from_numpy(row_interaction_ok).to(params.device)
    valid = (
        preliminary
        & row_spacing_t[:, None]
        & row_interaction_t[:, None])
    total = int(valid.numel())
    logging.info(
        "[fit-health] valid profiles %d/%d; contribution=%d, size=%d, "
        "contained=%d, fit=%d, not-clamped=%d; spaced rows=%d/%d, "
        "low-interaction rows=%d/%d.",
        int(valid.sum()), total, int(contribution_ok.sum()), int(size_ok.sum()),
        int(contained.sum()), int(fit_ok.expand_as(valid).sum()),
        int(not_clamped.sum()), int(row_spacing_ok.sum()), len(row_spacing_ok),
        int(row_interaction_ok.sum()), len(row_interaction_ok))

    if not valid.any():
        raise RuntimeError(
            "No fitted profile passed the geometry/quality checks. Inspect the "
            "fit-health log or relax model.geometry only after verifying labels.")
    return valid


def fit_distribution(params, shape_idx, active_mask):
    """Summarize valid profiles and retain their joint empirical distribution."""
    prm = params.reshape(-1, 7).cpu().numpy()
    shp = shape_idx.reshape(-1).cpu().numpy()
    keep = active_mask.reshape(-1).cpu().numpy().astype(bool)
    prm, shp = prm[keep], shp[keep]

    # theta is an angle: wrap to (-pi, pi] and report circular mean.
    theta = np.arctan2(np.sin(prm[:, TH]), np.cos(prm[:, TH]))
    prm = prm.copy()
    prm[:, TH] = theta

    dist = {
        "num_profiles": int(prm.shape[0]),
        "sampling": "empirical_joint",
        "params": {},
    }
    for i, name in enumerate(PARAM_NAMES):
        col = prm[:, i]
        if name == "theta":
            c_mean = float(np.arctan2(np.sin(col).mean(), np.cos(col).mean()))
            resultant = float(np.hypot(np.sin(col).mean(), np.cos(col).mean()))
            circular_std = float(np.sqrt(max(0.0, -2.0 * np.log(
                max(resultant, 1e-12)))))
            dist["params"][name] = {"mean": c_mean, "std": circular_std,
                                    "min": float(col.min()), "max": float(col.max())}
        else:
            dist["params"][name] = {"mean": float(col.mean()), "std": float(col.std()),
                                    "min": float(col.min()), "max": float(col.max())}

    shapes, counts = np.unique(shp, return_counts=True)
    probs = counts / counts.sum()
    dist["shape_distribution"] = {SHAPE_LABELS[int(s)]: float(pr) for s, pr in zip(shapes, probs)}
    dist["shape_index_probs"] = {int(s): float(pr) for s, pr in zip(shapes, probs)}
    # Preserve parameter correlations (especially rc*sx/rc*sy and center/size)
    # for re-sampling.  The marginal statistics above remain for compatibility
    # and diagnostics; new code samples complete valid rows from these arrays.
    dist["empirical_params"] = prm.astype(np.float32).tolist()
    dist["empirical_shapes"] = shp.astype(np.int64).tolist()
    return dist


def _sample_gaussian(spec, rng):
    """Draw one scalar from a fitted per-parameter Gaussian N(mean, std)."""
    return rng.normal(spec["mean"], max(spec["std"], 1e-6))


def _numpy_profile_geometry_valid(p, cfg, bounds):
    settings = _geometry_settings(cfg)
    theta, sx, sy, rc = float(p[TH]), float(p[SX]), float(p[SY]), float(p[RC])
    tx, ty = float(p[TX]), float(p[TY])
    if not (
        bounds["rc"][0] <= rc <= bounds["rc"][1]
        and bounds["n"][0] <= float(p[N]) <= bounds["n"][1]
        and bounds["s"][0] <= sx <= bounds["s"][1]
        and bounds["s"][0] <= sy <= bounds["s"][1]
        and abs(tx) <= bounds["t"]
        and abs(ty) <= bounds["t"]
    ):
        return False
    axis_x, axis_y = rc * sx, rc * sy
    if not (
        settings["min_core_axis"] <= axis_x <= settings["max_core_axis"]
        and settings["min_core_axis"] <= axis_y <= settings["max_core_axis"]
    ):
        return False
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    extent_x = np.sqrt((axis_x * cos_t) ** 2 + (axis_y * sin_t) ** 2)
    extent_y = np.sqrt((axis_x * sin_t) ** 2 + (axis_y * cos_t) ** 2)
    return bool(
        abs(tx) <= settings["center_limit"]
        and abs(ty) <= settings["center_limit"]
        and abs(tx) + extent_x <= settings["domain_limit"]
        and abs(ty) + extent_y <= settings["domain_limit"])


def _numpy_single_profile_velocity(points, p, shape, bounds):
    """Evaluate one profile with the same pull-back/push-forward as PyTorch."""
    points = np.asarray(points, dtype=np.float64)
    theta = float(p[TH])
    sx = float(np.clip(p[SX], bounds["s"][0], bounds["s"][1]))
    sy = float(np.clip(p[SY], bounds["s"][0], bounds["s"][1]))
    rc = float(np.clip(p[RC], bounds["rc"][0], bounds["rc"][1]))
    exponent = float(np.clip(p[N], bounds["n"][0], bounds["n"][1]))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    deformation = np.array(
        [[sx * cos_t, -sy * sin_t],
         [sx * sin_t, sy * cos_t]], dtype=np.float64)
    inverse_deformation = np.linalg.inv(deformation)
    center = np.array([p[TX], p[TY]], dtype=np.float64)
    base_points = (points - center) @ inverse_deformation.T
    radius = np.sqrt(np.sum(base_points * base_points, axis=-1) + 1e-9)
    log_ratio = np.log(radius) - np.log(rc)
    log_bracket = np.logaddexp(
        0.0, np.minimum(2.0 * exponent * log_ratio, 30.0))
    radial_factor = np.exp(-log_bracket / exponent) / (
        TWO_PI * rc * rc)
    shape_matrix = np.asarray(
        [[[1.0, 0.0], [0.0, -1.0]],
         [[0.0, 1.0], [-1.0, 0.0]],
         [[0.0, -1.0], [1.0, 0.0]]], dtype=np.float64)[int(shape)]
    base_velocity = (base_points @ shape_matrix.T) * radial_factor[..., None]
    return base_velocity @ deformation.T


def _profile_characteristic_speed(p, shape, cfg, bounds):
    """Peak self-induced speed on the profile's nominal core ellipse."""
    samples = max(8, _geometry_settings(cfg)["interaction_ring_samples"])
    angles = np.linspace(0.0, TWO_PI, samples, endpoint=False)
    rc = float(np.clip(p[RC], bounds["rc"][0], bounds["rc"][1]))
    base_ring = rc * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    theta = float(p[TH])
    sx = float(np.clip(p[SX], bounds["s"][0], bounds["s"][1]))
    sy = float(np.clip(p[SY], bounds["s"][0], bounds["s"][1]))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    deformation = np.array(
        [[sx * cos_t, -sy * sin_t],
         [sx * sin_t, sy * cos_t]], dtype=np.float64)
    center = np.array([p[TX], p[TY]], dtype=np.float64)
    world_ring = base_ring @ deformation.T + center
    speed = np.linalg.norm(
        _numpy_single_profile_velocity(world_ring, p, shape, bounds), axis=-1)
    return max(float(np.max(speed)), 1e-12)


def _profile_interaction_metrics(profile_index, profiles, cfg, bounds):
    """Return induced-speed ratio and estimated core-center displacement."""
    p, shape = profiles[profile_index]
    if int(shape) not in (1, 2):
        return 0.0, 0.0
    center = np.array([p[TX], p[TY]], dtype=np.float64)
    induced = np.zeros(2, dtype=np.float64)
    for other_index, (other, other_shape) in enumerate(profiles):
        if other_index != profile_index:
            induced += _numpy_single_profile_velocity(
                center, other, other_shape, bounds)
    interaction_ratio = float(np.linalg.norm(induced)) / (
        _profile_characteristic_speed(p, shape, cfg, bounds))

    theta = float(p[TH])
    sx = float(np.clip(p[SX], bounds["s"][0], bounds["s"][1]))
    sy = float(np.clip(p[SY], bounds["s"][0], bounds["s"][1]))
    rc = float(np.clip(p[RC], bounds["rc"][0], bounds["rc"][1]))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    deformation = np.array(
        [[sx * cos_t, -sy * sin_t],
         [sx * sin_t, sy * cos_t]], dtype=np.float64)
    inverse_deformation = np.linalg.inv(deformation)
    shape_matrix = np.asarray(
        [[[1.0, 0.0], [0.0, -1.0]],
         [[0.0, 1.0], [-1.0, 0.0]],
         [[0.0, -1.0], [1.0, 0.0]]], dtype=np.float64)[int(shape)]
    center_jacobian = (
        deformation @ shape_matrix @ inverse_deformation / (TWO_PI * rc * rc))
    try:
        displacement = -np.linalg.solve(center_jacobian, induced)
        base_displacement = inverse_deformation @ displacement
        shift_fraction = float(np.linalg.norm(base_displacement) / rc)
    except np.linalg.LinAlgError:
        shift_fraction = float("inf")
    return interaction_ratio, shift_fraction


def _profiles_have_low_interaction(profiles, cfg, bounds):
    """Keep analytic core labels aligned with centers in the summed field."""
    if len(profiles) < 2:
        return True
    settings = _geometry_settings(cfg)
    for profile_index, (_p, shape) in enumerate(profiles):
        if int(shape) not in (1, 2):
            continue
        ratio, shift = _profile_interaction_metrics(
            profile_index, profiles, cfg, bounds)
        if (
            ratio > settings["max_interaction_ratio"]
            or shift > settings["max_center_shift_fraction"]
        ):
            return False
    return True


def _profiles_are_spaced(candidate, existing, cfg):
    settings = _geometry_settings(cfg)
    candidate_radius = float(candidate[RC]) * max(
        float(candidate[SX]), float(candidate[SY]))
    for other, _shape in existing:
        other_radius = float(other[RC]) * max(
            float(other[SX]), float(other[SY]))
        distance = np.hypot(
            float(candidate[TX] - other[TX]),
            float(candidate[TY] - other[TY]))
        required = (
            settings["spacing_factor"] * (candidate_radius + other_radius)
            + settings["spacing_margin"])
        if distance <= required:
            return False
    return True


def generate_steady_fields(dist, cfg, bounds, device):
    """Sample `num_fields` parameter sets from the fitted distribution and render
    each as a steady velocity field on [-1,1]^2 (Eq. 10)."""
    gcfg = cfg.generate
    num_fields = int(gcfg.num_fields)   # NB: don't shadow the module-level N (param index)
    G = int(gcfg.grid)
    m_min, m_max = int(gcfg.min_vortices), int(gcfg.max_vortices)
    enforce_spacing = bool(gcfg.enforce_spacing)
    rng = np.random.default_rng(int(cfg.get("seed", 0)) + 1)

    shape_idx_list = sorted(dist["shape_index_probs"].keys())
    shape_probs = np.array([dist["shape_index_probs"][s] for s in shape_idx_list])
    pspec = dist["params"]
    empirical_params = np.asarray(
        dist.get("empirical_params", []), dtype=np.float32)
    empirical_shapes = np.asarray(
        dist.get("empirical_shapes", []), dtype=np.int64)
    use_empirical = (
        empirical_params.ndim == 2
        and empirical_params.shape[1:] == (7,)
        and len(empirical_params) == len(empirical_shapes)
        and len(empirical_params) > 0)
    empirical_jitter = float(gcfg.get("empirical_jitter", 0.02))
    if use_empirical:
        empirical_std = np.std(empirical_params, axis=0).astype(np.float32)
    else:
        empirical_std = None

    def sample_profile():
        if use_empirical:
            selected = int(rng.integers(0, len(empirical_params)))
            p = empirical_params[selected].copy()
            if empirical_jitter > 0.0:
                p += rng.normal(
                    0.0, empirical_jitter * np.maximum(empirical_std, 1e-6),
                    size=7).astype(np.float32)
            s = int(empirical_shapes[selected])
        else:
            p = np.empty(7, np.float32)
            for i, name in enumerate(PARAM_NAMES):
                p[i] = _sample_gaussian(pspec[name], rng)
            s = int(rng.choice(shape_idx_list, p=shape_probs))
        p[TH] = np.arctan2(np.sin(p[TH]), np.cos(p[TH]))
        return p, s

    gen_params = np.zeros((num_fields, m_max, 7), np.float32)
    gen_shapes = np.zeros((num_fields, m_max), np.int64)
    gen_mask = np.zeros((num_fields, m_max), np.float32)
    gen_m = np.zeros(num_fields, np.int64)

    max_sample_attempts = int(gcfg.get("max_sample_attempts", 500))
    max_field_attempts = int(gcfg.get("max_field_attempts", 200))
    for k in range(num_fields):
        m = int(rng.integers(m_min, m_max + 1))
        profiles = None
        for _field_try in range(max_field_attempts):
            candidate_profiles = []
            field_complete = True
            for _ in range(m):
                accepted = False
                for _try in range(max_sample_attempts):
                    p, s = sample_profile()
                    if not _numpy_profile_geometry_valid(p, cfg, bounds):
                        continue
                    if (
                        enforce_spacing
                        and candidate_profiles
                        and not _profiles_are_spaced(p, candidate_profiles, cfg)
                    ):
                        continue
                    proposed = candidate_profiles + [(p, s)]
                    if not _profiles_have_low_interaction(proposed, cfg, bounds):
                        continue
                    candidate_profiles.append((p, s))
                    accepted = True
                    break
                if not accepted:
                    field_complete = False
                    break
            if field_complete:
                profiles = candidate_profiles
                break
        if profiles is None:
            raise RuntimeError(
                f"Could not sample a label-safe {m}-profile field {k} after "
                f"{max_field_attempts} whole-field retries x "
                f"{max_sample_attempts} profile attempts. The valid fitted "
                "pool may be too small for the spacing/interaction limits.")
        for j, (p, s) in enumerate(profiles):
            gen_params[k, j] = p
            gen_shapes[k, j] = s
            gen_mask[k, j] = 1.0
        gen_m[k] = m

    # Render all fields with one batched forward pass (inactive profiles masked off).
    coords = make_local_grid(G, device)
    params_t = torch.from_numpy(gen_params).to(device)
    shapes_t = torch.from_numpy(gen_shapes).to(device)
    mask_t = torch.from_numpy(gen_mask).to(device)
    with torch.no_grad():
        fields = vatistas_mixture_velocity(coords, params_t, shapes_t, bounds, profile_mask=mask_t)
    logging.info(
        f"[generate] synthesized {num_fields} geometry-valid steady fields on "
        f"[-1,1]^2 at {G}x{G} (vortices per field in [{m_min},{m_max}], "
        f"sampling={'empirical_joint' if use_empirical else 'marginal_gaussian'}).")
    return {
        "params": gen_params, "shapes": gen_shapes, "mask": gen_mask,
        "num_vortices": gen_m, "fields": fields.cpu().numpy().astype(np.float32),
        "coords": coords.cpu().numpy().astype(np.float32),
    }




def save_outputs(out_dir, params, shape_idx, fit_loss, contrib, active_mask, meta, dist, gen):
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, "fitted_patch_params.npz"),
        params=params.cpu().numpy(), shapes=shape_idx.cpu().numpy(),
        fit_mse=fit_loss.cpu().numpy(), contribution=contrib.cpu().numpy(),
        active=active_mask.cpu().numpy(),
        meta_field=np.array([m["field"] for m in meta]),
        meta_t=np.array([m["t"] for m in meta]),
        meta_y0=np.array([m["y0"] for m in meta]),
        meta_x0=np.array([m["x0"] for m in meta]),
        meta_background_u=np.array([m.get("background_u", 0.0) for m in meta], np.float32),
        meta_background_v=np.array([m.get("background_v", 0.0) for m in meta], np.float32),
        meta_raw_rms=np.array([m.get("raw_rms", 1.0) for m in meta], np.float32),
        meta_residual_rms=np.array([m.get("residual_rms", 1.0) for m in meta], np.float32),
        meta_swirl_peak=np.array([m.get("swirl_peak", 0.0) for m in meta], np.float32),
    )
    with open(os.path.join(out_dir, "vatistas_param_distribution.json"), "w") as f:
        json.dump(dist, f, indent=2)
    np.savez_compressed(
        os.path.join(out_dir, "generated_steady_flows.npz"),
        params=gen["params"], shapes=gen["shapes"], mask=gen["mask"],
        num_vortices=gen["num_vortices"], fields=gen["fields"], coords=gen["coords"],
    )
    logging.info(f"[save] wrote fitted params, distribution JSON and generated fields to {out_dir}")


def save_plots(
        out_dir, coords, target, params, shape_idx, bounds, dist, gen,
        active_mask=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as ex:
        logging.warning(f"[plots] matplotlib unavailable ({ex}); skipping figures.")
        return

    # (1) parameter histograms
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    prm = params.reshape(-1, 7).cpu().numpy()
    if active_mask is not None:
        keep = active_mask.reshape(-1).cpu().numpy().astype(bool)
        prm = prm[keep]
    for i, name in enumerate(PARAM_NAMES):
        ax = axes.flat[i]
        ax.hist(prm[:, i], bins=40, color="steelblue", alpha=0.85)
        s = dist["params"][name]
        ax.set_title(f"{name}: mu={s['mean']:.3f}, sig={s['std']:.3f}")
    axes.flat[7].axis("off")
    axes.flat[7].text(0.05, 0.5, "shapes:\n" + "\n".join(f"{k}: {v:.2f}" for k, v in dist["shape_distribution"].items()),
                      fontsize=11, va="center")
    fig.suptitle("Fitted Vatistas parameter distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "param_histograms.png"), dpi=120)
    plt.close(fig)

    def quiver(ax, cd, vel, title):
        step = max(1, cd.shape[0] // 24)
        X, Y = cd[..., 0], cd[..., 1]
        speed = np.sqrt(vel[..., 0] ** 2 + vel[..., 1] ** 2)
        ax.imshow(speed, extent=[-1, 1, -1, 1], origin="lower", cmap="viridis", alpha=0.7)
        ax.quiver(X[::step, ::step], Y[::step, ::step],
                  vel[::step, ::step, 0], vel[::step, ::step, 1], color="white", scale=None)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])

    # (2) fit previews: target vs fitted for the first few patches
    with torch.no_grad():
        pred = vatistas_mixture_velocity(coords, params, shape_idx, bounds).cpu().numpy()
    cd = coords.cpu().numpy()
    tgt = target.cpu().numpy()
    npv = min(4, tgt.shape[0])
    fig, axes = plt.subplots(2, npv, figsize=(4 * npv, 8))
    axes = np.atleast_2d(axes)
    for j in range(npv):
        quiver(axes[0, j], cd, tgt[j], f"patch {j}: real")
        quiver(axes[1, j], cd, pred[j], f"patch {j}: fitted")
    fig.suptitle("Real patch (top) vs fitted Vatistas mixture (bottom)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fit_previews.png"), dpi=120)
    plt.close(fig)

    # (3) generated field previews
    gc, gf = gen["coords"], gen["fields"]
    ng = min(4, gf.shape[0])
    fig, axes = plt.subplots(1, ng, figsize=(4 * ng, 4))
    axes = np.atleast_1d(axes)
    for j in range(ng):
        quiver(axes[j], gc, gf[j], f"generated {j} (m={gen['num_vortices'][j]})")
    fig.suptitle("Synthesized steady fields sampled from the fitted distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "generated_previews.png"), dpi=120)
    plt.close(fig)

    # (4) geometry diagnostics for the generated pool.
    gp = gen["params"]
    gm = gen["mask"].astype(bool)
    physical_x = (gp[..., RC] * gp[..., SX])[gm]
    physical_y = (gp[..., RC] * gp[..., SY])[gm]
    centers = gp[..., [TX, TY]][gm]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(physical_x, bins=30, alpha=0.65, label="rc*sx")
    axes[0].hist(physical_y, bins=30, alpha=0.65, label="rc*sy")
    axes[0].set_title("Generated physical core semi-axes")
    axes[0].legend()
    axes[1].scatter(centers[:, 0], centers[:, 1], s=8, alpha=0.5)
    axes[1].set_xlim(-1, 1); axes[1].set_ylim(-1, 1)
    axes[1].set_aspect("equal")
    axes[1].set_title("Generated vortex centers")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "geometry_diagnostics.png"), dpi=120)
    plt.close(fig)
    logging.info(
        f"[plots] saved parameter, fit, generation, and geometry diagnostics "
        f"to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_bounds(cfg):
    b = cfg.model.bounds
    t = b["t"][0] if isinstance(b["t"], (list, tuple)) else float(b["t"])
    return {"rc": tuple(b["rc"]), "n": tuple(b["n"]), "s": tuple(b["s"]), "t": float(t)}


def main():
    parser = argparse.ArgumentParser(description="Fit Vatistas profile parameters to real flow patches.")
    parser.add_argument("--config", type=str, default="config/FittingVatistas.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_patches", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    cfg = EasyConfig()
    cfg.load(args.config, recursive=False)

    # Reuse the project's data-folder relocation (fills dataset.dat_dir + cache_dir).
    if cfg.dataset.get("dat_dir") is None:
        cfg.dataset["dat_dir"] = None
    relocate_flow2d_dataset_folder(cfg)

    if args.max_patches is not None:
        cfg.dataset["max_patches"] = args.max_patches
    if args.no_plots:
        cfg.output["save_plots"] = False

    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    dev_str = args.device or cfg.optim.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev_str)
    logging.info(f"[main] device={device}, seed={seed}")

    bounds = build_bounds(cfg)
    M = int(cfg.model.num_profiles)

    # 1) real-flow patches
    target, scales, meta = extract_patches(cfg, device)
    P = target.shape[0]
    coords = make_local_grid(int(cfg.dataset.patch_size), device)

    # 2) fit: simulated annealing -> gradient descent (optionally multi-restart)
    logging.info(f"[main] fitting {P} patches: simulated annealing -> gradient descent "
                 f"({int(cfg.optim.get('n_restarts', 1))} restart(s)) ...")
    params, shape_idx, fit_loss = fit_patches(coords, target, cfg, bounds, M, device)
    fit_quality_report(coords, target, params, shape_idx, bounds, tag="main")

    # 3) distribution over profiles that are energetic, well fitted, and
    # geometrically valid for clean segmentation labels.
    contrib = profile_contributions(coords, params, shape_idx, bounds)
    active_mask = valid_profile_mask(
        params, shape_idx, fit_loss, contrib, cfg, bounds)
    dist = fit_distribution(params, shape_idx, active_mask)

    # 4) generate new steady fields from the fitted distribution
    gen = generate_steady_fields(dist, cfg, bounds, device)

    out_dir = args.out_dir or cfg.output.get("out_dir") or os.path.join(cfg.get("cache_dir", "./outputs/"), "vatistas_fit")
    save_outputs(out_dir, params, shape_idx, fit_loss, contrib, active_mask, meta, dist, gen)
    if bool(cfg.output.get("save_plots", True)):
        save_plots(
            out_dir, coords, target, params, shape_idx, bounds, dist, gen,
            active_mask=active_mask)

    logging.info("[main] all done.")


if __name__ == "__main__":
    main()
