"""Generate VortexTransformer training data: synthesize steady Vatistas fields,
apply moving/rotating observers, and integrate feature-bearing pathlines.

Synthesized 3000 steady 2D fields:
1500 from real data, 1400 random samples, and 100 manually added Killing fields. Each
steady field was transformed via 20 randomly sampled rigid-body observers over T=[0,1],
yielding 60,000 unsteady fields"):

    STEADY   = 3000  = 1500 fitted (real-data patch fits)
                     + 1400 random (sampled from the fitted distribution)
                     + 100  manually-added Killing fields (rigid-motion hard negatives).
    UNSTEADY = 60000 = 3000 steady  x  20 rigid-body observers.

The default steady->unsteady transform is a NumPy port of
`CppProjects/src/transformation.cpp::Tobias_ObserverTransformation`.  The older
`killingABCtransformation` port remains selectable for compatibility.  For each
unsteady field the script also follows
`flowGenerator.cpp::generateMixUnsteadyFieldPathline`: cross-grid pathline
seeding, RK4 integration, per-point [x,y,t,IVD,distance,u,v] features, a seed
vortex label, a binary segmentation mask, observer parameters, and field meta.
In addition to NPZ shards, each pathline tensor is written as a
Cereal-compatible ``velocity_mix/sample_*_pathline.bin`` file.

NB: materializing all 60000 velocity grids and pathlines is tens of GB. By
default only a verification subset (400) of unsteady variants is written; pass
--full to stream the full set to compressed NPZ shards.

Run:
    python VatistasFlowDatasetGenerator.py --config config/VatistasDataset.yaml
    python VatistasFlowDatasetGenerator.py --full     # write all 60000 (sharded, ~39GB)
    python VatistasFlowDatasetGenerator.py --refit     # redo the vastistas parameters fitting first
"""

from __future__ import annotations

import os
import json
import math
import argparse
import logging
import platform
from ast import literal_eval
from typing import Any

import numpy as np
import torch
import yaml

try:
    import netCDF4 as nc
except ModuleNotFoundError:
    nc = None


class EasyConfig(dict):
    """Small nested config helper kept local so this script stays standalone."""

    def __init__(self, config_path: str | None = None):
        super().__init__()
        if config_path is not None:
            self.load(config_path)

    def __getattr__(self, key: str) -> Any:
        if key not in self:
            raise AttributeError(f"AttributeError:{key}")
        return self[key]

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        del self[key]

    @staticmethod
    def _convert(value: Any) -> Any:
        if isinstance(value, dict):
            cfg = EasyConfig()
            for sub_key, sub_value in value.items():
                cfg[sub_key] = EasyConfig._convert(sub_value)
            return cfg
        if isinstance(value, list):
            return [EasyConfig._convert(item) for item in value]
        return value

    def load(self, fpath: str, *, recursive: bool = False) -> None:
        if not os.path.exists(fpath):
            raise FileNotFoundError(fpath)
        fpaths = [fpath]
        if recursive:
            extension = os.path.splitext(fpath)[1]
            while os.path.dirname(fpath) != fpath:
                fpath = os.path.dirname(fpath)
                fpaths.append(os.path.join(fpath, "default" + extension))
        for path in reversed(fpaths):
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as handle:
                    payload = yaml.safe_load(handle) or {}
                self.update(payload)

    def update(self, other) -> None:
        if other is None:
            return
        if isinstance(other, dict):
            for key, value in other.items():
                converted = self._convert(value)
                if isinstance(converted, EasyConfig) and isinstance(self.get(key), EasyConfig):
                    self[key].update(converted)
                else:
                    self[key] = converted
            return
        if isinstance(other, (list, tuple)):
            index = 0
            while index < len(other):
                opt = other[index]
                if opt.startswith("--"):
                    opt = opt[2:]
                if "=" in opt:
                    key, value = opt.split("=", 1)
                    index += 1
                else:
                    key, value = opt, other[index + 1]
                    index += 2
                try:
                    value = literal_eval(value)
                except Exception:
                    pass
                current = self
                subkeys = key.split(".")
                for subkey in subkeys[:-1]:
                    if subkey not in current or not isinstance(current[subkey], EasyConfig):
                        current[subkey] = EasyConfig()
                    current = current[subkey]
                current[subkeys[-1]] = self._convert(value)
            return
        raise TypeError(f"Unsupported config update payload: {type(other)!r}")


class LocalUnsteadyVectorField2D:
    def __init__(self, field: np.ndarray):
        field = np.asarray(field, dtype=np.float32)
        if field.ndim != 4 or field.shape[-1] != 2:
            raise ValueError("Expected vector field data shaped [T, Y, X, 2].")
        self.field = field


def _create_analytical_flow(grid_size, time_steps, domain_min, domain_max, u_fn, v_fn):
    xdim, ydim = int(grid_size[0]), int(grid_size[1])
    t_values = np.linspace(domain_min[2], domain_max[2], int(time_steps), dtype=np.float32)
    x_values = np.linspace(domain_min[0], domain_max[0], xdim, dtype=np.float32)
    y_values = np.linspace(domain_min[1], domain_max[1], ydim, dtype=np.float32)
    xx, yy = np.meshgrid(x_values, y_values)
    field = np.zeros((int(time_steps), ydim, xdim, 2), dtype=np.float32)
    for i, t_val in enumerate(t_values):
        field[i, :, :, 0] = u_fn(xx, yy, float(t_val)).astype(np.float32, copy=False)
        field[i, :, :, 1] = v_fn(xx, yy, float(t_val)).astype(np.float32, copy=False)
    return LocalUnsteadyVectorField2D(field)


def _rotation_four_center(grid_size, timestep, domain_min=None, domain_max=None, scale=1.0):
    if domain_min is None:
        domain_min = [-2.0, -2.0, 0.0]
    if domain_max is None:
        domain_max = [2.0, 2.0, 2.0 * np.pi]
    al_t = 1.0
    flow_scale = 8.0

    def u_fn(x, y, t):
        ct = np.cos(al_t * t)
        st = np.sin(al_t * t)
        return np.exp(-(y * y + x * x)) * (
            al_t * y * np.exp(y * y + x * x)
            - 6.0 * flow_scale * ct * st * y * y * y
            + (12.0 * flow_scale * (ct * ct) - 6.0 * flow_scale) * x * y * y
            + (6.0 * flow_scale * ct * st * x * x + 6.0 * flow_scale * ct * st) * y
            + (3.0 * flow_scale - 6.0 * flow_scale * (ct * ct)) * x
        )

    def v_fn(x, y, t):
        ct = np.cos(al_t * t)
        st = np.sin(al_t * t)
        return -np.exp(-(y * y + x * x)) * (
            al_t * x * np.exp(y * y + x * x)
            - 6.0 * flow_scale * ct * st * x * y * y
            + ((12.0 * flow_scale * (ct * ct) - 6.0 * flow_scale) * x * x - 6.0 * flow_scale * (ct * ct) + 3.0 * flow_scale) * y
            + 6.0 * flow_scale * ct * st * x * x * x
            - 6.0 * flow_scale * ct * st * x
        )

    field = _create_analytical_flow(grid_size, timestep, domain_min, domain_max, u_fn, v_fn)
    if scale != 1.0:
        field.field *= float(scale)
    return field


def _beads_flow(grid_size, timestep, domain_min=None, domain_max=None, scale=1.0):
    if domain_min is None:
        domain_min = [-2.0, -2.0, 0.0]
    if domain_max is None:
        domain_max = [2.0, 2.0, 2.0 * np.pi]

    def u_fn(x, y, t):
        return -(y - np.sin(t) / 3.0) - (x - np.cos(t) / 3.0)

    def v_fn(x, y, t):
        return (x - np.cos(t) / 3.0) - (y - np.sin(t) / 3.0)

    field = _create_analytical_flow(grid_size, timestep, domain_min, domain_max, u_fn, v_fn)
    if scale != 1.0:
        field.field *= float(scale)
    return field


def _double_gyre_2d(grid_size, timestep, t_max=10.0, omega=0.2 * np.pi):
    eps = 0.25
    amp = 0.1

    def u_fn(x, y, t):
        f = eps * np.sin(omega * t) * x * x + (1.0 - 2.0 * eps * np.sin(omega * t)) * x
        return -np.sin(np.pi * f) * np.cos(np.pi * y) * (np.pi * amp)

    def v_fn(x, y, t):
        s = np.sin(omega * t)
        f = eps * s * x * x + (1.0 - 2.0 * eps * s) * x
        dfdx = (1.0 - 2.0 * eps * s) + 2.0 * eps * s * x
        return np.cos(np.pi * f) * np.sin(np.pi * y) * dfdx * (np.pi * amp)

    return _create_analytical_flow(grid_size, timestep, [0.0, 0.0, 0.0], [2.0, 1.0, t_max], u_fn, v_fn)


def _load_vector_field2d_netcdf(file_path: str):
    if nc is None:
        raise ModuleNotFoundError(
            "netCDF4 is required to load NetCDF flow fields. Install it or switch to synthetic fields only."
        )
    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)

    with nc.Dataset(file_path, "r") as dataset:
        dimension_names = list(dataset.dimensions.keys())
        variable_names = list(dataset.variables.keys())
        time_axis_name = next((name for name in dimension_names if str(name).lower() in {"time", "tdim", "t"}), None)
        if time_axis_name is None:
            raise ValueError(f"Could not find a time dimension in {file_path}.")
        x_axis_name = next((name for name in dimension_names if str(name).lower() in {"x", "xdim"}), None)
        y_axis_name = next((name for name in dimension_names if str(name).lower() in {"y", "ydim"}), None)
        if x_axis_name is None or y_axis_name is None:
            raise ValueError(f"Could not find x/y dimensions in {file_path}.")

        field_data = None
        for u_name, v_name in [("u", "v"), ("x", "y"), ("a", "b"), ("Component1", "Component2"), ("velocity_x", "velocity_y")]:
            if u_name in variable_names and v_name in variable_names:
                u = np.asarray(dataset.variables[u_name][:], dtype=np.float32)
                v = np.asarray(dataset.variables[v_name][:], dtype=np.float32)
                field_data = np.stack([u, v], axis=-1)
                break
        if field_data is None:
            raise ValueError(f"Could not find a supported velocity-component pair in {file_path}.")
        if field_data.ndim != 4 or field_data.shape[-1] != 2:
            raise ValueError(f"Expected NetCDF vector field shaped [T, Y, X, 2], got {field_data.shape}.")
    return LocalUnsteadyVectorField2D(field_data)


def relocate_flow2d_dataset_folder(config):
    script_root = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = config.dataset.get("dat_dir")
    if dataset_dir:
        config.dataset["dat_dir"] = os.path.abspath(os.path.expanduser(str(dataset_dir)))
    else:
        candidates = [
            os.environ.get("PYFLOWVIS_FLOW2D_DIR"),
            os.environ.get("FLOW2D_DATA_DIR"),
            os.environ.get("PYFLOWVIS_DATA_DIR"),
            os.path.join(script_root, "flowData2d"),
            os.path.join(script_root, "data", "flowData2d"),
            os.path.join(script_root, "assets", "flowData2d"),
            os.path.join(os.path.expanduser("~"), "flowData2d"),
        ]
        resolved = next((os.path.abspath(path) for path in candidates if path and os.path.exists(path)), None)
        config.dataset["dat_dir"] = resolved
        if resolved is None:
            logging.warning(
                "[config] dataset.dat_dir is not set and no default flow-data directory was found on %s. "
                "Synthetic fields still work, but NetCDF-backed fields require dataset.dat_dir or one of the "
                "env vars PYFLOWVIS_FLOW2D_DIR/FLOW2D_DATA_DIR/PYFLOWVIS_DATA_DIR.",
                platform.system(),
            )

    cache_dir = config.get("cache_dir")
    if cache_dir:
        config["cache_dir"] = os.path.abspath(os.path.expanduser(str(cache_dir)))
    else:
        config["cache_dir"] = os.path.join(script_root, "outputs")


def load_UnsteadyVectorFields_general(flowDataFolder, vector_field_names: list[str] | str):
    names = vector_field_names if isinstance(vector_field_names, list) else [vector_field_names]
    fields = []
    for name in names:
        name_lower = str(name).lower()
        try:
            if name_lower == "beads2d":
                fields.append(_beads_flow([128, 128], 32))
            elif name_lower == "doublegyre2d":
                fields.append(_double_gyre_2d([256, 128], 64))
            elif name_lower == "rfc2d":
                fields.append(_rotation_four_center([128, 128], 32))
            elif "gerrisflowsolverdata" in name_lower:
                logging.warning(
                    "[load_UnsteadyVectorFields_general] standalone generator does not inline the Amira folder loader; field '%s' is skipped.",
                    name,
                )
                fields.append(None)
            else:
                if not flowDataFolder:
                    raise RuntimeError(
                        f"dataset.dat_dir is required to load NetCDF field '{name}'. "
                        "Set it in the YAML or via PYFLOWVIS_FLOW2D_DIR/FLOW2D_DATA_DIR/PYFLOWVIS_DATA_DIR."
                    )
                vectorfield_datapath = os.path.join(flowDataFolder, f"{name}.nc")
                fields.append(_load_vector_field2d_netcdf(vectorfield_datapath))
        except Exception as exc:
            logging.warning(f"[load_UnsteadyVectorFields_general] load {name} failed: {exc}. Skip this field.")
            fields.append(None)
    return fields


PARAM_NAMES = ["theta", "sx", "sy", "rc", "n", "tx", "ty"]
TH, SX, SY, RC, N, TX, TY = range(7)
S_MATRICES = torch.tensor(
    [
        [[1.0, 0.0], [0.0, -1.0]],
        [[0.0, 1.0], [-1.0, 0.0]],
        [[0.0, -1.0], [1.0, 0.0]],
    ],
    dtype=torch.float32,
)
SHAPE_LABELS = {0: "S1_saddle", 1: "S2_center_cw", 2: "S3_center_ccw"}
TWO_PI = 2.0 * np.pi


def vatistas_mixture_velocity(coords, params, shape_idx, bounds, profile_mask=None):
    P, M, _ = params.shape
    H, W = coords.shape[0], coords.shape[1]
    eps = 1e-9
    theta = params[..., TH]
    rc = params[..., RC].clamp(bounds["rc"][0], bounds["rc"][1])
    n = params[..., N].clamp(bounds["n"][0], bounds["n"][1])
    sx = params[..., SX].clamp(bounds["s"][0], bounds["s"][1])
    sy = params[..., SY].clamp(bounds["s"][0], bounds["s"][1])
    tx = params[..., TX]
    ty = params[..., TY]
    cos, sin = torch.cos(theta), torch.sin(theta)
    inv_det = 1.0 / (sx * sy)
    di00 = inv_det * sy * cos
    di01 = inv_det * sy * sin
    di10 = inv_det * (-sx * sin)
    di11 = inv_det * sx * cos
    xp = coords[..., 0].view(1, 1, H, W)
    yp = coords[..., 1].view(1, 1, H, W)

    def e(a):
        return a.view(P, M, 1, 1)

    dx = xp - e(tx)
    dy = yp - e(ty)
    bx = e(di00) * dx + e(di01) * dy
    by = e(di10) * dx + e(di11) * dy
    r = torch.sqrt(bx * bx + by * by + eps)
    log_ratio = torch.log(r) - torch.log(e(rc))
    p2n = ((2.0 * e(n)) * log_ratio).clamp(max=30.0)
    log_bracket = torch.log1p(torch.exp(p2n))
    g = torch.exp(-(1.0 / e(n)) * log_bracket) / (TWO_PI * e(rc) * e(rc))
    s_mat = S_MATRICES.to(params.device)[shape_idx]
    sx_base = s_mat[..., 0, 0].unsqueeze(-1).unsqueeze(-1) * bx + s_mat[..., 0, 1].unsqueeze(-1).unsqueeze(-1) * by
    sy_base = s_mat[..., 1, 0].unsqueeze(-1).unsqueeze(-1) * bx + s_mat[..., 1, 1].unsqueeze(-1).unsqueeze(-1) * by
    vbx = g * sx_base
    vby = g * sy_base
    vpx = e(sx * cos) * vbx + e(-sy * sin) * vby
    vpy = e(sx * sin) * vbx + e(sy * cos) * vby
    if profile_mask is not None:
        vpx = vpx * e(profile_mask)
        vpy = vpy * e(profile_mask)
    vx = vpx.sum(dim=1)
    vy = vpy.sum(dim=1)
    return torch.stack([vx, vy], dim=-1)


def per_patch_mse(pred, target):
    return ((pred - target) ** 2).reshape(pred.shape[0], -1).mean(dim=1)


def _clamp_time_range(t_start, t_end, n_steps):
    a = 0 if t_start is None else int(t_start)
    b = n_steps if t_end is None else int(t_end)
    a = max(0, min(a, n_steps))
    b = max(0, min(b, n_steps))
    if b <= a:
        a, b = 0, n_steps
    return a, b


def extract_patches(cfg, device):
    dcfg = cfg.dataset
    ps = int(dcfg.patch_size)
    stride = int(dcfg.patch_stride)
    min_rms = float(dcfg.min_patch_rms)
    normalize = bool(dcfg.normalize_patch)
    field_specs = list(dcfg.fields)
    names = [str(f["name"]) for f in field_specs]
    logging.info(f"[patches] loading real flows: {names}")
    fields = load_UnsteadyVectorFields_general(dcfg.dat_dir, names)

    patches, scales, meta = [], [], []
    for spec, vf in zip(field_specs, fields):
        if vf is None:
            logging.warning(f"[patches] field '{spec['name']}' failed to load; skip.")
            continue
        field = vf.field
        field = field.cpu().numpy() if isinstance(field, torch.Tensor) else np.asarray(field)
        field = field.astype(np.float32)
        t_steps, ydim, xdim = field.shape[0], field.shape[1], field.shape[2]
        a, b = _clamp_time_range(spec.get("t_start"), spec.get("t_end"), t_steps)
        t_stride = max(1, int(spec.get("t_stride", 1)))
        t_indices = list(range(a, b, t_stride))
        if ydim < ps or xdim < ps:
            logging.warning(f"[patches] field '{spec['name']}' ({ydim}x{xdim}) smaller than patch {ps}; skip.")
            continue

        row_starts = list(range(0, ydim - ps + 1, stride))
        col_starts = list(range(0, xdim - ps + 1, stride))
        n_before = len(patches)
        for t in t_indices:
            slice_t = field[t]
            for y0 in row_starts:
                for x0 in col_starts:
                    patch = slice_t[y0:y0 + ps, x0:x0 + ps, :]
                    rms = float(np.sqrt(np.mean(patch[..., 0] ** 2 + patch[..., 1] ** 2)))
                    if rms < min_rms:
                        continue
                    scale = rms if normalize else 1.0
                    patches.append(patch / (scale + 1e-12) if normalize else patch.copy())
                    scales.append(scale)
                    meta.append({"field": str(spec["name"]), "t": int(t), "y0": int(y0), "x0": int(x0)})
        logging.info(f"[patches] '{spec['name']}': {len(t_indices)} slices -> {len(patches) - n_before} patches (time steps {a}:{b}:{t_stride}, grid {ydim}x{xdim}).")

    if not patches:
        raise RuntimeError("No patches extracted. Check dataset paths / field names / min_patch_rms.")

    targets = np.stack(patches, axis=0).astype(np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    max_patches = int(dcfg.max_patches)
    if targets.shape[0] > max_patches:
        idx = np.random.default_rng(int(cfg.get("seed", 0))).permutation(targets.shape[0])[:max_patches]
        idx.sort()
        targets, scales, meta = targets[idx], scales[idx], [meta[i] for i in idx]
        logging.info(f"[patches] subsampled to max_patches={max_patches}.")

    logging.info(f"[patches] total {targets.shape[0]} patches of size {ps}x{ps}.")
    return torch.from_numpy(targets).to(device), torch.from_numpy(scales).to(device), meta


def make_local_grid(ps, device):
    lin = torch.linspace(-1.0, 1.0, ps, device=device)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    return torch.stack([xx, yy], dim=-1)


def _project(params, shape_idx, bounds, rng=None):
    with torch.no_grad():
        params[..., RC].clamp_(bounds["rc"][0], bounds["rc"][1])
        params[..., N].clamp_(bounds["n"][0], bounds["n"][1])
        params[..., SX].clamp_(bounds["s"][0], bounds["s"][1])
        params[..., SY].clamp_(bounds["s"][0], bounds["s"][1])
        params[..., TX].clamp_(-bounds["t"], bounds["t"])
        params[..., TY].clamp_(-bounds["t"], bounds["t"])
    return params


def datadriven_seeds(target, coords, cfg, n_profiles):
    p_count, height, width, _ = target.shape
    u, v = target[..., 0], target[..., 1]
    dvdx = torch.gradient(v, spacing=2.0 / (width - 1), dim=2)[0]
    dudy = torch.gradient(u, spacing=2.0 / (height - 1), dim=1)[0]
    omega = dvdx - dudy
    oabs = omega.abs().clone()
    cx, cy = coords[..., 0], coords[..., 1]
    cand = [int(c) for c in cfg.model.shape_candidates]
    has_cw, has_ccw = (1 in cand), (2 in cand)
    dev = target.device
    ii, jj = torch.meshgrid(torch.arange(height, device=dev), torch.arange(width, device=dev), indexing="ij")
    rad = max(2, int(0.25 * min(height, width)))
    tx0 = torch.zeros(p_count, n_profiles, device=dev)
    ty0 = torch.zeros(p_count, n_profiles, device=dev)
    shape0 = torch.zeros(p_count, n_profiles, dtype=torch.long, device=dev)
    ar = torch.arange(p_count, device=dev)
    for m in range(n_profiles):
        idx = oabs.reshape(p_count, -1).argmax(dim=1)
        pi, pj = idx // width, idx % width
        tx0[:, m], ty0[:, m] = cx[pi, pj], cy[pi, pj]
        w_at = omega[ar, pi, pj]
        s = torch.where(w_at >= 0, torch.full_like(idx, 2), torch.full_like(idx, 1))
        if not has_ccw:
            s = torch.where(s == 2, torch.full_like(s, 1), s)
        if not has_cw:
            s = torch.where(s == 1, torch.full_like(s, 2), s)
        shape0[:, m] = s
        d2 = (ii.view(1, height, width) - pi.view(p_count, 1, 1)) ** 2 + (jj.view(1, height, width) - pj.view(p_count, 1, 1)) ** 2
        oabs = torch.where(d2 <= rad * rad, torch.zeros_like(oabs), oabs)
    return tx0, ty0, shape0


def init_params(p_count, n_profiles, cfg, bounds, device, restart=0, seeds=None):
    generator = torch.Generator(device="cpu").manual_seed(int(cfg.get("seed", 0)) + 9973 * int(restart))

    def u(lo, hi, shape):
        return (lo + (hi - lo) * torch.rand(shape, generator=generator)).to(device)

    params = torch.empty(p_count, n_profiles, 7, device=device)
    params[..., TH] = u(-np.pi, np.pi, (p_count, n_profiles))
    params[..., SX] = u(0.5, 1.5, (p_count, n_profiles))
    params[..., SY] = u(0.5, 1.5, (p_count, n_profiles))
    params[..., RC] = u(0.2, 0.6, (p_count, n_profiles))
    params[..., N] = u(1.0, 3.0, (p_count, n_profiles))
    candidates = torch.tensor([int(c) for c in cfg.model.shape_candidates], dtype=torch.long)
    if seeds is not None:
        tx0, ty0, shape0 = seeds
        jit = 0.0 if restart == 0 else min(0.5, 0.12 * restart)
        params[..., TX] = tx0 + jit * torch.randn((p_count, n_profiles), generator=generator).to(device)
        params[..., TY] = ty0 + jit * torch.randn((p_count, n_profiles), generator=generator).to(device)
        if restart == 0:
            shape_idx = shape0.clone()
        else:
            flip = torch.rand((p_count, n_profiles), generator=generator).to(device) < 0.3
            rnd = candidates[torch.randint(0, len(candidates), (p_count, n_profiles), generator=generator)].to(device)
            shape_idx = torch.where(flip, rnd, shape0)
    else:
        params[..., TX] = u(-0.5, 0.5, (p_count, n_profiles))
        params[..., TY] = u(-0.5, 0.5, (p_count, n_profiles))
        pick = torch.randint(0, len(candidates), (p_count, n_profiles), generator=generator)
        shape_idx = candidates[pick].to(device)
    return _project(params, shape_idx, bounds), shape_idx


def simulated_annealing(coords, target, params, shape_idx, cfg, bounds):
    ocfg = cfg.optim
    iters = int(ocfg.sa_iters)
    t0, t_end = float(ocfg.sa_T0), float(ocfg.sa_Tend)
    flip_p = float(ocfg.sa_shape_flip_p)
    cooling = (t_end / t0) ** (1.0 / max(1, iters))
    p_count, n_profiles, _ = params.shape
    device = params.device
    candidates = torch.tensor([int(c) for c in cfg.model.shape_candidates], dtype=torch.long, device=device)
    sigma = torch.tensor([0.5, 0.3, 0.3, 0.15, 0.5, 0.2, 0.2], device=device)

    with torch.no_grad():
        cur = params.clone()
        cur_shape = shape_idx.clone()
        cur_loss = per_patch_mse(vatistas_mixture_velocity(coords, cur, cur_shape, bounds), target)
        best, best_shape, best_loss = cur.clone(), cur_shape.clone(), cur_loss.clone()
        temp = t0
        for it in range(iters):
            frac = temp / t0
            prop = _project(cur + torch.randn_like(cur) * (sigma.view(1, 1, 7) * frac), cur_shape, bounds)
            prop_shape = cur_shape
            if flip_p > 0.0:
                flip = torch.rand(p_count, n_profiles, device=device) < flip_p
                rnd_shape = candidates[torch.randint(0, len(candidates), (p_count, n_profiles), device=device)]
                prop_shape = torch.where(flip, rnd_shape, cur_shape)
            prop_loss = per_patch_mse(vatistas_mixture_velocity(coords, prop, prop_shape, bounds), target)
            delta = prop_loss - cur_loss
            accept = (delta < 0) | (torch.rand(p_count, device=device) < torch.exp(-delta / max(temp, 1e-8)))
            cur = torch.where(accept.view(p_count, 1, 1), prop, cur)
            cur_shape = torch.where(accept.view(p_count, 1), prop_shape, cur_shape)
            cur_loss = torch.where(accept, prop_loss, cur_loss)
            improved = cur_loss < best_loss
            best = torch.where(improved.view(p_count, 1, 1), cur, best)
            best_shape = torch.where(improved.view(p_count, 1), cur_shape, best_shape)
            best_loss = torch.where(improved, cur_loss, best_loss)
            temp *= cooling
            if (it + 1) % 50 == 0 or it == iters - 1:
                logging.info(f"[SA] iter {it+1}/{iters}  T={temp:.4g}  mean best mse={best_loss.mean().item():.6f}")
    return best, best_shape, best_loss


def gradient_descent(coords, target, params, shape_idx, cfg, bounds):
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
        loss = per_patch_mse(pred, target).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p], max_norm=10.0)
        opt.step()
        _project(p, shape_idx, bounds)
        if (it + 1) % 50 == 0 or it == iters - 1:
            logging.info(f"[GD] iter {it+1}/{iters}  lr={cur_lr:.4g}  mean mse={loss.item():.6f}")
    with torch.no_grad():
        final = p.detach()
        final_loss = per_patch_mse(vatistas_mixture_velocity(coords, final, shape_idx, bounds), target)
    return final, final_loss


def lbfgs_polish(coords, target, params, shape_idx, cfg, bounds):
    iters = int(cfg.optim.get("lbfgs_iters", 0))
    if iters <= 0:
        with torch.no_grad():
            return params, per_patch_mse(vatistas_mixture_velocity(coords, params, shape_idx, bounds), target)
    p = params.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([p], max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        loss = per_patch_mse(vatistas_mixture_velocity(coords, p, shape_idx, bounds), target).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        _project(p, shape_idx, bounds)
        final = p.detach()
        final_loss = per_patch_mse(vatistas_mixture_velocity(coords, final, shape_idx, bounds), target)
    orig_loss = per_patch_mse(vatistas_mixture_velocity(coords, params, shape_idx, bounds), target)
    bad = (final_loss > orig_loss) | ~torch.isfinite(final_loss)
    if bad.any():
        final = torch.where(bad.view(-1, 1, 1), params, final)
        final_loss = per_patch_mse(vatistas_mixture_velocity(coords, final, shape_idx, bounds), target)
    return final, final_loss


def fit_patches(coords, target, cfg, bounds, n_profiles, device):
    p_count = target.shape[0]
    n_restarts = int(cfg.optim.get("n_restarts", 1))
    seeds = None
    if str(cfg.optim.get("init_mode", "random")) == "datadriven":
        seeds = datadriven_seeds(target, coords, cfg, n_profiles)
    best_params = best_shape = best_loss = None
    for r in range(n_restarts):
        params, shape_idx = init_params(p_count, n_profiles, cfg, bounds, device, restart=r, seeds=seeds)
        params, shape_idx, _ = simulated_annealing(coords, target, params, shape_idx, cfg, bounds)
        params, loss = gradient_descent(coords, target, params, shape_idx, cfg, bounds)
        if int(cfg.optim.get("lbfgs_iters", 0)) > 0:
            params, loss = lbfgs_polish(coords, target, params, shape_idx, cfg, bounds)
        if best_loss is None:
            best_params, best_shape, best_loss = params, shape_idx, loss
        else:
            take = loss < best_loss
            best_params = torch.where(take.view(p_count, 1, 1), params, best_params)
            best_shape = torch.where(take.view(p_count, 1), shape_idx, best_shape)
            best_loss = torch.where(take, loss, best_loss)
        if n_restarts > 1:
            logging.info(f"[fit] restart {r+1}/{n_restarts}  this-run mean mse={loss.mean().item():.6f}  best-so-far mean mse={best_loss.mean().item():.6f}")
    return best_params, best_shape, best_loss


def fit_quality_report(coords, target, params, shape_idx, bounds, tag="fit"):
    with torch.no_grad():
        pred = vatistas_mixture_velocity(coords, params, shape_idx, bounds)
        p_count = target.shape[0]
        se = (pred - target) ** 2
        mse_pp = se.reshape(p_count, -1).mean(dim=1)
        tgt = target.reshape(p_count, -1)
        rng = (tgt.max(dim=1).values - tgt.min(dim=1).values).clamp_min(1e-8)
        psnr_pp = 20.0 * torch.log10(rng) - 10.0 * torch.log10(mse_pp.clamp_min(1e-12))
        global_mse = se.mean()
        grng = (target.max() - target.min()).clamp_min(1e-8)
        global_psnr = 20.0 * torch.log10(grng) - 10.0 * torch.log10(global_mse.clamp_min(1e-12))
    metrics = {
        "mse_mean": float(mse_pp.mean()), "mse_median": float(mse_pp.median()),
        "psnr_mean": float(psnr_pp.mean()), "psnr_median": float(psnr_pp.median()),
        "global_mse": float(global_mse), "global_psnr": float(global_psnr),
    }
    logging.info(f"[{tag}] velocity MSE  mean={metrics['mse_mean']:.6f}  median={metrics['mse_median']:.6f}  global={metrics['global_mse']:.6f}")
    logging.info(f"[{tag}] PSNR (dB)     mean={metrics['psnr_mean']:.2f}  median={metrics['psnr_median']:.2f}  global={metrics['global_psnr']:.2f}")
    return metrics


def profile_contributions(coords, params, shape_idx, bounds):
    p_count, n_profiles, _ = params.shape
    with torch.no_grad():
        rms = torch.zeros(p_count, n_profiles, device=params.device)
        for m in range(n_profiles):
            mask = torch.zeros(p_count, n_profiles, device=params.device)
            mask[:, m] = 1.0
            v = vatistas_mixture_velocity(coords, params, shape_idx, bounds, profile_mask=mask)
            rms[:, m] = torch.sqrt((v ** 2).reshape(p_count, -1).mean(dim=1) + 1e-12)
        total = rms.sum(dim=1, keepdim=True) + 1e-12
        return rms / total


def fit_distribution(params, shape_idx, active_mask):
    prm = params.reshape(-1, 7).cpu().numpy()
    shp = shape_idx.reshape(-1).cpu().numpy()
    keep = active_mask.reshape(-1).cpu().numpy().astype(bool)
    prm, shp = prm[keep], shp[keep]
    theta = np.arctan2(np.sin(prm[:, TH]), np.cos(prm[:, TH]))
    prm = prm.copy()
    prm[:, TH] = theta
    dist = {"num_profiles": int(prm.shape[0]), "params": {}}
    for i, name in enumerate(PARAM_NAMES):
        col = prm[:, i]
        if name == "theta":
            c_mean = float(np.arctan2(np.sin(col).mean(), np.cos(col).mean()))
            dist["params"][name] = {"mean": c_mean, "std": float(col.std()), "min": float(col.min()), "max": float(col.max())}
        else:
            dist["params"][name] = {"mean": float(col.mean()), "std": float(col.std()), "min": float(col.min()), "max": float(col.max())}
    shapes, counts = np.unique(shp, return_counts=True)
    probs = counts / counts.sum()
    dist["shape_distribution"] = {SHAPE_LABELS[int(s)]: float(pr) for s, pr in zip(shapes, probs)}
    dist["shape_index_probs"] = {int(s): float(pr) for s, pr in zip(shapes, probs)}
    return dist


def _sample_gaussian(spec, rng):
    return rng.normal(spec["mean"], max(spec["std"], 1e-6))


def generate_steady_fields(dist, cfg, bounds, device):
    gcfg = cfg.generate
    num_fields = int(gcfg.num_fields)
    grid_size = int(gcfg.grid)
    m_min, m_max = int(gcfg.min_vortices), int(gcfg.max_vortices)
    enforce_spacing = bool(gcfg.enforce_spacing)
    rng = np.random.default_rng(int(cfg.get("seed", 0)) + 1)
    shape_idx_list = sorted(dist["shape_index_probs"].keys())
    shape_probs = np.array([dist["shape_index_probs"][s] for s in shape_idx_list])
    pspec = dist["params"]

    def sample_profile():
        p = np.empty(7, np.float32)
        for i, name in enumerate(PARAM_NAMES):
            p[i] = _sample_gaussian(pspec[name], rng)
        p[RC] = np.clip(p[RC], bounds["rc"][0], bounds["rc"][1])
        p[N] = np.clip(p[N], bounds["n"][0], bounds["n"][1])
        p[SX] = np.clip(p[SX], bounds["s"][0], bounds["s"][1])
        p[SY] = np.clip(p[SY], bounds["s"][0], bounds["s"][1])
        p[TX] = np.clip(p[TX], -bounds["t"], bounds["t"])
        p[TY] = np.clip(p[TY], -bounds["t"], bounds["t"])
        p[TH] = np.arctan2(np.sin(p[TH]), np.cos(p[TH]))
        s = int(rng.choice(shape_idx_list, p=shape_probs))
        return p, s

    gen_params = np.zeros((num_fields, m_max, 7), np.float32)
    gen_shapes = np.zeros((num_fields, m_max), np.int64)
    gen_mask = np.zeros((num_fields, m_max), np.float32)
    gen_m = np.zeros(num_fields, np.int64)
    for k in range(num_fields):
        m = int(rng.integers(m_min, m_max + 1))
        profiles = []
        for _ in range(m):
            for _try in range(50):
                p, s = sample_profile()
                if not enforce_spacing or not profiles:
                    break
                ok = all(np.hypot(p[TX] - q[TX], p[TY] - q[TY]) > (p[RC] + q[RC]) for q, _ in profiles)
                if ok:
                    break
            profiles.append((p, s))
        for j, (p, s) in enumerate(profiles):
            gen_params[k, j] = p
            gen_shapes[k, j] = s
            gen_mask[k, j] = 1.0
        gen_m[k] = m
    coords = make_local_grid(grid_size, device)
    params_t = torch.from_numpy(gen_params).to(device)
    shapes_t = torch.from_numpy(gen_shapes).to(device)
    mask_t = torch.from_numpy(gen_mask).to(device)
    with torch.no_grad():
        fields = vatistas_mixture_velocity(coords, params_t, shapes_t, bounds, profile_mask=mask_t)
    logging.info(f"[generate] synthesized {num_fields} steady fields on [-1,1]^2 at {grid_size}x{grid_size} (vortices per field in [{m_min},{m_max}]).")
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
    )
    with open(os.path.join(out_dir, "vatistas_param_distribution.json"), "w", encoding="utf-8") as handle:
        json.dump(dist, handle, indent=2)
    np.savez_compressed(
        os.path.join(out_dir, "generated_steady_flows.npz"),
        params=gen["params"], shapes=gen["shapes"], mask=gen["mask"],
        num_vortices=gen["num_vortices"], fields=gen["fields"], coords=gen["coords"],
    )
    logging.info(f"[save] wrote fitted params, distribution JSON and generated fields to {out_dir}")


def build_bounds(cfg):
    b = cfg.model.bounds
    t = b["t"][0] if isinstance(b["t"], (list, tuple)) else float(b["t"])
    return {"rc": tuple(b["rc"]), "n": tuple(b["n"]), "s": tuple(b["s"]), "t": float(t)}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

CENTER_SHAPES = (1, 2)  # S2 (cw) / S3 (ccw); saddles carry no vortex label
SRC_RANDOM, SRC_FITTED, SRC_KILLING = 0, 1, 2


# ===========================================================================
# 0. Fitting + distribution  (load the cached best fit, or refit on request)
# ===========================================================================
def load_fit_outputs(fit_dir):
    dist_path = os.path.join(fit_dir, "vatistas_param_distribution.json")
    npz_path = os.path.join(fit_dir, "fitted_patch_params.npz")
    if not (os.path.isfile(dist_path) and os.path.isfile(npz_path)):
        raise FileNotFoundError(
            f"No cached fit in '{fit_dir}'. Run `python FittingVatistasParam.py` first, "
            f"or pass --refit to fit here.")
    with open(dist_path) as f:
        dist = json.load(f)
    d = np.load(npz_path, allow_pickle=True)
    fitted = {
        "params": d["params"].astype(np.float32),
        "shapes": d["shapes"].astype(np.int64),
        "mask": d["active"].astype(np.float32),
        "fit_mse": d["fit_mse"].astype(np.float32),
    }
    logging.info(f"[fit] loaded cached best fit: {fitted['params'].shape[0]} patches "
                 f"(mean MSE {float(fitted['fit_mse'].mean()):.4f}); distribution over "
                 f"{dist['num_profiles']} profiles.")
    return dist, fitted


def refit(cfg_fit_path, device):
    cfg = EasyConfig()
    cfg.load(cfg_fit_path, recursive=False)
    if cfg.dataset.get("dat_dir") is None:
        cfg.dataset["dat_dir"] = None
    relocate_flow2d_dataset_folder(cfg)
    bounds = build_bounds(cfg)
    M = int(cfg.model.num_profiles)
    target, _scales, meta = extract_patches(cfg, device)
    coords = make_local_grid(int(cfg.dataset.patch_size), device)
    params, shape_idx, fit_loss = fit_patches(coords, target, cfg, bounds, M, device)
    fit_quality_report(coords, target, params, shape_idx, bounds, tag="refit")
    contrib = profile_contributions(coords, params, shape_idx, bounds)
    active = contrib >= float(cfg.generate.min_contribution)
    dist = fit_distribution(params, shape_idx, active)
    fit_dir = cfg.output.get("out_dir") or os.path.join(cfg.get("cache_dir", "./outputs/"), "vatistas_fit")
    gen = generate_steady_fields(dist, cfg, bounds, device)
    save_outputs(fit_dir, params, shape_idx, fit_loss, contrib, active, meta, dist, gen)
    fitted = {"params": params.cpu().numpy().astype(np.float32),
              "shapes": shape_idx.cpu().numpy().astype(np.int64),
              "mask": active.cpu().numpy().astype(np.float32),
              "fit_mse": fit_loss.cpu().numpy().astype(np.float32)}
    return dist, fitted


# ===========================================================================
# 1. Steady field pool:  random (distribution) + fitted (real patches) + Killing
# ===========================================================================
def _subsample_fitted(fitted, cfg):
    f_params, f_shapes, f_mask = fitted["params"], fitted["shapes"], fitted["mask"]
    fcfg = cfg.get("fitted", {}) or {}
    want = fcfg.get("num_fields", None)
    want = None if want in (None, "null") else int(want)
    if want is not None and 0 < want < f_params.shape[0]:
        if bool(fcfg.get("prefer_low_mse", True)):
            order = np.argsort(fitted["fit_mse"])[:want]
        else:
            order = np.random.default_rng(int(cfg.get("seed", 0)) + 5).permutation(f_params.shape[0])[:want]
        order = np.sort(order)
        f_params, f_shapes, f_mask = f_params[order], f_shapes[order], f_mask[order]
    return f_params, f_shapes, f_mask


def make_killing_steady(cfg):
    """Sample the 100 manually-added Killing (rigid-motion) steady fields.
    v(x) = (a, b) + c*(-y, x). These are observer-producible -> NOT objective vortices,
    so they act as hard negatives with an EMPTY segmentation label."""
    kcfg = cfg.get("killing_fields", {}) or {}
    n = int(kcfg.get("num", 100))
    ar = kcfg.get("abc_range", [0.8, 0.8, 1.0])
    rng = np.random.default_rng(int(cfg.get("seed", 0)) + 11)
    abc = np.stack([rng.uniform(-ar[0], ar[0], n),
                    rng.uniform(-ar[1], ar[1], n),
                    rng.uniform(-ar[2], ar[2], n)], axis=1).astype(np.float32)  # [n,3]
    return abc


def build_steady_specs(dist, fitted, cfg, bounds, device):
    """Assemble the 3000-field steady pool as analytic specs. Returns a list of dicts;
    each is either a Vatistas mixture ({type:'vatistas', params[M,7], shapes[M], mask[M]})
    or a Killing field ({type:'killing', abc:[a,b,c]}), all tagged with 'source'."""
    specs = []

    # (a) random-from-distribution
    gen = generate_steady_fields(dist, cfg, bounds, device)
    for i in range(gen["params"].shape[0]):
        specs.append({"type": "vatistas", "params": gen["params"][i], "shapes": gen["shapes"][i],
                      "mask": gen["mask"][i], "source": SRC_RANDOM})

    # (b) fitted real-flow patches (rendered from their own fitted parameters)
    f_params, f_shapes, f_mask = _subsample_fitted(fitted, cfg)
    for i in range(f_params.shape[0]):
        specs.append({"type": "vatistas", "params": f_params[i], "shapes": f_shapes[i],
                      "mask": f_mask[i], "source": SRC_FITTED})

    # (c) manually-added Killing fields (hard negatives, empty label)
    kabc = make_killing_steady(cfg)
    for i in range(kabc.shape[0]):
        specs.append({"type": "killing", "abc": kabc[i], "source": SRC_KILLING})

    counts = {SRC_RANDOM: gen["params"].shape[0], SRC_FITTED: f_params.shape[0], SRC_KILLING: kabc.shape[0]}
    logging.info(f"[steady] pool = {len(specs)} fields  "
                 f"(random {counts[SRC_RANDOM]} + fitted {counts[SRC_FITTED]} + killing {counts[SRC_KILLING]}).")
    return specs, counts


# ---- per-field steady evaluator + label (analytic, works at arbitrary points) ----
def make_steady_eval(spec, bounds, device):
    """Return (vel_eval, label_eval): coords[H,W,2] -> velocity[H,W,2] / signed-distance[H,W]."""
    if spec["type"] == "killing":
        a, b, c = [float(x) for x in spec["abc"]]

        def vel(coords_np):
            x, y = coords_np[..., 0], coords_np[..., 1]
            return np.stack([a - c * y, b + c * x], axis=-1).astype(np.float32)

        def lab(coords_np):                      # rigid motion -> no objective vortex core
            return np.full(coords_np.shape[:2], -1.0, np.float32)
        return vel, lab

    params, shapes, mask = spec["params"], spec["shapes"], spec["mask"]
    pt = torch.from_numpy(params[None]).to(device)
    st = torch.from_numpy(shapes[None]).to(device)
    mt = torch.from_numpy(mask[None]).to(device)

    def vel(coords_np):
        c = torch.from_numpy(np.ascontiguousarray(coords_np, np.float32)).to(device)
        with torch.no_grad():
            v = vatistas_mixture_velocity(c, pt, st, bounds, profile_mask=mt)[0]
        return v.cpu().numpy().astype(np.float32)

    def lab(coords_np):
        return vatistas_signed_distance(coords_np.astype(np.float32), params, shapes, mask, bounds)
    return vel, lab


def vatistas_signed_distance(coords_np, params, shapes, mask, bounds):
    """[H,W,2] -> [H,W]: max over active CENTER profiles of (rc - ||D^-1 (x - T)||)
    (Eq. 5/8). Mirrors the forward model's pull-back exactly."""
    H, W, _ = coords_np.shape
    out = np.full((H, W), -1e9, np.float32)
    rc_lo, rc_hi = bounds["rc"]; s_lo, s_hi = bounds["s"]
    for m in range(params.shape[0]):
        if mask[m] <= 0.0 or int(shapes[m]) not in CENTER_SHAPES:
            continue
        th, sx, sy = params[m, TH], params[m, SX], params[m, SY]
        rc = float(np.clip(params[m, RC], rc_lo, rc_hi))
        sx = float(np.clip(sx, s_lo, s_hi)); sy = float(np.clip(sy, s_lo, s_hi))
        tx, ty = params[m, TX], params[m, TY]
        cos, sin = np.cos(th), np.sin(th); inv_det = 1.0 / (sx * sy)
        dx = coords_np[..., 0] - tx; dy = coords_np[..., 1] - ty
        bx = inv_det * (sy * cos) * dx + inv_det * (sy * sin) * dy
        by = inv_det * (-sx * sin) * dx + inv_det * (sx * cos) * dy
        out = np.maximum(out, (rc - np.sqrt(bx * bx + by * by)).astype(np.float32))
    return out


# ===========================================================================
# 2. Killing-ABC transformation (steady -> unsteady) -- port of transformation.cpp
# ===========================================================================
def sample_observer(rng, ocfg):
    """Sample the observer's six motion scalars.

    Tobias interprets them as initial [xdot,ydot,theta_dot] plus constant
    acceleration; the legacy Killing port interprets them as linear abc(t).
    """
    a0 = rng.uniform(-ocfg["a0_range"], ocfg["a0_range"]); ad = rng.uniform(-ocfg["adot_range"], ocfg["adot_range"])
    b0 = rng.uniform(-ocfg["a0_range"], ocfg["a0_range"]); bd = rng.uniform(-ocfg["adot_range"], ocfg["adot_range"])
    c0 = rng.uniform(-ocfg["c0_range"], ocfg["c0_range"]); cd = rng.uniform(-ocfg["cdot_range"], ocfg["cdot_range"])
    six = np.array([a0, b0, c0, ad, bd, cd], np.float32)

    def abc_func(t):
        return np.array([a0 + ad * t, b0 + bd * t, c0 + cd * t], np.float64)
    return abc_func, six


def _killing_velocity(abc_func, center, pos, t):
    a, b, c = abc_func(t)
    rx, ry = pos[0] - center[0], pos[1] - center[1]
    return np.array([a - c * ry, b + c * rx], np.float64)


def integrate_observer_pathline(abc_func, start_pos, center, tmin, tmax, timesteps):
    """RK4-integrate the observer seed through its Killing field -> positions[T,2]."""
    dt = (tmax - tmin) / (timesteps - 1)
    pos = np.array(start_pos, np.float64)
    out = np.zeros((timesteps, 2), np.float64); out[0] = pos
    for i in range(1, timesteps):
        t = tmin + (i - 1) * dt
        k1 = _killing_velocity(abc_func, center, pos, t)
        k2 = _killing_velocity(abc_func, center, pos + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = _killing_velocity(abc_func, center, pos + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = _killing_velocity(abc_func, center, pos + dt * k3, t + dt)
        pos = pos + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i] = pos
    return out


def killing_abc_transform(vel_eval, abc_func, ocfg, coords_np):
    """Observe a steady field (vel_eval: coords->vel) with one Killing observer -> [T,G,G,2].
    Faithful port of transformation.cpp::killingABCtransformation (analytic branch)."""
    T = int(ocfg["timesteps"]); G = coords_np.shape[0]
    tmin, tmax = float(ocfg["tmin"]), float(ocfg["tmax"])
    center = np.asarray(ocfg["center"], np.float64)
    dt = (tmax - tmin) / (T - 1)

    path = integrate_observer_pathline(abc_func, ocfg["start_pos"], center, tmin, tmax, T)
    Os = path[0]

    finv = np.zeros((T, G, G, 2), np.float64)
    Q_all = np.zeros((T, 2, 2)); Qdot_all = np.zeros((T, 2, 2)); Tdot_all = np.zeros((T, 2))
    theta = 0.0
    for i in range(T):
        ti = tmin + i * dt
        if i > 0:
            theta += dt * float(abc_func(ti)[2])            # cumulative rotation angle
        ct, stt = math.cos(theta), math.sin(theta)
        R_i = np.array([[ct, -stt], [stt, ct]])              # observer rotation Rot(theta)
        Q_i = R_i.T                                          # frame rotation Q = R^T
        c_t = Os - Q_i @ path[i]                             # frame translation
        shifted = coords_np - c_t.reshape(1, 1, 2)
        finv[i, ..., 0] = R_i[0, 0] * shifted[..., 0] + R_i[0, 1] * shifted[..., 1]
        finv[i, ..., 1] = R_i[1, 0] * shifted[..., 0] + R_i[1, 1] * shifted[..., 1]
        a_now, b_now, c_now = abc_func(ti)
        Q_all[i] = Q_i
        Qdot_all[i] = Q_i @ np.array([[0.0, c_now], [-c_now, 0.0]])
        Tdot_all[i] = np.array([-a_now, -b_now])

    v = vel_eval(finv.reshape(T * G, G, 2)).astype(np.float64).reshape(T, G, G, 2)
    field = np.zeros((T, G, G, 2), np.float32)
    for i in range(T):
        Q, Qd, Td, vi, fi = Q_all[i], Qdot_all[i], Tdot_all[i], v[i], finv[i]
        field[i, ..., 0] = Q[0, 0] * vi[..., 0] + Q[0, 1] * vi[..., 1] + Qd[0, 0] * fi[..., 0] + Qd[0, 1] * fi[..., 1] + Td[0]
        field[i, ..., 1] = Q[1, 0] * vi[..., 0] + Q[1, 1] * vi[..., 1] + Qd[1, 0] * fi[..., 0] + Qd[1, 1] * fi[..., 1] + Td[1]
    return field


# ===========================================================================
# 3. Tobias observer + pathlines -- ports of transformation.cpp/flowGenerator.cpp
# ===========================================================================
def _steady_velocity_at_points(vel_eval, points):
    """Evaluate a steady-field callable at arbitrary [...,2] points."""
    points = np.asarray(points, dtype=np.float32)
    shape = points.shape
    if shape[-1] != 2:
        raise ValueError(f"Expected points shaped [...,2], got {shape}.")
    values = vel_eval(points.reshape(-1, 1, 2))
    return np.asarray(values, dtype=np.float64).reshape(shape)


def _tobias_kinematics(abc, abc_dot, tmin, tmax, timesteps):
    """Match the explicit update order in Tobias_ObserverTransformation."""
    if timesteps < 2:
        raise ValueError("Tobias observer requires at least two time steps.")
    abc = np.asarray(abc, dtype=np.float64)
    abc_dot = np.asarray(abc_dot, dtype=np.float64)
    if abc.shape != (3,) or abc_dot.shape != (3,):
        raise ValueError("abc and abc_dot must both contain three values.")

    dt = (float(tmax) - float(tmin)) / (int(timesteps) - 1)
    q_all = np.zeros((timesteps, 2, 2), dtype=np.float64)
    c_all = np.zeros((timesteps, 2), dtype=np.float64)
    velocities = np.zeros((timesteps, 2), dtype=np.float64)
    angular_velocities = np.zeros(timesteps, dtype=np.float64)
    q_all[0] = np.eye(2)
    velocities[0] = abc[:2]
    angular_velocities[0] = abc[2]

    theta = 0.0
    angular_velocity = float(abc[2])
    translation = np.zeros(2, dtype=np.float64)
    translation_velocity = abc[:2].copy()
    for i in range(1, timesteps):
        # This is intentionally explicit Euler and intentionally updates theta
        # before omega, exactly like transformation.cpp.
        theta += dt * angular_velocity
        angular_velocity += dt * float(abc_dot[2])
        angular_velocities[i] = angular_velocity

        translation += dt * translation_velocity
        translation_velocity += dt * abc_dot[:2]
        c_all[i] = translation
        velocities[i] = translation_velocity

        ct, st = math.cos(theta), math.sin(theta)
        q_all[i] = np.array([[ct, -st], [st, ct]], dtype=np.float64)
    return dt, q_all, c_all, velocities, angular_velocities


def _make_tobias_velocity_evaluator(
        vel_eval, abc, abc_dot, tmin, tmax, timesteps):
    """Create the analytical unsteady evaluator used both for grids and RK4."""
    dt, q_all, c_all, velocities, angular_velocities = _tobias_kinematics(
        abc, abc_dot, tmin, tmax, timesteps)

    def evaluate(points, time):
        points = np.asarray(points, dtype=np.float64)
        original_shape = points.shape
        if original_shape[-1] != 2:
            raise ValueError(f"Expected points shaped [...,2], got {original_shape}.")

        floating = (float(time) - float(tmin)) / dt
        lower = int(np.clip(math.floor(floating), 0, timesteps - 1))
        upper = int(np.clip(math.floor(floating) + 1, 0, timesteps - 1))
        ratio = floating - lower
        q_t = q_all[lower] * (1.0 - ratio) + q_all[upper] * ratio
        c_t = c_all[lower] * (1.0 - ratio) + c_all[upper] * ratio
        velocity = velocities[lower] * (1.0 - ratio) + velocities[upper] * ratio
        omega = (angular_velocities[lower] * (1.0 - ratio)
                 + angular_velocities[upper] * ratio)

        flat = points.reshape(-1, 2)
        finv = (flat - c_t) @ q_t
        steady_velocity = _steady_velocity_at_points(vel_eval, finv)
        spin = np.array([[0.0, omega], [-omega, 0.0]], dtype=np.float64)
        q_dot = q_t @ spin
        result = steady_velocity @ q_t.T + finv @ q_dot.T + velocity
        return result.reshape(original_shape)

    return evaluate


def tobias_observer_transformation(
        vel_eval, abc, abc_dot, ocfg, coords_np, return_evaluator=False):
    """Port of ``transformation.cpp::Tobias_ObserverTransformation``.

    Args:
        vel_eval: steady analytical velocity callable.
        abc: [xdot, ydot, theta_dot] at tmin.
        abc_dot: constant [xddot, yddot, theta_ddot].
        ocfg: observer config dictionary.
        coords_np: [Y,X,2] output grid.
        return_evaluator: also return the analytical unsteady velocity callable.
    """
    timesteps = int(ocfg["timesteps"])
    tmin, tmax = float(ocfg["tmin"]), float(ocfg["tmax"])
    evaluate = _make_tobias_velocity_evaluator(
        vel_eval, abc, abc_dot, tmin, tmax, timesteps)
    times = np.linspace(tmin, tmax, timesteps, dtype=np.float64)
    field = np.stack([evaluate(coords_np, time) for time in times]).astype(np.float32)
    return (field, evaluate) if return_evaluator else field


def Tobias_ObserverTransformation(
        vel_eval, abc, abc_dot, tmin, tmax, timestep, coords_np,
        return_evaluator=False):
    """C++-named convenience wrapper for tests and direct reuse."""
    ocfg = {"timesteps": int(timestep), "tmin": float(tmin), "tmax": float(tmax)}
    return tobias_observer_transformation(
        vel_eval, abc, abc_dot, ocfg, coords_np, return_evaluator)


def _grid_cross_seedings(k_lines, rng, jitter_std=0.05):
    """Four-arm cross sampling: left, down, right and up (no center seed)."""
    grid_count = int(k_lines) // 2
    if grid_count < 2:
        raise ValueError("pathline.count_k must be at least 4.")
    domain_min = -1.0
    domain_range = 2.0
    sampling_start = domain_min + 0.1 * domain_range
    sampling_end = domain_min + 0.9 * domain_range
    spacing = (sampling_end - sampling_start) / (grid_count - 1)
    cross_delta = spacing / 3.0

    seeds = []
    for iy in range(grid_count):
        for ix in range(grid_count):
            cx = sampling_start + spacing * ix + rng.normal(0.0, jitter_std)
            cy = sampling_start + spacing * iy + rng.normal(0.0, jitter_std)
            seeds.extend([
                (cx - cross_delta, cy),
                (cx, cy - cross_delta),
                (cx + cross_delta, cy),
                (cx, cy + cross_delta),
            ])
    seeds = np.asarray(seeds, dtype=np.float64)
    # C++ normally stays in range statistically; clipping makes a seeded sample
    # valid even for an unusually large Gaussian jitter.
    return np.clip(seeds, -1.0 + 1e-7, 1.0 - 1e-7)


def _integrate_pathline_positions(unsteady_eval, seeds, tmin, tmax, length):
    """Vectorized RK4 integration with the C++ strict domain-boundary rule."""
    times = np.linspace(tmin, tmax, int(length), dtype=np.float64)
    positions = np.full((len(seeds), len(times), 2), np.nan, dtype=np.float64)
    positions[:, 0] = seeds
    active = np.all((seeds > -1.0) & (seeds < 1.0), axis=1)
    lengths = active.astype(np.int64)

    for step in range(1, len(times)):
        if not np.any(active):
            break
        ids = np.flatnonzero(active)
        point = positions[ids, step - 1]
        time = times[step - 1]
        h = times[step] - time
        k1 = unsteady_eval(point, time)
        k2 = unsteady_eval(point + 0.5 * h * k1, time + 0.5 * h)
        k3 = unsteady_eval(point + 0.5 * h * k2, time + 0.5 * h)
        k4 = unsteady_eval(point + h * k3, time + h)
        new_point = point + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        inside = (np.all(np.isfinite(new_point), axis=1)
                  & np.all((new_point > -1.0) & (new_point < 1.0), axis=1))
        valid_ids = ids[inside]
        positions[valid_ids, step] = new_point[inside]
        lengths[valid_ids] += 1
        active[ids[~inside]] = False
    return positions, lengths, times


def _compute_ivd_fields(field):
    """Match ComputeCurl + ComputeIVD on the sampled [-1,1]^2 velocity grid."""
    timesteps, ydim, xdim, _ = field.shape
    dx = 2.0 / (xdim - 1)
    dy = 2.0 / (ydim - 1)
    curl = np.zeros((timesteps, ydim, xdim), dtype=np.float64)
    curl[:, 1:-1, 1:-1] = (
        (field[:, 1:-1, 2:, 1] - field[:, 1:-1, :-2, 1]) / (2.0 * dx)
        - (field[:, 2:, 1:-1, 0] - field[:, :-2, 1:-1, 0]) / (2.0 * dy)
    )
    mean_curl = curl.sum(axis=(1, 2)) / ((xdim - 2) * (ydim - 2))
    ivd = np.zeros_like(curl)
    ivd[:, 1:-1, 1:-1] = np.abs(
        curl[:, 1:-1, 1:-1] - mean_curl[:, None, None])
    return ivd


def _trilinear_sample_scalar(field, points, times, tmin, tmax):
    """Sample [T,Y,X] data with the clamped interpolation used by C++."""
    points = np.asarray(points, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    timesteps, ydim, xdim = field.shape
    fx = np.clip((points[..., 0] + 1.0) * 0.5 * (xdim - 1), 0, xdim - 1)
    fy = np.clip((points[..., 1] + 1.0) * 0.5 * (ydim - 1), 0, ydim - 1)
    ft = np.clip(
        (times - tmin) / (tmax - tmin) * (timesteps - 1), 0, timesteps - 1)
    x0, y0, t0 = np.floor(fx).astype(int), np.floor(fy).astype(int), np.floor(ft).astype(int)
    x1 = np.minimum(x0 + 1, xdim - 1)
    y1 = np.minimum(y0 + 1, ydim - 1)
    t1 = np.minimum(t0 + 1, timesteps - 1)
    wx, wy, wt = fx - x0, fy - y0, ft - t0

    f00 = field[t0, y0, x0] * (1.0 - wx) + field[t0, y0, x1] * wx
    f01 = field[t0, y1, x0] * (1.0 - wx) + field[t0, y1, x1] * wx
    at_t0 = f00 * (1.0 - wy) + f01 * wy
    f10 = field[t1, y0, x0] * (1.0 - wx) + field[t1, y0, x1] * wx
    f11 = field[t1, y1, x0] * (1.0 - wx) + field[t1, y1, x1] * wx
    at_t1 = f10 * (1.0 - wy) + f11 * wy
    return at_t0 * (1.0 - wt) + at_t1 * wt


def collect_pathline_info(
        unsteady_eval, sampled_field, label_eval, pcfg, ocfg, rng):
    """Port of PathlineIntegrationInfoCollect2D.

    Returns [num_lines, length, 7], where features are
    [x, y, time, IVD, distance_or_seed_label, u, v].
    """
    length = int(pcfg["length"])
    seeds = _grid_cross_seedings(
        int(pcfg["count_k"]), rng, float(pcfg["jitter_std"]))
    positions, valid_lengths, times = _integrate_pathline_positions(
        unsteady_eval, seeds, ocfg["tmin"], ocfg["tmax"], length)

    # C++ moves seeds toward the domain center until a path has >2 points.
    for _ in range(int(pcfg["max_reseed_attempts"])):
        failed = valid_lengths <= 2
        if not np.any(failed):
            break
        shifts = rng.uniform(1e-5, 0.5, int(failed.sum()))
        seeds[failed] *= (1.0 - shifts[:, None])
        positions, valid_lengths, times = _integrate_pathline_positions(
            unsteady_eval, seeds, ocfg["tmin"], ocfg["tmax"], length)

    padding = float(pcfg["padding_value"])
    features = np.full((len(seeds), length, 7), padding, dtype=np.float32)
    valid = np.isfinite(positions[..., 0])
    line_ids, step_ids = np.nonzero(valid)
    valid_points = positions[line_ids, step_ids]
    valid_times = times[step_ids]
    velocities = np.empty_like(valid_points)
    # Group evaluation by time to preserve the scalar-time Tobias interpolation.
    for step in np.unique(step_ids):
        at_step = step_ids == step
        velocities[at_step] = unsteady_eval(valid_points[at_step], times[step])

    ivd = _compute_ivd_fields(sampled_field)
    ivd_values = _trilinear_sample_scalar(
        ivd, valid_points, valid_times, ocfg["tmin"], ocfg["tmax"])
    distance = np.linalg.norm(valid_points - seeds[line_ids], axis=1)
    features[line_ids, step_ids, 0:2] = valid_points
    features[line_ids, step_ids, 2] = valid_times
    features[line_ids, step_ids, 3] = ivd_values
    features[line_ids, step_ids, 4] = distance
    features[line_ids, step_ids, 5:7] = velocities

    seed_sd = label_eval(seeds.reshape(-1, 1, 2)).reshape(-1)
    seed_labels = (seed_sd > 0.0).astype(np.float32)
    features[:, 0, 4] = seed_labels
    return features, seed_labels, valid_lengths


def PathlineIntegrationInfoCollect2D(
        unsteady_eval, sampled_field, label_eval, count_k, output_pathline_length,
        tmin=0.0, tmax=1.0, rng=None):
    """C++-named convenience wrapper using GRID_CROSS_SEEDING."""
    if rng is None:
        rng = np.random.default_rng()
    pcfg = {
        "count_k": int(count_k),
        "length": int(output_pathline_length),
        "jitter_std": 0.05,
        "padding_value": -1000.0,
        "max_reseed_attempts": 32,
    }
    ocfg = {"tmin": float(tmin), "tmax": float(tmax)}
    return collect_pathline_info(
        unsteady_eval, sampled_field, label_eval, pcfg, ocfg, rng)


def _write_cereal_float_vector(file_path, values):
    """Write a Cereal BinaryOutputArchive-compatible ``std::vector<float>``.

    Cereal serializes the vector size as one uint64 followed by contiguous
    float32 values. ``flatten3DvecAs1Dfloat`` uses pathline-major,
    point-major, feature-major order, identical to NumPy C-order here.
    """
    flat = np.ascontiguousarray(values, dtype="<f4").reshape(-1)
    if not np.all(np.isfinite(flat)):
        raise ValueError(f"Cannot write non-finite pathline data to {file_path}.")
    with open(file_path, "wb") as handle:
        handle.write(np.asarray([flat.size], dtype="<u8").tobytes())
        handle.write(flat.tobytes(order="C"))


def _write_sample_meta_json(file_path, grid_size, time_steps, iso_value):
    """Write the per-sample metadata shared by pathline/segmentation binaries."""
    payload = {
        "Xdim": int(grid_size),
        "Ydim": int(grid_size),
        "time_steps": int(time_steps),
        "Patch_domainMinBoundary": {
            "v0": -1.0,
            "v1": -1.0,
        },
        "Patch_domainMaxBoundary": {
            "v0": 1.0,
            "v1": 1.0,
        },
        "IsoValue": float(iso_value),
    }
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)


def _pathline_sample_type_tag(spec):
    """Compact filename tag analogous to C++'s concatenated vortex type name."""
    if spec["type"] == "killing":
        return "zero_field"
    active_shapes = [
        int(shape) for shape, active in zip(spec["shapes"], spec["mask"])
        if active > 0.0
    ]
    if not active_shapes:
        return "zero_field"
    names = {0: "saddle", 1: "center_cw", 2: "center_ccw"}
    return "_".join(names.get(shape, f"shape{shape}") for shape in active_shapes)


def _bilinear_sample_numpy(data, x, y):
    """Vectorized clamped bilinear sampling for [H,W] or [H,W,C] arrays."""
    data = np.asarray(data)
    height, width = data.shape[:2]
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, width - 1.0)
    y = np.clip(np.asarray(y, dtype=np.float64), 0.0, height - 1.0)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    wx, wy = x - x0, y - y0
    if data.ndim == 3:
        wx, wy = wx[..., None], wy[..., None]
    top = data[y0, x0] * (1.0 - wx) + data[y0, x1] * wx
    bottom = data[y1, x0] * (1.0 - wx) + data[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def _lic_output_coords(image_size):
    """Match C++ LIC's x/image_size and y/image_size sampling convention."""
    ratio = np.arange(int(image_size), dtype=np.float64) / float(image_size)
    yy, xx = np.meshgrid(ratio, ratio, indexing="ij")
    return np.stack([2.0 * xx - 1.0, 2.0 * yy - 1.0], axis=-1)


def _velocity_colormap(values):
    """Small dependency-free approximation of the C++ scalar colormap."""
    anchors = np.asarray([
        [0.267, 0.005, 0.329],
        [0.230, 0.322, 0.546],
        [0.128, 0.567, 0.551],
        [0.369, 0.789, 0.383],
        [0.993, 0.906, 0.144],
    ], dtype=np.float64)
    scaled = np.clip(values, 0.0, 1.0) * (len(anchors) - 1)
    lower = np.floor(scaled).astype(np.int64)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    weight = (scaled - lower)[..., None]
    return anchors[lower] * (1.0 - weight) + anchors[upper] * weight


def render_lic_image(field_slice, lcfg, noise_texture, label_eval=None):
    """Vectorized analogue of C++ LICAlgorithm for one steady time slice."""
    field_slice = np.asarray(field_slice, dtype=np.float64)
    if field_slice.ndim != 3 or field_slice.shape[-1] != 2:
        raise ValueError("LIC expects one velocity slice shaped [Y,X,2].")
    size = int(lcfg["image_size"])
    positions0 = _lic_output_coords(size)
    height, width = field_slice.shape[:2]
    noise_height, noise_width = noise_texture.shape
    accumulated = np.zeros((size, size), dtype=np.float64)
    counts = np.zeros((size, size), dtype=np.int32)

    for direction in (1.0, -1.0):
        positions = positions0.copy()
        active = np.ones((size, size), dtype=bool)
        for _ in range(int(lcfg["max_steps"])):
            field_x = (positions[..., 0] + 1.0) * 0.5 * (width - 1)
            field_y = (positions[..., 1] + 1.0) * 0.5 * (height - 1)
            active &= (
                np.isfinite(field_x) & np.isfinite(field_y)
                & (field_x >= 0.0) & (field_x < width)
                & (field_y >= 0.0) & (field_y < height)
            )
            if not np.any(active):
                break
            noise_x = field_x * (noise_width - 1) / max(1, width - 1)
            noise_y = field_y * (noise_height - 1) / max(1, height - 1)
            noise_values = _bilinear_sample_numpy(
                noise_texture, noise_x, noise_y)
            accumulated[active] += noise_values[active]
            counts[active] += 1
            velocity = _bilinear_sample_numpy(
                field_slice, field_x, field_y)
            positions[active] += (
                direction * float(lcfg["step_size"]) * velocity[active])

    lic = accumulated / np.maximum(counts, 1)
    rgb = np.repeat(lic[..., None], 3, axis=-1)
    if lcfg["color_by_velocity_magnitude"]:
        speed = np.linalg.norm(field_slice, axis=-1)
        query = positions0
        sx = (query[..., 0] + 1.0) * 0.5 * (width - 1)
        sy = (query[..., 1] + 1.0) * 0.5 * (height - 1)
        speed_at_pixels = _bilinear_sample_numpy(speed, sx, sy)
        speed_min, speed_max = float(speed.min()), float(speed.max())
        if speed_max > speed_min + 1e-12:
            normalized_speed = (
                (speed_at_pixels - speed_min) / (speed_max - speed_min))
            criterion_color = _velocity_colormap(normalized_speed)
            whiteish = np.clip((lic - 0.4) * (1.5 / 0.4), 0.0, 1.0)
            rgb = (
                rgb * (1.0 - whiteish[..., None])
                + criterion_color * whiteish[..., None])

    if lcfg["overlay_segmentation"] and label_eval is not None:
        labels = label_eval(positions0.astype(np.float32))
        inside = np.asarray(labels) > 0.0
        rgb[inside] = (
            0.5 * rgb[inside] + 0.5 * np.array([1.0, 1.0, 0.0]))
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


def _save_png(image, file_path):
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Pillow is required when lic.enabled is true.") from exc
    Image.fromarray(image, mode="RGB").save(file_path)


def _make_lic_noise(lcfg, seed):
    rng = np.random.default_rng(int(seed) + int(lcfg["noise_seed_offset"]))
    size = int(lcfg["noise_size"])
    return rng.random((size, size), dtype=np.float32)


def read_lic_cfg(cfg):
    lic = cfg.get("lic", {}) or {}
    return {
        "enabled": bool(lic.get("enabled", True)),
        "image_size": int(lic.get("image_size", 256)),
        "noise_size": int(lic.get("noise_size", 256)),
        "noise_seed_offset": int(lic.get("noise_seed_offset", 811)),
        "step_size": float(lic.get("step_size", 0.002)),
        "max_steps": int(lic.get("max_steps", 64)),
        "frame_stride": max(1, int(lic.get("frame_stride", 1))),
        "color_by_velocity_magnitude": bool(
            lic.get("color_by_velocity_magnitude", True)),
        "overlay_segmentation": bool(
            lic.get("overlay_segmentation", True)),
    }


def save_steady_lic_images(
        steady_fields, label_evals, observers_per_field, out_dir, lcfg, seed):
    """Write one ``*_steady_lic.png`` for every steady field."""
    if not lcfg["enabled"]:
        return 0
    lic_dir = os.path.join(out_dir, "velocity_mix", "LIC")
    os.makedirs(lic_dir, exist_ok=True)
    noise = _make_lic_noise(lcfg, seed)
    for steady_index, field in enumerate(steady_fields):
        first_id = steady_index * observers_per_field
        last_id = first_id + observers_per_field - 1
        image = render_lic_image(
            field, lcfg, noise, label_evals[steady_index])
        _save_png(
            image,
            os.path.join(
                lic_dir, f"{first_id}_{last_id}_steady_lic.png"))
    logging.info(
        f"[LIC] wrote {len(steady_fields)} steady LIC images to {lic_dir}.")
    return len(steady_fields)


# ===========================================================================
# Orchestration
# ===========================================================================
def read_observer_cfg(cfg):
    o = cfg.observer
    return {"per_field": int(o.per_field), "timesteps": int(o.timesteps),
            "grid": int(cfg.generate.grid), "tmin": float(o.tmin), "tmax": float(o.tmax),
            "center": list(o.center), "start_pos": list(o.start_pos),
            "a0_range": float(o.a0_range), "adot_range": float(o.adot_range),
            "c0_range": float(o.c0_range), "cdot_range": float(o.cdot_range),
            "make_labels": bool(o.make_labels),
            "transformation": str(o.get("transformation", "tobias")).lower()}


def read_pathline_cfg(cfg):
    p = cfg.get("pathline", {}) or {}
    return {
        "enabled": bool(p.get("enabled", True)),
        "count_k": int(p.get("count_k", 16)),
        "length": int(p.get("length", 16)),
        "sampling": str(p.get("sampling", "GRID_CROSS_SEEDING")),
        "jitter_std": float(p.get("jitter_std", 0.05)),
        "padding_value": float(p.get("padding_value", -1000.0)),
        "max_reseed_attempts": int(p.get("max_reseed_attempts", 32)),
    }


def render_steady_set(specs, ocfg, bounds, device):
    """Render all steady fields + their shared (t0) segmentation label, for saving."""
    G = ocfg["grid"]
    grid = make_local_grid(G, device).cpu().numpy().astype(np.float32)
    N = len(specs)
    fields = np.zeros((N, G, G, 2), np.float32)
    labels = np.zeros((N, G, G), np.float32)
    source = np.zeros(N, np.int64)
    evals, label_evals = [], []
    for i, spec in enumerate(specs):
        vel, lab = make_steady_eval(spec, bounds, device)
        fields[i] = vel(grid)
        labels[i] = lab(grid)
        source[i] = spec["source"]
        evals.append(vel)
        label_evals.append(lab)
    return grid, fields, labels, source, evals, label_evals


def generate_unsteady(
        specs, evals, label_evals, steady_labels, grid, ocfg, pcfg, lcfg, cfg,
        out_dir, write_full):
    """For every steady field, sample `per_field` observers and transform. Streams the
    fields, pathlines, labels and metadata to sharded NPZ files."""
    N = len(specs); per = ocfg["per_field"]
    target_total = N * per
    rng = np.random.default_rng(int(cfg.get("seed", 0)) + 17)

    ucfg = cfg.output
    subset = int(ucfg.get("unsteady_subset", 400))
    shard_size = int(ucfg.get("shard_size", 2000))
    to_write = target_total if write_full else min(subset, target_total)
    save_unsteady_npz = bool(ucfg.get("save_unsteady_npz", True))
    save_pathline_bin = bool(ucfg.get("save_pathline_bin", True))
    pathline_bin_dir = os.path.join(
        out_dir, str(ucfg.get("pathline_bin_dir", "velocity_mix")))
    if save_pathline_bin and not pcfg["enabled"]:
        raise ValueError(
            "output.save_pathline_bin requires pathline.enabled: true.")
    if save_pathline_bin:
        os.makedirs(pathline_bin_dir, exist_ok=True)
    save_segmentation_bin = bool(
        ucfg.get("save_segmentation_bin", True))
    segmentation_bin_dir = os.path.join(
        out_dir,
        str(ucfg.get(
            "segmentation_bin_dir",
            ucfg.get("pathline_bin_dir", "velocity_mix"))))
    if save_segmentation_bin:
        os.makedirs(segmentation_bin_dir, exist_ok=True)
    save_sample_meta = bool(ucfg.get("save_sample_meta", True))
    sample_meta_dir = os.path.join(
        out_dir,
        str(ucfg.get(
            "sample_meta_dir",
            ucfg.get("pathline_bin_dir", "velocity_mix"))))
    sample_meta_iso_value = float(ucfg.get("iso_value", 0.0))
    if save_sample_meta:
        os.makedirs(sample_meta_dir, exist_ok=True)
    lic_dir = os.path.join(out_dir, "velocity_mix", "LIC")
    lic_noise = None
    if lcfg["enabled"]:
        os.makedirs(lic_dir, exist_ok=True)
        lic_noise = _make_lic_noise(lcfg, int(cfg.get("seed", 0)))

    # visit steady fields in an interleaved order so a subset still spans all 3 sources.
    order = rng.permutation(N)
    shard_fields, shard_sidx, shard_obs = [], [], []
    shard_pathlines, shard_pathline_labels = [], []
    shard_pathline_lengths, shard_segmentations = [], []
    written = 0
    shard_id = 0
    abc_all = np.zeros((to_write, 6), np.float32)
    min_velocity, max_velocity = np.inf, -np.inf
    deformed_lic_count = 0

    def flush():
        nonlocal shard_id, shard_fields, shard_sidx, shard_obs
        nonlocal shard_pathlines, shard_pathline_labels
        nonlocal shard_pathline_lengths, shard_segmentations
        if not shard_fields:
            return
        payload = dict(
            fields=np.stack(shard_fields).astype(np.float32),
            steady_index=np.asarray(shard_sidx, np.int64),
            observer=np.stack(shard_obs).astype(np.float32),
            segmentation=np.stack(shard_segmentations).astype(np.uint8))
        if pcfg["enabled"]:
            payload.update(
                pathlines=np.stack(shard_pathlines).astype(np.float32),
                pathline_labels=np.stack(shard_pathline_labels).astype(np.float32),
                pathline_valid_lengths=np.stack(shard_pathline_lengths).astype(np.int16))
        np.savez_compressed(
            os.path.join(out_dir, f"unsteady_shard_{shard_id:04d}.npz"),
            **payload)
        logging.info(f"[unsteady] wrote shard {shard_id} ({len(shard_fields)} fields).")
        shard_id += 1
        shard_fields, shard_sidx, shard_obs = [], [], []
        shard_pathlines, shard_pathline_labels = [], []
        shard_pathline_lengths, shard_segmentations = [], []

    done = False
    for si in order:
        vel = evals[si]
        label_eval = label_evals[si]
        for _ in range(per):
            abc_func, six = sample_observer(rng, ocfg)
            sample_tag = (
                f"sample_{written}{_pathline_sample_type_tag(specs[si])}")
            if save_sample_meta:
                # Match the C++/loader naming convention: replacing
                # "meta.json" with "_pathline.bin" yields the paired file.
                _write_sample_meta_json(
                    os.path.join(sample_meta_dir, f"{sample_tag}meta.json"),
                    ocfg["grid"],
                    ocfg["timesteps"],
                    sample_meta_iso_value)
            if ocfg["transformation"] == "tobias":
                field, unsteady_eval = tobias_observer_transformation(
                    vel, six[:3], six[3:], ocfg, grid, return_evaluator=True)
            elif ocfg["transformation"] in {"killing", "killing_abc"}:
                if pcfg["enabled"]:
                    raise ValueError(
                        "Pathline generation currently requires the Tobias observer. "
                        "Set observer.transformation: tobias or pathline.enabled: false.")
                field = killing_abc_transform(vel, abc_func, ocfg, grid)
                unsteady_eval = None
            else:
                raise ValueError(
                    "observer.transformation must be 'tobias' or 'killing', got "
                    f"{ocfg['transformation']!r}.")

            segmentation = (steady_labels[si] > 0.0).astype(np.uint8)
            if save_unsteady_npz:
                shard_fields.append(field)
                shard_sidx.append(int(si))
                shard_obs.append(six)
                shard_segmentations.append(segmentation)
            if save_segmentation_bin:
                # flowGenerator.cpp calls flatten2DvecAs1Dfloat, so the mask is
                # serialized as vector<float>, despite originating as uint8.
                _write_cereal_float_vector(
                    os.path.join(
                        segmentation_bin_dir,
                        f"{sample_tag}_segmentation.bin"),
                    segmentation.astype(np.float32))
            if pcfg["enabled"]:
                pathlines, pathline_labels, valid_lengths = collect_pathline_info(
                    unsteady_eval, field, label_eval, pcfg, ocfg, rng)
                if save_unsteady_npz:
                    shard_pathlines.append(pathlines)
                    shard_pathline_labels.append(pathline_labels)
                    shard_pathline_lengths.append(valid_lengths)
                if save_pathline_bin:
                    _write_cereal_float_vector(
                        os.path.join(
                            pathline_bin_dir, f"{sample_tag}_pathline.bin"),
                        pathlines)
            if lcfg["enabled"]:
                for frame_index in range(
                        0, field.shape[0], int(lcfg["frame_stride"])):
                    image = render_lic_image(
                        field[frame_index], lcfg, lic_noise, label_eval)
                    _save_png(
                        image,
                        os.path.join(
                            lic_dir,
                            f"{sample_tag}deformed_{frame_index}lic.png"))
                    deformed_lic_count += 1
            min_velocity = min(min_velocity, float(np.min(field)))
            max_velocity = max(max_velocity, float(np.max(field)))
            abc_all[written] = six
            written += 1
            if save_unsteady_npz and len(shard_fields) >= shard_size:
                flush()
            if written >= to_write:
                done = True
                break
        if (written % max(1, to_write // 10 or 1) == 0) or done:
            logging.info(f"[unsteady] transformed {written}/{to_write} "
                         f"(target total = {target_total})")
        if done:
            break
    flush()
    stats = {
        "min_velocity": float(min_velocity) if written else float("nan"),
        "max_velocity": float(max_velocity) if written else float("nan"),
        "pathline_count": (
            (pcfg["count_k"] // 2) ** 2 * 4 if pcfg["enabled"] else 0),
        "pathline_feature": 7 if pcfg["enabled"] else 0,
        "pathline_bin_count": written if save_pathline_bin else 0,
        "pathline_bin_dir": (
            os.path.relpath(pathline_bin_dir, out_dir)
            if save_pathline_bin else None),
        "segmentation_bin_count": written if save_segmentation_bin else 0,
        "segmentation_bin_dir": (
            os.path.relpath(segmentation_bin_dir, out_dir)
            if save_segmentation_bin else None),
        "sample_meta_count": written if save_sample_meta else 0,
        "sample_meta_dir": (
            os.path.relpath(sample_meta_dir, out_dir)
            if save_sample_meta else None),
        "iso_value": sample_meta_iso_value,
        "unsteady_npz_count": written if save_unsteady_npz else 0,
        "unsteady_npz_shard_count": shard_id if save_unsteady_npz else 0,
        "deformed_lic_count": deformed_lic_count,
        "lic_dir": (
            os.path.relpath(lic_dir, out_dir)
            if lcfg["enabled"] else None),
    }
    return target_total, written, abc_all[:written], stats


def save_steady(out_dir, specs, grid, fields, labels, source, counts):
    os.makedirs(out_dir, exist_ok=True)
    # pack vatistas params (killing rows zeroed) + killing abc for reproducibility.
    N = len(specs)
    M = max((s["params"].shape[0] for s in specs if s["type"] == "vatistas"), default=2)
    params = np.zeros((N, M, 7), np.float32)
    shapes = np.zeros((N, M), np.int64)
    mask = np.zeros((N, M), np.float32)
    killing_abc = np.zeros((N, 3), np.float32)
    for i, s in enumerate(specs):
        if s["type"] == "vatistas":
            m = s["params"].shape[0]
            params[i, :m] = s["params"]; shapes[i, :m] = s["shapes"]; mask[i, :m] = s["mask"]
        else:
            killing_abc[i] = s["abc"]
    np.savez_compressed(
        os.path.join(out_dir, "steady_fields.npz"),
        params=params, shapes=shapes, mask=mask, killing_abc=killing_abc,
        source=source, fields=fields, labels=labels, coords=grid,
        counts=np.array([counts[SRC_RANDOM], counts[SRC_FITTED], counts[SRC_KILLING]], np.int64))
    logging.info(f"[save] wrote {N} steady fields (+shared labels) to {out_dir}/steady_fields.npz")


def save_plots(out_dir, specs, grid, steady_fields, steady_labels, source, ocfg, evals, cfg):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as ex:
        logging.warning(f"[plots] matplotlib unavailable ({ex}); skipping."); return
    G = steady_fields.shape[1]
    lin = np.linspace(-1, 1, G); X, Y = np.meshgrid(lin, lin); step = max(1, G // 20)
    rng = np.random.default_rng(int(cfg.get("seed", 0)) + 123)

    def quiver(ax, vel, title, label=None):
        sp = np.sqrt(vel[..., 0] ** 2 + vel[..., 1] ** 2)
        ax.imshow(sp, extent=[-1, 1, -1, 1], origin="lower", cmap="viridis", alpha=0.75)
        if label is not None and label.max() > 0:
            ax.contour(X, Y, label, levels=[0.0], colors="red", linewidths=1.5)
        ax.quiver(X[::step, ::step], Y[::step, ::step], vel[::step, ::step, 0], vel[::step, ::step, 1],
                  color="white", scale=None)
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])

    picks = []
    for src, name in [(SRC_RANDOM, "random"), (SRC_FITTED, "fitted"), (SRC_KILLING, "killing")]:
        ids = np.where(source == src)[0]
        if len(ids):
            picks.append((int(ids[0]), name))
    T = ocfg["timesteps"]; frames = [0, T // 2, T - 1]
    for which, name in picks:
        obs_func, six = sample_observer(rng, ocfg)
        if ocfg["transformation"] == "tobias":
            uf = tobias_observer_transformation(
                evals[which], six[:3], six[3:], ocfg, grid)
        else:
            uf = killing_abc_transform(evals[which], obs_func, ocfg, grid)
        fig, axes = plt.subplots(1, 1 + len(frames), figsize=(4 * (1 + len(frames)), 4))
        quiver(axes[0], steady_fields[which], f"steady ({name}) #{which}", steady_labels[which])
        for a, t in zip(axes[1:], frames):
            quiver(a, uf[t], f"observed t={t}/{T-1}")
        fig.suptitle(f"Steady Vatistas field -> {ocfg['transformation']}-observed frames "
                     "(red = shared objective vortex label at t0)")
        fig.tight_layout()
        p = os.path.join(out_dir, f"preview_{name}_{which}.png")
        fig.savefig(p, dpi=120); plt.close(fig); logging.info(f"[plots] saved {p}")


def _save_dataset_meta(out_dir, ocfg, pcfg, stats, target_total, written):
    """Write total metadata at the output root and in ``velocity_mix``."""
    meta = {
        "format": "pyflowvis_npz_shards_v2",
        "Xdim": int(ocfg["grid"]),
        "Ydim": int(ocfg["grid"]),
        "unsteadyFieldTimeStep": int(ocfg["timesteps"]),
        "domainMinBoundary": {
            "value0": -1.0,
            "value1": -1.0,
        },
        "domainMaxBoundary": {
            "value0": 1.0,
            "value1": 1.0,
        },
        "tmin": float(ocfg["tmin"]),
        "tmax": float(ocfg["tmax"]),
        "minV": stats["min_velocity"],
        "maxV": stats["max_velocity"],
        "observerTransformation": ocfg["transformation"],
        "pathlineSamplingMethod": pcfg["sampling"] if pcfg["enabled"] else None,
        "outputPathlineLength": int(pcfg["length"]) if pcfg["enabled"] else 0,
        "outputPathlinesCountK": int(pcfg["count_k"]) if pcfg["enabled"] else 0,
        "PathlineCount": int(stats["pathline_count"]),
        "PathlineFeature": int(stats["pathline_feature"]),
        "pathlineFeatureOrder": [
            "position_x", "position_y", "time", "ivd",
            "distance_or_seed_label", "velocity_u", "velocity_v",
        ] if pcfg["enabled"] else [],
        "pathlineBinaryDirectory": stats["pathline_bin_dir"],
        "pathlineBinaryCount": int(stats["pathline_bin_count"]),
        "pathlineBinaryLayout": (
            "cereal uint64 count + float32[pathline,point,feature]"
            if stats["pathline_bin_count"] else None),
        "segmentationBinaryDirectory": stats["segmentation_bin_dir"],
        "segmentationBinaryCount": int(stats["segmentation_bin_count"]),
        "segmentationBinaryLayout": (
            "cereal uint64 count + float32[y,x]"
            if stats["segmentation_bin_count"] else None),
        "sampleMetaDirectory": stats["sample_meta_dir"],
        "sampleMetaCount": int(stats["sample_meta_count"]),
        "IsoValue": float(stats["iso_value"]),
        "unsteadyNpzCount": int(stats["unsteady_npz_count"]),
        "unsteadyNpzShardCount": int(stats["unsteady_npz_shard_count"]),
        "licDirectory": stats["lic_dir"],
        "steadyLicCount": int(stats["steady_lic_count"]),
        "deformedLicCount": int(stats["deformed_lic_count"]),
        "targetUnsteady": int(target_total),
        "writtenUnsteady": int(written),
    }
    meta_paths = [
        os.path.join(out_dir, "meta.json"),
        os.path.join(out_dir, "velocity_mix", "meta.json"),
    ]
    os.makedirs(os.path.dirname(meta_paths[1]), exist_ok=True)
    for meta_path in meta_paths:
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)


def main():
    ap = argparse.ArgumentParser(
        description="Generate Vatistas/Tobias unsteady velocity, pathline and segmentation data.")
    ap.add_argument("--config", type=str, default="config/VatistasDataset.yaml")
    ap.add_argument("--refit", action="store_true", help="re-run the full parameter fit instead of loading the cache")
    ap.add_argument(
        "--full", action="store_true",
        help="materialize all 60000 unsteady variants (velocity + pathline data, tens of GB)")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    cfg = EasyConfig(); cfg.load(args.config, recursive=False)
    seed = int(cfg.get("seed", 0)); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    bounds = build_bounds(cfg)
    logging.info(f"[main] device={device}, seed={seed}")

    # 0. fit + distribution
    if args.refit:
        dist, fitted = refit(cfg.get("fit_config", "config/FittingVatistas.yaml"), device)
    else:
        dist, fitted = load_fit_outputs(cfg.get("fit_dir", "outputs/vatistas_fit"))

    # 1. steady pool (random + fitted + killing = 3000)
    specs, counts = build_steady_specs(dist, fitted, cfg, bounds, device)
    ocfg = read_observer_cfg(cfg)
    pcfg = read_pathline_cfg(cfg)
    lcfg = read_lic_cfg(cfg)
    if pcfg["sampling"].upper() != "GRID_CROSS_SEEDING":
        raise ValueError("Only GRID_CROSS_SEEDING is supported, matching flowGenerator.cpp.")
    grid, steady_fields, steady_labels, source, evals, label_evals = render_steady_set(
        specs, ocfg, bounds, device)

    out_dir = args.out_dir or cfg.output.get("out_dir", "outputs/vatistas_unsteady")
    os.makedirs(out_dir, exist_ok=True)
    if bool(cfg.output.get("save_steady", True)):
        save_steady(out_dir, specs, grid, steady_fields, steady_labels, source, counts)
    steady_lic_count = save_steady_lic_images(
        steady_fields, label_evals, ocfg["per_field"], out_dir, lcfg, seed)

    # 2. Tobias observation + pathline integration, mirroring
    # generateMixUnsteadyFieldPathline (60000 target at the paper scale).
    target_total, written, abc_all, stats = generate_unsteady(
        specs, evals, label_evals, steady_labels, grid, ocfg, pcfg, lcfg, cfg,
        out_dir, args.full)
    stats["steady_lic_count"] = steady_lic_count

    np.savez_compressed(os.path.join(out_dir, "dataset_manifest.npz"),
                        target_steady=np.int64(len(specs)), target_unsteady=np.int64(target_total),
                        written_unsteady=np.int64(written), observers_per_field=np.int64(ocfg["per_field"]),
                        source_counts=np.array([counts[SRC_RANDOM], counts[SRC_FITTED], counts[SRC_KILLING]], np.int64),
                        written_observers=abc_all,
                        observer_transformation=np.asarray(ocfg["transformation"]),
                        pathline_count=np.int64(stats["pathline_count"]),
                        pathline_length=np.int64(pcfg["length"] if pcfg["enabled"] else 0),
                        pathline_feature=np.int64(stats["pathline_feature"]),
                        pathline_bin_count=np.int64(stats["pathline_bin_count"]),
                        pathline_bin_dir=np.asarray(
                            stats["pathline_bin_dir"] or ""),
                        segmentation_bin_count=np.int64(
                            stats["segmentation_bin_count"]),
                        segmentation_bin_dir=np.asarray(
                            stats["segmentation_bin_dir"] or ""),
                        sample_meta_count=np.int64(
                            stats["sample_meta_count"]),
                        sample_meta_dir=np.asarray(
                            stats["sample_meta_dir"] or ""),
                        iso_value=np.float32(stats["iso_value"]),
                        unsteady_npz_count=np.int64(
                            stats["unsteady_npz_count"]),
                        unsteady_npz_shard_count=np.int64(
                            stats["unsteady_npz_shard_count"]),
                        steady_lic_count=np.int64(stats["steady_lic_count"]),
                        deformed_lic_count=np.int64(
                            stats["deformed_lic_count"]),
                        lic_dir=np.asarray(stats["lic_dir"] or ""))
    _save_dataset_meta(out_dir, ocfg, pcfg, stats, target_total, written)

    logging.info("=" * 78)
    logging.info(f"[totals] PAPER TARGET  steady = {len(specs)}  "
                 f"(random {counts[SRC_RANDOM]} + fitted {counts[SRC_FITTED]} + killing {counts[SRC_KILLING]})")
    logging.info(f"[totals] PAPER TARGET  unsteady = {target_total}  "
                 f"(= {len(specs)} steady x {ocfg['per_field']} observers)")
    logging.info(f"[totals] MATERIALIZED  steady = {len(specs)}  unsteady = {written}"
                 f"{'  (FULL)' if args.full else '  (subset; pass --full for all 60000)'}")
    if pcfg["enabled"]:
        logging.info(
            f"[totals] PATHLINES per sample = {stats['pathline_count']} x "
            f"{pcfg['length']} points x {stats['pathline_feature']} features")
        if stats["pathline_bin_count"]:
            logging.info(
                f"[totals] PATHLINE BIN files = {stats['pathline_bin_count']} "
                f"in {os.path.join(out_dir, stats['pathline_bin_dir'])}")
    if stats["segmentation_bin_count"]:
        logging.info(
            f"[totals] SEGMENTATION BIN files = "
            f"{stats['segmentation_bin_count']} in "
            f"{os.path.join(out_dir, stats['segmentation_bin_dir'])}")
    if stats["sample_meta_count"]:
        logging.info(
            f"[totals] SAMPLE META files = {stats['sample_meta_count']} "
            f"in {os.path.join(out_dir, stats['sample_meta_dir'])}")
    logging.info(
        f"[totals] UNSTEADY NPZ fields = {stats['unsteady_npz_count']} "
        f"in {stats['unsteady_npz_shard_count']} shard(s)")
    if stats["lic_dir"]:
        logging.info(
            f"[totals] LIC images = {stats['steady_lic_count']} steady + "
            f"{stats['deformed_lic_count']} deformed in "
            f"{os.path.join(out_dir, stats['lic_dir'])}")
    logging.info("=" * 78)

    if bool(cfg.output.get("save_plots", True)) and not args.no_plots:
        save_plots(out_dir, specs, grid, steady_fields, steady_labels, source, ocfg, evals, cfg)
    logging.info("[main] done.")


if __name__ == "__main__":
    main()
