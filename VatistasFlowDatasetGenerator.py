"""
Generate VortexTransformer training data: synthesize steady Vatistas fields,
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
`flowGenerator.cpp::generateMixUnsteadyFieldPathline`: selectable cross-grid,
four-quadrant random, IVD-importance, or IVD-adaptive-mesh pathline
seeding, RK4 integration, per-point
[x,y,t,IVD,distance,u,v] features, a seed
vortex label, a binary segmentation mask, observer parameters, and field meta.
In addition to NPZ shards, each pathline tensor is written as a
Cereal-compatible ``velocity_mix/sample_*_pathline.bin`` file.
When LIC output is enabled, an optional paired ``*_mesh.png`` image overlays
the actual pathline seeds and their Delaunay unstructured mesh on every LIC.

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
    s_mat = S_MATRICES.to(device=params.device, dtype=params.dtype)[shape_idx]
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


def _cfg_value(section, name, default):
    """Read an optional EasyConfig/dict value without making it mandatory."""
    if section is None:
        return default
    if hasattr(section, "get"):
        return section.get(name, default)
    return getattr(section, name, default)


def _geometry_settings(cfg):
    """Geometry constraints in the normalized patch frame [-1,1]^2."""
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
    """Velocity MSE plus physical vortex-core geometry penalties."""
    settings = _geometry_settings(cfg)
    data_mse = per_patch_mse(pred, target)
    axis_x, axis_y, extent_x, extent_y = _physical_core_geometry(params)
    min_axis, max_axis = settings["min_core_axis"], settings["max_core_axis"]
    size_penalty = (
        torch.relu(min_axis - axis_x) ** 2
        + torch.relu(min_axis - axis_y) ** 2
        + torch.relu(axis_x - max_axis) ** 2
        + torch.relu(axis_y - max_axis) ** 2
    ).mean(dim=1)
    tx, ty = params[..., TX], params[..., TY]
    containment_penalty = (
        torch.relu(tx.abs() + extent_x - settings["domain_limit"]) ** 2
        + torch.relu(ty.abs() + extent_y - settings["domain_limit"]) ** 2
    ).mean(dim=1)
    center_penalty = (
        torch.relu(tx.abs() - settings["center_limit"]) ** 2
        + torch.relu(ty.abs() - settings["center_limit"]) ** 2
    ).mean(dim=1)

    spacing_penalty = torch.zeros_like(data_mse)
    profile_count = params.shape[1]
    if profile_count > 1:
        outer_radius = torch.maximum(axis_x, axis_y)
        pair_penalties = []
        for first in range(profile_count):
            for second in range(first + 1, profile_count):
                distance = torch.sqrt(
                    (tx[:, first] - tx[:, second]) ** 2
                    + (ty[:, first] - ty[:, second]) ** 2 + 1e-12)
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
        + settings["spacing_weight"] * spacing_penalty)
    return score, data_mse


def _swirling_strength_peak(velocity, margin=3):
    height, width = velocity.shape[:2]
    dy, dx = 2.0 / max(1, height - 1), 2.0 / max(1, width - 1)
    du_dy, du_dx = np.gradient(velocity[..., 0], dy, dx)
    dv_dy, dv_dx = np.gradient(velocity[..., 1], dy, dx)
    trace = du_dx + dv_dy
    determinant = du_dx * dv_dy - du_dy * dv_dx
    swirl = 0.5 * np.sqrt(np.maximum(4.0 * determinant - trace * trace, 0.0))
    margin = max(0, int(margin))
    if margin > 0 and height > 2 * margin and width > 2 * margin:
        swirl = swirl[margin:-margin, margin:-margin]
    return float(np.max(swirl)) if swirl.size else 0.0


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
    remove_mean = bool(dcfg.get("remove_mean_velocity", True))
    min_residual_fraction = float(dcfg.get("min_residual_fraction", 0.05))
    min_swirl_strength = float(dcfg.get("min_swirl_strength", 0.05))
    swirl_margin = int(dcfg.get("swirl_margin", 3))
    field_specs = list(dcfg.fields)
    names = [str(f["name"]) for f in field_specs]
    logging.info(f"[patches] loading real flows: {names}")
    fields = load_UnsteadyVectorFields_general(dcfg.dat_dir, names)

    patches, scales, meta = [], [], []
    skipped_low_speed = skipped_low_residual = skipped_no_swirl = 0
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
                    patch = slice_t[y0:y0 + ps, x0:x0 + ps, :].copy()
                    raw_rms = float(np.sqrt(np.mean(
                        patch[..., 0] ** 2 + patch[..., 1] ** 2)))
                    if raw_rms < min_rms:
                        skipped_low_speed += 1
                        continue

                    background = patch.mean(axis=(0, 1), keepdims=True)
                    working = patch - background if remove_mean else patch
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
    logging.info(
        "[patches] preprocessing remove_mean=%s; skipped low-speed=%d, "
        "low-residual=%d, no-swirl=%d (min residual fraction %.3f, "
        "min swirl %.3f).",
        remove_mean, skipped_low_speed, skipped_low_residual, skipped_no_swirl,
        min_residual_fraction, min_swirl_strength)
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
    seed_margin = int(cfg.optim.get("seed_margin", 3))
    if seed_margin > 0 and height > 2 * seed_margin and width > 2 * seed_margin:
        oabs[:, :seed_margin, :] = 0
        oabs[:, -seed_margin:, :] = 0
        oabs[:, :, :seed_margin] = 0
        oabs[:, :, -seed_margin:] = 0
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
        cur_score, cur_mse = fit_objective(
            vatistas_mixture_velocity(coords, cur, cur_shape, bounds),
            target, cur, cfg)
        best, best_shape = cur.clone(), cur_shape.clone()
        best_score, best_mse = cur_score.clone(), cur_mse.clone()
        temp = t0
        for it in range(iters):
            frac = temp / t0
            prop = _project(cur + torch.randn_like(cur) * (sigma.view(1, 1, 7) * frac), cur_shape, bounds)
            prop_shape = cur_shape
            if flip_p > 0.0:
                flip = torch.rand(p_count, n_profiles, device=device) < flip_p
                rnd_shape = candidates[torch.randint(0, len(candidates), (p_count, n_profiles), device=device)]
                prop_shape = torch.where(flip, rnd_shape, cur_shape)
            prop_score, prop_mse = fit_objective(
                vatistas_mixture_velocity(coords, prop, prop_shape, bounds),
                target, prop, cfg)
            delta = prop_score - cur_score
            accept = (delta < 0) | (torch.rand(p_count, device=device) < torch.exp(-delta / max(temp, 1e-8)))
            cur = torch.where(accept.view(p_count, 1, 1), prop, cur)
            cur_shape = torch.where(accept.view(p_count, 1), prop_shape, cur_shape)
            cur_score = torch.where(accept, prop_score, cur_score)
            cur_mse = torch.where(accept, prop_mse, cur_mse)
            improved = cur_score < best_score
            best = torch.where(improved.view(p_count, 1, 1), cur, best)
            best_shape = torch.where(improved.view(p_count, 1), cur_shape, best_shape)
            best_score = torch.where(improved, cur_score, best_score)
            best_mse = torch.where(improved, cur_mse, best_mse)
            temp *= cooling
            if (it + 1) % 50 == 0 or it == iters - 1:
                logging.info(
                    f"[SA] iter {it+1}/{iters}  T={temp:.4g}  "
                    f"mean score={best_score.mean().item():.6f}  "
                    f"mean mse={best_mse.mean().item():.6f}")
    return best, best_shape, best_score


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
    iters = int(cfg.optim.get("lbfgs_iters", 0))
    if iters <= 0:
        with torch.no_grad():
            score, mse = fit_objective(
                vatistas_mixture_velocity(coords, params, shape_idx, bounds),
                target, params, cfg)
            return params, mse, score
    p = params.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([p], max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)

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
    orig_score, orig_mse = fit_objective(
        vatistas_mixture_velocity(coords, params, shape_idx, bounds),
        target, params, cfg)
    bad = (final_score > orig_score) | ~torch.isfinite(final_score)
    if bad.any():
        final = torch.where(bad.view(-1, 1, 1), params, final)
        final_score = torch.where(bad, orig_score, final_score)
        final_mse = torch.where(bad, orig_mse, final_mse)
    return final, final_mse, final_score


def fit_patches(coords, target, cfg, bounds, n_profiles, device):
    p_count = target.shape[0]
    n_restarts = int(cfg.optim.get("n_restarts", 1))
    seeds = None
    if str(cfg.optim.get("init_mode", "random")) == "datadriven":
        seeds = datadriven_seeds(target, coords, cfg, n_profiles)
    best_params = best_shape = best_loss = best_score = None
    for r in range(n_restarts):
        params, shape_idx = init_params(p_count, n_profiles, cfg, bounds, device, restart=r, seeds=seeds)
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
            take = score < best_score
            best_params = torch.where(take.view(p_count, 1, 1), params, best_params)
            best_shape = torch.where(take.view(p_count, 1), shape_idx, best_shape)
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


def valid_profile_mask(params, shape_idx, fit_loss, contribution, cfg, bounds):
    """Keep only profiles that can produce clean in-domain labels."""
    settings = _geometry_settings(cfg)
    axis_x, axis_y, extent_x, extent_y = _physical_core_geometry(params)
    tx, ty = params[..., TX], params[..., TY]
    contribution_ok = contribution >= float(cfg.generate.min_contribution)
    size_ok = (
        (axis_x >= settings["min_core_axis"])
        & (axis_x <= settings["max_core_axis"])
        & (axis_y >= settings["min_core_axis"])
        & (axis_y <= settings["max_core_axis"]))
    contained = (
        (tx.abs() + extent_x <= settings["domain_limit"])
        & (ty.abs() + extent_y <= settings["domain_limit"])
        & (tx.abs() <= settings["center_limit"])
        & (ty.abs() <= settings["center_limit"]))
    fit_ok = fit_loss.view(-1, 1) <= settings["max_fit_mse"]
    geometry = _cfg_value(cfg.model, "geometry", {}) or {}
    if bool(_cfg_value(geometry, "exclude_bound_hits", True)):
        eps = 1e-4
        not_clamped = (
            (params[..., RC] > bounds["rc"][0] + eps)
            & (params[..., RC] < bounds["rc"][1] - eps)
            & (params[..., SX] > bounds["s"][0] + eps)
            & (params[..., SX] < bounds["s"][1] - eps)
            & (params[..., SY] > bounds["s"][0] + eps)
            & (params[..., SY] < bounds["s"][1] - eps)
            & (tx.abs() < bounds["t"] - eps)
            & (ty.abs() < bounds["t"] - eps))
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
    logging.info(
        "[fit-health] valid profiles %d/%d; contribution=%d, size=%d, "
        "contained=%d, fit=%d, not-clamped=%d; spaced rows=%d/%d, "
        "low-interaction rows=%d/%d.",
        int(valid.sum()), int(valid.numel()), int(contribution_ok.sum()),
        int(size_ok.sum()), int(contained.sum()),
        int(fit_ok.expand_as(valid).sum()), int(not_clamped.sum()),
        int(row_spacing_ok.sum()), len(row_spacing_ok),
        int(row_interaction_ok.sum()), len(row_interaction_ok))
    if not valid.any():
        raise RuntimeError(
            "No fitted profile passed the geometry/quality checks. Inspect the "
            "fit-health log before relaxing model.geometry.")
    return valid


def fit_distribution(params, shape_idx, active_mask):
    prm = params.reshape(-1, 7).cpu().numpy()
    shp = shape_idx.reshape(-1).cpu().numpy()
    keep = active_mask.reshape(-1).cpu().numpy().astype(bool)
    prm, shp = prm[keep], shp[keep]
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
            circular_std = float(np.sqrt(max(
                0.0, -2.0 * np.log(max(resultant, 1e-12)))))
            dist["params"][name] = {
                "mean": c_mean, "std": circular_std,
                "min": float(col.min()), "max": float(col.max())}
        else:
            dist["params"][name] = {"mean": float(col.mean()), "std": float(col.std()), "min": float(col.min()), "max": float(col.max())}
    shapes, counts = np.unique(shp, return_counts=True)
    probs = counts / counts.sum()
    dist["shape_distribution"] = {SHAPE_LABELS[int(s)]: float(pr) for s, pr in zip(shapes, probs)}
    dist["shape_index_probs"] = {int(s): float(pr) for s, pr in zip(shapes, probs)}
    dist["empirical_params"] = prm.astype(np.float32).tolist()
    dist["empirical_shapes"] = shp.astype(np.int64).tolist()
    return dist


def _sample_gaussian(spec, rng):
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
        and abs(tx) <= bounds["t"] and abs(ty) <= bounds["t"]
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


def _profiles_have_low_interaction(
        profiles, cfg, bounds, max_ratio=None, max_shift=None):
    """Keep analytic core labels aligned with centers in the summed field."""
    if len(profiles) < 2:
        return True
    settings = _geometry_settings(cfg)
    if max_ratio is None:
        max_ratio = settings["max_interaction_ratio"]
    if max_shift is None:
        max_shift = settings["max_center_shift_fraction"]
    for profile_index, (_p, shape) in enumerate(profiles):
        if int(shape) not in (1, 2):
            continue
        ratio, shift = _profile_interaction_metrics(
            profile_index, profiles, cfg, bounds)
        if (
            ratio > float(max_ratio)
            or shift > float(max_shift)
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
    """
        Sample `num_fields` parameter sets from the fitted distribution and render
        each as a steady velocity field on [-1,1]^2 (Eq. 10).
    """
    gcfg = cfg.generate
    num_fields = int(gcfg.num_fields)
    grid_size = int(gcfg.grid)
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
    empirical_std = (
        np.std(empirical_params, axis=0).astype(np.float32)
        if use_empirical else None)

    def sample_profile():
        if use_empirical:
            selected = int(rng.integers(0, len(empirical_params)))
            p = empirical_params[selected].copy()
            if empirical_jitter > 0.0:
                p += rng.normal(
                    0.0, empirical_jitter * np.maximum(empirical_std, 1e-6),
                    size=7).astype(np.float32)
            shape = int(empirical_shapes[selected])
        else:
            p = np.empty(7, np.float32)
            for i, name in enumerate(PARAM_NAMES):
                p[i] = _sample_gaussian(pspec[name], rng)
            shape = int(rng.choice(shape_idx_list, p=shape_probs))
        p[TH] = np.arctan2(np.sin(p[TH]), np.cos(p[TH]))
        return p, shape

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
                        and not _profiles_are_spaced(
                            p, candidate_profiles, cfg)
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
    coords = make_local_grid(grid_size, device)
    params_t = torch.from_numpy(gen_params).to(device)
    shapes_t = torch.from_numpy(gen_shapes).to(device)
    mask_t = torch.from_numpy(gen_mask).to(device)
    with torch.no_grad():
        fields = vatistas_mixture_velocity(coords, params_t, shapes_t, bounds, profile_mask=mask_t)
    logging.info(
        f"[generate] synthesized {num_fields} geometry-valid steady fields on "
        f"[-1,1]^2 at {grid_size}x{grid_size} "
        f"(vortices per field in [{m_min},{m_max}], "
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
        meta_background_u=np.array(
            [m.get("background_u", 0.0) for m in meta], np.float32),
        meta_background_v=np.array(
            [m.get("background_v", 0.0) for m in meta], np.float32),
        meta_raw_rms=np.array(
            [m.get("raw_rms", 1.0) for m in meta], np.float32),
        meta_residual_rms=np.array(
            [m.get("residual_rms", 1.0) for m in meta], np.float32),
        meta_swirl_peak=np.array(
            [m.get("swirl_peak", 0.0) for m in meta], np.float32),
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
        "contribution": (
            d["contribution"].astype(np.float32)
            if "contribution" in d.files else None),
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
    active = valid_profile_mask(
        params, shape_idx, fit_loss, contrib, cfg, bounds)
    dist = fit_distribution(params, shape_idx, active)
    fit_dir = cfg.output.get("out_dir") or os.path.join(cfg.get("cache_dir", "./outputs/"), "vatistas_fit")
    gen = generate_steady_fields(dist, cfg, bounds, device)
    save_outputs(fit_dir, params, shape_idx, fit_loss, contrib, active, meta, dist, gen)
    fitted = {"params": params.cpu().numpy().astype(np.float32),
              "shapes": shape_idx.cpu().numpy().astype(np.int64),
              "mask": active.cpu().numpy().astype(np.float32),
              "contribution": contrib.cpu().numpy().astype(np.float32),
              "fit_mse": fit_loss.cpu().numpy().astype(np.float32)}
    return dist, fitted


# ===========================================================================
# 1. Steady field pool:  random (distribution) + fitted (real patches) + Killing
# ===========================================================================
def _build_clean_fitted_fields(fitted, cfg, bounds, num_fields):
    """Build 1/2-profile fitted-derived fields with clean analytic labels.

    None of the current cache's original two-profile rows satisfies all of the
    clean-label constraints simultaneously.  This function therefore builds a
    pool of individually valid fitted profiles, samples one or two profiles per
    field just like ``generate_steady_fields``, and only uses compatible pairs
    for two-profile fields.  All requested fields are retained without forcing
    either a single-vortex-only or a two-vortex-only dataset.
    """
    source_params = np.asarray(fitted["params"], dtype=np.float32)
    source_shapes = np.asarray(fitted["shapes"], dtype=np.int64)
    contribution = fitted.get("contribution")
    if contribution is None:
        contribution = np.ones(source_shapes.shape, dtype=np.float32)
    else:
        contribution = np.asarray(contribution, dtype=np.float32)
    min_contribution = float(
        cfg.generate.get("min_contribution", 0.05))
    fit_mse = np.asarray(fitted["fit_mse"], dtype=np.float32)
    max_fit_mse = _geometry_settings(cfg)["max_fit_mse"]

    flat_params = source_params.reshape(-1, source_params.shape[-1])
    flat_shapes = source_shapes.reshape(-1)
    flat_contribution = contribution.reshape(-1)
    flat_rows = np.repeat(
        np.arange(source_params.shape[0]), source_params.shape[1])
    valid = np.asarray([
        flat_contribution[index] >= min_contribution
        and int(flat_shapes[index]) in CENTER_SHAPES
        and fit_mse[flat_rows[index]] <= max_fit_mse
        and _numpy_profile_geometry_valid(
            flat_params[index], cfg, bounds)
        for index in range(flat_params.shape[0])
    ], dtype=bool)
    pool_params = flat_params[valid]
    pool_shapes = flat_shapes[valid]
    if pool_params.shape[0] < 2:
        raise RuntimeError(
            "Fewer than two individually valid fitted profiles are available.")

    fcfg = cfg.get("fitted", {}) or {}
    min_vortices = int(fcfg.get(
        "min_vortices", cfg.generate.get("min_vortices", 1)))
    max_vortices = int(fcfg.get(
        "max_vortices", cfg.generate.get("max_vortices", 2)))
    if not (1 <= min_vortices <= max_vortices <= 2):
        raise ValueError(
            "fitted min_vortices/max_vortices must satisfy "
            "1 <= min <= max <= 2.")
    geometry_settings = _geometry_settings(cfg)
    pair_max_ratio = float(fcfg.get(
        "max_interaction_ratio",
        geometry_settings["max_interaction_ratio"]))
    pair_max_shift = float(fcfg.get(
        "max_center_shift_fraction",
        geometry_settings["max_center_shift_fraction"]))

    compatible_pairs = []
    for first in range(pool_params.shape[0]):
        for second in range(first + 1, pool_params.shape[0]):
            profiles = [
                (pool_params[first], int(pool_shapes[first])),
                (pool_params[second], int(pool_shapes[second])),
            ]
            if not _profiles_are_spaced(
                    pool_params[second], profiles[:1], cfg):
                continue
            if not _profiles_have_low_interaction(
                    profiles, cfg, bounds,
                    max_ratio=pair_max_ratio,
                    max_shift=pair_max_shift):
                continue
            compatible_pairs.append((first, second))
    if not compatible_pairs:
        raise RuntimeError(
            "No two individually valid fitted profiles form a label-safe pair.")

    rng = np.random.default_rng(int(cfg.get("seed", 0)) + 37)
    jitter = float(fcfg.get(
        "profile_jitter_std",
        fcfg.get(
            "pair_jitter_std",
            cfg.generate.get("empirical_jitter", 0.02))))
    jitter_attempts = max(1, int(fcfg.get(
        "profile_jitter_attempts",
        fcfg.get("pair_jitter_attempts", 16))))
    pool_std = np.std(pool_params, axis=0).astype(np.float32)
    params = np.zeros((int(num_fields), 2, 7), dtype=np.float32)
    shapes = np.zeros((int(num_fields), 2), dtype=np.int64)
    mask = np.zeros((int(num_fields), 2), dtype=np.float32)
    jitter_fallbacks = 0
    vortex_counts = np.zeros(3, dtype=np.int64)

    for field_index in range(int(num_fields)):
        vortex_count = int(rng.integers(
            min_vortices, max_vortices + 1))
        vortex_counts[vortex_count] += 1
        if vortex_count == 1:
            profile_index = int(rng.integers(0, pool_params.shape[0]))
            base = pool_params[profile_index:profile_index + 1].copy()
            field_shapes = pool_shapes[
                profile_index:profile_index + 1].copy()
        else:
            pair_index = int(rng.integers(0, len(compatible_pairs)))
            first, second = compatible_pairs[pair_index]
            base = np.stack(
                [pool_params[first], pool_params[second]]).astype(np.float32)
            field_shapes = np.asarray(
                [pool_shapes[first], pool_shapes[second]], dtype=np.int64)
        accepted = None
        for _ in range(jitter_attempts):
            candidate = base + rng.normal(
                0.0,
                jitter * np.maximum(pool_std, 1e-6),
                size=base.shape,
            ).astype(np.float32)
            candidate[:, TH] = np.arctan2(
                np.sin(candidate[:, TH]), np.cos(candidate[:, TH]))
            profiles = [
                (candidate[index], int(field_shapes[index]))
                for index in range(vortex_count)
            ]
            if not all(
                    _numpy_profile_geometry_valid(profile, cfg, bounds)
                    for profile, _shape in profiles):
                continue
            if vortex_count == 2:
                if not _profiles_are_spaced(
                        candidate[1], profiles[:1], cfg):
                    continue
                if not _profiles_have_low_interaction(
                        profiles, cfg, bounds,
                        max_ratio=pair_max_ratio,
                        max_shift=pair_max_shift):
                    continue
            accepted = candidate
            break
        if accepted is None:
            accepted = base.copy()
            jitter_fallbacks += 1

        # Quarter turns preserve the square domain and every geometric/
        # interaction constraint while augmenting the fitted pair distribution.
        quarter_turn = int(rng.integers(0, 4))
        if quarter_turn:
            angle = quarter_turn * (0.5 * np.pi)
            cos_t, sin_t = np.cos(angle), np.sin(angle)
            rotation = np.asarray(
                [[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
            accepted[:, [TX, TY]] = accepted[:, [TX, TY]] @ rotation.T
            accepted[:, TH] = np.arctan2(
                np.sin(accepted[:, TH] + angle),
                np.cos(accepted[:, TH] + angle))
        if vortex_count == 2 and rng.random() < 0.5:
            accepted = accepted[::-1]
            field_shapes = field_shapes[::-1]
        params[field_index, :vortex_count] = accepted
        shapes[field_index, :vortex_count] = field_shapes
        mask[field_index, :vortex_count] = 1.0

    logging.info(
        "[steady] built %d clean fitted-derived fields: %d single-profile + "
        "%d two-profile, from %d valid fitted profiles and %d compatible "
        "pairs (%d jitter fallbacks; pair limits ratio<=%.3f, shift<=%.3f).",
        params.shape[0], int(vortex_counts[1]), int(vortex_counts[2]),
        pool_params.shape[0], len(compatible_pairs),
        jitter_fallbacks, pair_max_ratio, pair_max_shift)
    return params, shapes, mask


def _subsample_fitted(fitted, cfg, bounds):
    fcfg = cfg.get("fitted", {}) or {}
    f_params = fitted["params"]
    f_shapes = fitted["shapes"]
    row_indices = np.arange(f_params.shape[0])
    want = fcfg.get("num_fields", None)
    want = None if want in (None, "null") else int(want)

    # The strict fit-time geometry mask remains useful for learning the random
    # distribution, but fitted output needs its own explicit policy.
    profile_policy = str(
        fcfg.get("profile_policy", "clean_random_1_or_2")).strip().lower()
    if profile_policy == "clean_random_1_or_2":
        target_count = f_params.shape[0] if want is None else want
        return _build_clean_fitted_fields(
            fitted, cfg, bounds, target_count)
    elif profile_policy == "geometry_valid":
        f_mask = fitted["mask"].copy()
        valid_rows = np.any(f_mask > 0.0, axis=1)
        if not np.all(valid_rows):
            logging.info(
                "[steady] dropped %d fitted patches with no geometry-valid "
                "profile (fitted.profile_policy=geometry_valid).",
                int((~valid_rows).sum()))
            row_indices = row_indices[valid_rows]
            f_params = f_params[valid_rows]
            f_shapes = f_shapes[valid_rows]
            f_mask = f_mask[valid_rows]
    elif profile_policy == "contribution":
        contribution = fitted.get("contribution")
        if contribution is None:
            logging.warning(
                "[steady] cached contribution data is unavailable; enabling "
                "all fitted profiles to preserve every fitted patch.")
            f_mask = np.ones(f_shapes.shape, dtype=np.float32)
        else:
            min_contribution = float(
                cfg.generate.get("min_contribution", 0.05))
            f_mask = (contribution >= min_contribution).astype(np.float32)
            empty_rows = ~np.any(f_mask > 0.0, axis=1)
            if np.any(empty_rows):
                # Preserve the fitted-patch count for unusual caches by keeping
                # the strongest fitted component in each otherwise-empty row.
                strongest = np.argmax(contribution[empty_rows], axis=1)
                f_mask[empty_rows] = 0.0
                f_mask[np.flatnonzero(empty_rows), strongest] = 1.0
            logging.info(
                "[steady] retained all %d fitted patches using the profile "
                "contribution mask (threshold %.4g).",
                f_params.shape[0], min_contribution)
    else:
        raise ValueError(
            "fitted.profile_policy must be one of: clean_random_1_or_2, "
            f"geometry_valid, contribution; got {profile_policy!r}.")

    if want is not None and 0 < want < f_params.shape[0]:
        if bool(fcfg.get("prefer_low_mse", True)):
            order = np.argsort(fitted["fit_mse"][row_indices])[:want]
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
    f_params, f_shapes, f_mask = _subsample_fitted(
        fitted, cfg, bounds)
    for i in range(f_params.shape[0]):
        specs.append({"type": "vatistas", "params": f_params[i], "shapes": f_shapes[i],
                      "mask": f_mask[i], "source": SRC_FITTED})



    # (c) manually-added Killing fields (hard negatives, empty label)
    kabc = make_killing_steady(cfg)
    for i in range(kabc.shape[0]):
        specs.append({"type": "killing", "abc": kabc[i], "source": SRC_KILLING})

    counts = {SRC_RANDOM: gen["params"].shape[0], SRC_FITTED: f_params.shape[0], SRC_KILLING: kabc.shape[0]}
    # counts = {SRC_FITTED: f_params.shape[0]}
    logging.info(f"[steady] pool = {len(specs)} fields  "
                 f"(random {counts[SRC_RANDOM]} + fitted {counts[SRC_FITTED]} + killing {counts[SRC_KILLING]}).")
                 #f"(fitted {counts[SRC_FITTED]}).")

    return specs, counts


# ---- per-field steady evaluator + label (analytic, works at arbitrary points) ----
def make_steady_eval(spec, bounds, device):
    """Return NumPy velocity/label evaluators and attach a differentiable velocity evaluator.

    ``vel.torch_eval`` accepts a torch tensor shaped ``[..., 2]`` plus an
    optional, ignored time argument.  This gives steady and unsteady
    differentiable evaluators one compatible interface while preserving the
    coordinate computation graph.  The NumPy callable remains the public
    evaluator used by field rendering and pathline integration.
    """
    if spec["type"] == "killing":
        a, b, c = [float(x) for x in spec["abc"]]

        def vel(coords_np):
            x, y = coords_np[..., 0], coords_np[..., 1]
            return np.stack([a - c * y, b + c * x], axis=-1).astype(np.float32)

        def vel_torch(coords, _time=None):
            x, y = coords[..., 0], coords[..., 1]
            return torch.stack([a - c * y, b + c * x], dim=-1)

        vel_torch.device = torch.device(device)
        vel_torch.dtype = torch.float32
        vel.torch_eval = vel_torch

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

    def vel_torch(coords, _time=None):
        original_shape = coords.shape
        if original_shape[-1] != 2:
            raise ValueError(
                f"Expected differentiable coordinates shaped [...,2], got {original_shape}.")
        coords = coords.to(device=device, dtype=pt.dtype)
        coords_grid = coords.reshape(-1, 1, 2)
        velocity = vatistas_mixture_velocity(
            coords_grid, pt, st, bounds, profile_mask=mt)[0, :, 0]
        return velocity.reshape(original_shape)

    vel_torch.device = torch.device(device)
    vel_torch.dtype = pt.dtype
    vel.torch_eval = vel_torch

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
        sx = float(np.clip(sx, s_lo, s_hi))
        sy = float(np.clip(sy, s_lo, s_hi))
        tx, ty = params[m, TX], params[m, TY]
        cos, sin = np.cos(th), np.sin(th)
        inv_det = 1.0 / (sx * sy)
        dx = coords_np[..., 0] - tx
        dy = coords_np[..., 1] - ty
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
    T = int(ocfg["timesteps"])
    G = coords_np.shape[0]
    tmin, tmax = float(ocfg["tmin"]), float(ocfg["tmax"])
    center = np.asarray(ocfg["center"], np.float64)
    dt = (tmax - tmin) / (T - 1)

    path = integrate_observer_pathline(abc_func, ocfg["start_pos"], center, tmin, tmax, T)
    Os = path[0]

    finv = np.zeros((T, G, G, 2), np.float64)
    Q_all = np.zeros((T, 2, 2))
    Qdot_all = np.zeros((T, 2, 2))
    Tdot_all = np.zeros((T, 2))
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

    differentiable_steady_eval = getattr(vel_eval, "torch_eval", None)
    if differentiable_steady_eval is not None:
        evaluate.torch_eval = _make_tobias_velocity_evaluator_torch(
            differentiable_steady_eval,
            dt,
            q_all,
            c_all,
            velocities,
            angular_velocities,
            tmin,
            timesteps,
        )
    return evaluate


def _make_tobias_velocity_evaluator_torch(
        steady_eval, dt, q_all, c_all, velocities, angular_velocities,
        tmin, timesteps):
    """Torch counterpart of the Tobias evaluator, differentiable in space."""
    device = torch.device(getattr(steady_eval, "device", "cpu"))
    dtype = getattr(steady_eval, "dtype", torch.float32)
    q_all_t = torch.as_tensor(q_all, device=device, dtype=dtype)
    c_all_t = torch.as_tensor(c_all, device=device, dtype=dtype)
    velocities_t = torch.as_tensor(velocities, device=device, dtype=dtype)
    angular_velocities_t = torch.as_tensor(
        angular_velocities, device=device, dtype=dtype)

    def evaluate(points, time):
        if not torch.is_tensor(points):
            points = torch.as_tensor(points, device=device, dtype=dtype)
        else:
            points = points.to(device=device, dtype=dtype)
        original_shape = points.shape
        if original_shape[-1] != 2:
            raise ValueError(
                f"Expected differentiable points shaped [...,2], got {original_shape}.")

        floating = (float(time) - float(tmin)) / dt
        lower = int(np.clip(math.floor(floating), 0, timesteps - 1))
        upper = int(np.clip(math.floor(floating) + 1, 0, timesteps - 1))
        ratio = floating - lower
        q_t = q_all_t[lower] * (1.0 - ratio) + q_all_t[upper] * ratio
        c_t = c_all_t[lower] * (1.0 - ratio) + c_all_t[upper] * ratio
        translation_velocity = (
            velocities_t[lower] * (1.0 - ratio)
            + velocities_t[upper] * ratio
        )
        omega = (
            angular_velocities_t[lower] * (1.0 - ratio)
            + angular_velocities_t[upper] * ratio
        )

        flat = points.reshape(-1, 2)
        finv = (flat - c_t) @ q_t
        steady_velocity = steady_eval(finv)
        zero = torch.zeros((), device=device, dtype=dtype)
        spin = torch.stack([
            torch.stack([zero, omega]),
            torch.stack([-omega, zero]),
        ])
        q_dot = q_t @ spin
        result = (
            steady_velocity @ q_t.T
            + finv @ q_dot.T
            + translation_velocity
        )
        return result.reshape(original_shape)

    evaluate.device = device
    evaluate.dtype = dtype
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


GRID_CROSS_SEEDING = "GRID_CROSS_SEEDING"
RECTANGULAR_CLUSTER4_RANDOM = "RECTANGULAR_CLUSTER4_RANDOM"
VORTICITY_IMPORTANCE_SEEDING = "VORTICITY_IMPORTANCE_SEEDING"
IVD_IMPORTANCE_SEEDING = "IVD_IMPORTANCE_SEEDING"
FEATURE_ADAPTIVE_MESH_SEEDING = "FEATURE_ADAPTIVE_MESH_SEEDING"
_PATHLINE_SAMPLING_METHODS = {
    GRID_CROSS_SEEDING,
    RECTANGULAR_CLUSTER4_RANDOM,
    IVD_IMPORTANCE_SEEDING,
    FEATURE_ADAPTIVE_MESH_SEEDING,
}


def _normalize_pathline_sampling(sampling):
    method = str(sampling).strip().upper()
    # Older configurations now use IVD under the legacy sampling name too.
    if method == VORTICITY_IMPORTANCE_SEEDING:
        method = IVD_IMPORTANCE_SEEDING
    if method not in _PATHLINE_SAMPLING_METHODS:
        supported = ", ".join(sorted(_PATHLINE_SAMPLING_METHODS))
        raise ValueError(
            f"Unsupported pathline sampling method {sampling!r}; "
            f"expected one of: {supported}.")
    return method


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


def _rectangular4_cluster_seedings(
        samples_per_cluster, rng, domain_min=(-1.0, -1.0),
        domain_max=(1.0, 1.0)):
    """Port of C++ ``RecTangular4ClusterSampling``.

    Split the rectangular domain into four quadrants and sample the same
    number of uniformly distributed points in each quadrant.  Results retain
    the C++ cluster order: lower-left, upper-left, lower-right, upper-right.
    """
    samples_per_cluster = int(samples_per_cluster)
    if samples_per_cluster < 1:
        raise ValueError("samples_per_cluster must be at least 1.")

    domain_min = np.asarray(domain_min, dtype=np.float64)
    domain_max = np.asarray(domain_max, dtype=np.float64)
    if domain_min.shape != (2,) or domain_max.shape != (2,):
        raise ValueError("domain_min and domain_max must be two-dimensional.")
    if np.any(domain_max <= domain_min):
        raise ValueError("domain_max must be greater than domain_min.")

    domain_range = domain_max - domain_min
    center_fractions = np.asarray([
        (0.25, 0.25),
        (0.25, 0.75),
        (0.75, 0.25),
        (0.75, 0.75),
    ], dtype=np.float64)
    domain_centers = domain_min + center_fractions * domain_range

    clusters = []
    half_cluster_range = 0.25 * domain_range
    for domain_center in domain_centers:
        cluster_min = np.maximum(
            domain_center - half_cluster_range, domain_min)
        cluster_max = np.minimum(
            domain_center + half_cluster_range, domain_max)
        clusters.append(rng.uniform(
            cluster_min, cluster_max, size=(samples_per_cluster, 2)))
    return np.concatenate(clusters, axis=0).astype(np.float64, copy=False)


def _common_existing_seed_count(k_lines):
    """Seed count shared by both legacy samplers for an even ``count_k``."""
    k_lines = int(k_lines)
    if k_lines < 4:
        raise ValueError("pathline.count_k must be at least 4.")
    grid_cross_count = (k_lines // 2) ** 2 * 4
    rectangular_count = int(k_lines * k_lines * 0.25) * 4
    if grid_cross_count != rectangular_count:
        raise ValueError(
            "Feature-based pathline seeding requires an even pathline.count_k "
            "so its seed count can match both legacy sampling methods; got "
            f"{k_lines} (grid-cross={grid_cross_count}, "
            f"rectangular={rectangular_count}).")
    return grid_cross_count


def _feature_mesh_candidates(resolution, boundary_margin, rng):
    """Return a jittered Cartesian background mesh over ``[-1, 1]^2``.

    Cell-centred vertices avoid placing a seed on the strict integration
    boundary.  The small jitter removes axis-aligned CVT artefacts while the
    fixed background topology keeps the result deterministic for a given RNG.
    """
    resolution = int(resolution)
    boundary_margin = float(boundary_margin)
    if resolution < 8:
        raise ValueError("feature_mesh_resolution must be at least 8.")
    if not 0.0 <= boundary_margin < 1.0:
        raise ValueError("feature_boundary_margin must be in [0, 1).")

    lower = -1.0 + boundary_margin
    upper = 1.0 - boundary_margin
    spacing = (upper - lower) / resolution
    coordinates = lower + (np.arange(resolution) + 0.5) * spacing
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    candidates = np.column_stack([xx.ravel(), yy.ravel()])
    candidates += rng.uniform(
        -0.20 * spacing, 0.20 * spacing, size=candidates.shape)
    return np.clip(candidates, lower, upper).astype(np.float64, copy=False)


def _weighted_cvt_mesh_points(
        candidates, target_size, point_count, rng, iterations):
    """Optimize adaptive-mesh generators with a discrete weighted CVT.

    ``target_size`` is the feature-derived local edge-length monitor.  Initial
    generators are selected by density-weighted farthest-point insertion and
    are then relaxed to weighted Voronoi centroids.  Snapping every centroid
    back to its own background-mesh cell keeps points inside their requested
    high- or low-IVD region and guarantees unique output points.
    """
    candidates = np.asarray(candidates, dtype=np.float64)
    target_size = np.asarray(target_size, dtype=np.float64)
    point_count = int(point_count)
    iterations = int(iterations)
    if point_count == 0:
        return np.empty((0, 2), dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != 2:
        raise ValueError("Adaptive-mesh candidates must be shaped [N, 2].")
    if target_size.shape != (len(candidates),):
        raise ValueError("Adaptive target_size must contain one value per candidate.")
    if point_count > len(candidates):
        raise ValueError(
            f"Adaptive mesh has only {len(candidates)} eligible vertices for "
            f"{point_count} requested seeding points. Increase "
            "feature_mesh_resolution or enlarge that IVD region.")
    if iterations < 0:
        raise ValueError("feature_mesh_iterations must be non-negative.")

    eps = np.finfo(np.float64).eps
    density = 1.0 / np.maximum(target_size, eps) ** 2
    density /= max(float(np.max(density)), eps)

    # Feature-aware farthest insertion is a robust initialization for the CVT
    # and prevents the optimized mesh generators from collapsing together.
    first = int(rng.choice(len(candidates), p=density / density.sum()))
    selected = np.empty(point_count, dtype=np.int64)
    selected[0] = first
    min_distance_sq = np.sum((candidates - candidates[first]) ** 2, axis=1)
    min_distance_sq[first] = 0.0
    available = np.ones(len(candidates), dtype=bool)
    available[first] = False
    for index in range(1, point_count):
        score = min_distance_sq * density
        score[~available] = -1.0
        chosen = int(np.argmax(score))
        selected[index] = chosen
        available[chosen] = False
        distance_sq = np.sum(
            (candidates - candidates[chosen]) ** 2, axis=1)
        min_distance_sq = np.minimum(min_distance_sq, distance_sq)

    centers = candidates[selected].copy()
    for _ in range(iterations):
        distance_sq = np.sum(
            (candidates[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        owner = np.argmin(distance_sq, axis=1)
        relaxed = centers.copy()
        for center_index in range(point_count):
            member_ids = np.flatnonzero(owner == center_index)
            # A generator is itself a candidate at zero distance, so a cell is
            # normally non-empty; retaining it makes the invariant explicit.
            if not len(member_ids):
                continue
            weights = density[member_ids]
            centroid = np.average(
                candidates[member_ids], axis=0, weights=weights)
            nearest_local = np.argmin(np.sum(
                (candidates[member_ids] - centroid) ** 2, axis=1))
            relaxed[center_index] = candidates[member_ids[nearest_local]]
        if np.array_equal(relaxed, centers):
            break
        centers = relaxed
    return centers


def _feature_adaptive_mesh_seedings(
        k_lines, rng, velocity_eval, time, high_fraction=0.8,
        high_ivd_percentile=70.0, mesh_resolution=64,
        mesh_size_ratio=4.0, monitor_power=1.5, mesh_iterations=5,
        clip_percentile=99.0, boundary_margin=1e-6, ivd_boundary_samples=64):
    """Seed pathlines using IVD-based mesh adaptation.

    A jittered background mesh is equipped with a target edge-length monitor
    derived from robustly normalized ``IVD = abs(omega - mean(omega))`` over
    the full domain ``[-1, 1]^2`` at the seeding time.  The top-IVD
    region and its complement are optimized independently using weighted CVT,
    with an exact (up to integer rounding) high/low point quota.  By default,
    80% of seeds lie in the top 30% IVD region and 20% lie in the rest.
    """
    target_count = _common_existing_seed_count(k_lines)
    high_fraction = float(high_fraction)
    high_ivd_percentile = float(high_ivd_percentile)
    mesh_resolution = int(mesh_resolution)
    mesh_size_ratio = float(mesh_size_ratio)
    monitor_power = float(monitor_power)
    mesh_iterations = int(mesh_iterations)
    clip_percentile = float(clip_percentile)
    boundary_margin = float(boundary_margin)

    if not 0.0 <= high_fraction <= 1.0:
        raise ValueError("feature_high_fraction must be in [0, 1].")
    if not 0.0 < high_ivd_percentile < 100.0:
        raise ValueError(
            "feature_high_ivd_percentile must be in (0, 100).")
    if mesh_size_ratio < 1.0:
        raise ValueError("feature_mesh_size_ratio must be at least 1.")
    if monitor_power <= 0.0:
        raise ValueError("feature_monitor_power must be greater than 0.")
    if not 0.0 < clip_percentile <= 100.0:
        raise ValueError("feature_clip_percentile must be in (0, 100].")
    if velocity_eval is None:
        raise ValueError(
            "FEATURE_ADAPTIVE_MESH_SEEDING requires the analytical unsteady "
            "velocity evaluator.")
    differentiable_eval = getattr(velocity_eval, "torch_eval", None)
    if differentiable_eval is None:
        raise ValueError(
            "FEATURE_ADAPTIVE_MESH_SEEDING requires velocity_eval.torch_eval "
            "to compute IVD with autograd.")

    high_count = int(round(target_count * high_fraction))
    low_count = target_count - high_count
    candidates = _feature_mesh_candidates(
        mesh_resolution, boundary_margin, rng)
    ivd = _autograd_ivd_at_points(
        differentiable_eval, candidates, float(time), ivd_boundary_samples)
    magnitude = np.nan_to_num(ivd, nan=0.0, posinf=0.0, neginf=0.0)

    # Rank-based partitioning makes the regional quotas well-defined even for
    # plateaus. Random tie-breaking avoids directional bias for zero-IVD
    # Killing fields, where no physically meaningful feature split exists.
    tie_break = rng.random(len(candidates))
    order = np.lexsort((tie_break, magnitude))
    requested_high_candidates = int(round(
        len(candidates) * (1.0 - high_ivd_percentile / 100.0)))
    high_candidate_count = requested_high_candidates
    if high_candidate_count < high_count or (
            len(candidates) - high_candidate_count) < low_count:
        raise ValueError(
            "feature_mesh_resolution and "
            "feature_high_ivd_percentile leave too few background-mesh "
            "vertices for the requested high/low-IVD quotas. Increase "
            "feature_mesh_resolution or enlarge the affected region.")
    high_ids = order[-high_candidate_count:]
    low_ids = order[:-high_candidate_count]

    contrast = np.maximum(magnitude - float(np.min(magnitude)), 0.0)
    scale = float(np.percentile(contrast, clip_percentile))
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        feature = np.full(len(candidates), 0.5, dtype=np.float64)
    else:
        feature = np.clip(contrast / scale, 0.0, 1.0)

    # h(x) is the mesh-adaptation monitor: smaller target edges produce higher
    # CVT density.  Low-region adaptation uses the complementary feature so its
    # 20% quota preferentially represents low-IVD background.
    high_strength = np.power(feature[high_ids], monitor_power)
    low_strength = np.power(1.0 - feature[low_ids], monitor_power)
    high_target_size = (
        mesh_size_ratio - (mesh_size_ratio - 1.0) * high_strength)
    low_target_size = (
        mesh_size_ratio - (mesh_size_ratio - 1.0) * low_strength)

    high_seeds = _weighted_cvt_mesh_points(
        candidates[high_ids], high_target_size, high_count, rng,
        mesh_iterations)
    low_seeds = _weighted_cvt_mesh_points(
        candidates[low_ids], low_target_size, low_count, rng,
        mesh_iterations)
    seeds = np.concatenate([high_seeds, low_seeds], axis=0)
    rng.shuffle(seeds)
    if seeds.shape != (target_count, 2):
        raise RuntimeError(
            "Feature-adaptive mesh seeding produced an unexpected shape: "
            f"{seeds.shape}, expected {(target_count, 2)}.")
    return seeds.astype(np.float64, copy=False)


def _ivd_importance_seedings(
        k_lines, rng, velocity_eval, time, candidate_multiplier=8,
        uniform_fraction=0.2, importance_power=1.5,
        clip_percentile=99.0, boundary_margin=1e-6, ivd_boundary_samples=64):
    """Sample more initial positions where IVD is larger.

    A uniform candidate cloud is evaluated with the differentiable analytical
    velocity field at the seeding time.  A configurable fraction of the final
    seeds is selected uniformly to retain background coverage; the remainder
    is sampled without replacement with probability proportional to a robustly
    scaled ``IVD = abs(omega - mean(omega))``, with the spatial mean over the
    full domain ``[-1, 1]^2``.  The returned count exactly matches both legacy
    samplers for an even ``count_k``.
    """
    target_count = _common_existing_seed_count(k_lines)
    candidate_multiplier = int(candidate_multiplier)
    uniform_fraction = float(uniform_fraction)
    importance_power = float(importance_power)
    clip_percentile = float(clip_percentile)
    boundary_margin = float(boundary_margin)

    if candidate_multiplier < 2:
        raise ValueError("ivd_candidate_multiplier must be at least 2.")
    if not 0.0 <= uniform_fraction <= 1.0:
        raise ValueError("ivd_uniform_fraction must be in [0, 1].")
    if importance_power <= 0.0:
        raise ValueError("ivd_importance_power must be greater than 0.")
    if not 0.0 < clip_percentile <= 100.0:
        raise ValueError("ivd_clip_percentile must be in (0, 100].")
    if not 0.0 <= boundary_margin < 1.0:
        raise ValueError("ivd_boundary_margin must be in [0, 1).")
    if velocity_eval is None:
        raise ValueError(
            "IVD_IMPORTANCE_SEEDING requires the analytical unsteady "
            "velocity evaluator.")

    differentiable_eval = getattr(velocity_eval, "torch_eval", None)
    if differentiable_eval is None:
        raise ValueError(
            "IVD_IMPORTANCE_SEEDING requires velocity_eval.torch_eval "
            "to compute IVD with autograd.")

    candidate_count = max(
        target_count, target_count * candidate_multiplier)
    lower = -1.0 + boundary_margin
    upper = 1.0 - boundary_margin
    candidates = rng.uniform(
        lower, upper, size=(candidate_count, 2)).astype(
            np.float64, copy=False)

    ivd = _autograd_ivd_at_points(
        differentiable_eval, candidates, float(time), ivd_boundary_samples)
    magnitude = np.nan_to_num(ivd, nan=0.0, posinf=0.0, neginf=0.0)

    # Remove the minimum IVD before scaling, preserving the IVD ranking.
    # Percentile clipping prevents one extreme candidate from taking almost
    # all probability mass.
    contrast = np.maximum(magnitude - float(np.min(magnitude)), 0.0)
    scale = float(np.percentile(contrast, clip_percentile))
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        importance_weights = np.ones(candidate_count, dtype=np.float64)
    else:
        scaled = np.clip(contrast / scale, 0.0, 1.0)
        importance_weights = np.power(scaled, importance_power)
        # Keep every finite candidate selectable; uniform_fraction determines
        # the explicit background quota, while this epsilon avoids a degenerate
        # probability vector when only a few candidates have non-zero IVD.
        importance_weights += np.finfo(np.float64).eps

    uniform_count = int(round(target_count * uniform_fraction))
    importance_count = target_count - uniform_count
    available = np.arange(candidate_count)
    if uniform_count:
        uniform_ids = rng.choice(
            available, size=uniform_count, replace=False)
        available_mask = np.ones(candidate_count, dtype=bool)
        available_mask[uniform_ids] = False
        available = available[available_mask]
    else:
        uniform_ids = np.empty(0, dtype=np.int64)

    if importance_count:
        weights = importance_weights[available]
        weights /= np.sum(weights)
        importance_ids = rng.choice(
            available, size=importance_count, replace=False, p=weights)
    else:
        importance_ids = np.empty(0, dtype=np.int64)

    selected_ids = np.concatenate([uniform_ids, importance_ids])
    rng.shuffle(selected_ids)
    seeds = candidates[selected_ids]
    if seeds.shape != (target_count, 2):
        raise RuntimeError(
            "IVD importance sampling produced an unexpected shape: "
            f"{seeds.shape}, expected {(target_count, 2)}.")
    return seeds.astype(np.float64, copy=False)


def _pathline_seedings(pcfg, rng, velocity_eval=None, time=None):
    """Generate pathline seeds according to ``pathline.sampling``."""
    method = _normalize_pathline_sampling(
        pcfg.get("sampling", GRID_CROSS_SEEDING))
    k_lines = int(pcfg["count_k"])
    if method == GRID_CROSS_SEEDING:
        return _grid_cross_seedings(
            k_lines, rng, float(pcfg.get("jitter_std", 0.05)))

    if method == IVD_IMPORTANCE_SEEDING:
        if time is None:
            raise ValueError(
                "IVD_IMPORTANCE_SEEDING requires the seeding time.")
        return _ivd_importance_seedings(
            k_lines,
            rng,
            velocity_eval,
            time,
            candidate_multiplier=pcfg.get(
                "ivd_candidate_multiplier",
                pcfg.get("vorticity_candidate_multiplier", 8)),
            uniform_fraction=pcfg.get(
                "ivd_uniform_fraction", pcfg.get("vorticity_uniform_fraction", 0.2)),
            importance_power=pcfg.get(
                "ivd_importance_power", pcfg.get("vorticity_importance_power", 1.5)),
            clip_percentile=pcfg.get(
                "ivd_clip_percentile", pcfg.get("vorticity_clip_percentile", 99.0)),
            boundary_margin=pcfg.get(
                "ivd_boundary_margin", pcfg.get("vorticity_boundary_margin", 1e-6)),
            ivd_boundary_samples=pcfg.get("ivd_boundary_samples", 64),
        )

    if method == FEATURE_ADAPTIVE_MESH_SEEDING:
        if time is None:
            raise ValueError(
                "FEATURE_ADAPTIVE_MESH_SEEDING requires the seeding time.")
        return _feature_adaptive_mesh_seedings(
            k_lines,
            rng,
            velocity_eval,
            time,
            high_fraction=pcfg.get("feature_high_fraction", 0.8),
            high_ivd_percentile=pcfg.get(
                "feature_high_ivd_percentile",
                pcfg.get("feature_high_vorticity_percentile", 70.0)),
            mesh_resolution=pcfg.get("feature_mesh_resolution", 64),
            mesh_size_ratio=pcfg.get("feature_mesh_size_ratio", 4.0),
            monitor_power=pcfg.get("feature_monitor_power", 1.5),
            mesh_iterations=pcfg.get("feature_mesh_iterations", 5),
            clip_percentile=pcfg.get("feature_clip_percentile", 99.0),
            boundary_margin=pcfg.get("feature_boundary_margin", 1e-6),
            ivd_boundary_samples=pcfg.get("ivd_boundary_samples", 64),
        )

    # Match the implicit double-to-int conversion in the C++ caller:
    # RecTangular4ClusterSampling(KLines * KLines * 0.25, ...).
    samples_per_cluster = int(k_lines * k_lines * 0.25)
    return _rectangular4_cluster_seedings(samples_per_cluster, rng)


def _pathline_seed_count(pcfg):
    """Return the deterministic number of seeds for the configured method."""
    method = _normalize_pathline_sampling(
        pcfg.get("sampling", GRID_CROSS_SEEDING))
    k_lines = int(pcfg["count_k"])
    if method == GRID_CROSS_SEEDING:
        return (k_lines // 2) ** 2 * 4
    if method in {
            IVD_IMPORTANCE_SEEDING,
            FEATURE_ADAPTIVE_MESH_SEEDING}:
        return _common_existing_seed_count(k_lines)
    return int(k_lines * k_lines * 0.25) * 4


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


def _autograd_vorticity_at_points(velocity_eval, points, time):
    """Evaluate ``dv/dx - du/dy`` directly from a differentiable velocity field."""
    device = torch.device(getattr(velocity_eval, "device", "cpu"))
    dtype = getattr(velocity_eval, "dtype", torch.float32)
    points_np = np.ascontiguousarray(points, dtype=np.float32)
    if points_np.size == 0:
        return np.empty(points_np.shape[:-1], dtype=np.float32)

    points_t = torch.as_tensor(
        points_np, device=device, dtype=dtype).detach().requires_grad_(True)
    velocity = velocity_eval(points_t, time)
    if velocity.shape != points_t.shape:
        raise ValueError(
            "Differentiable velocity evaluator must preserve the input shape; "
            f"got points {points_t.shape}, velocity {velocity.shape}.")

    grad_u = torch.autograd.grad(
        velocity[..., 0].sum(), points_t, retain_graph=True)[0]
    grad_v = torch.autograd.grad(
        velocity[..., 1].sum(), points_t)[0]
    vorticity = grad_v[..., 0] - grad_u[..., 1]
    return vorticity.detach().cpu().numpy().astype(np.float64, copy=False)


def _trapezoid_1d(values, coordinates):
    """Version-independent torch trapezoidal integration."""
    return torch.sum(
        0.5 * (values[1:] + values[:-1])
        * (coordinates[1:] - coordinates[:-1])
    )


def _mean_vorticity_from_boundary(
        velocity_eval, time, boundary_samples,
        domain=(-1.0, 1.0, -1.0, 1.0)):
    """Compute spatial mean vorticity from velocity using Green's theorem.

    For ``omega = dv/dx - du/dy``,
    ``integral(omega dA) = contour(u dx + v dy)``.  This avoids both a
    finite-difference curl and an expensive dense autograd grid.
    """
    xmin, xmax, ymin, ymax = [float(value) for value in domain]
    samples = max(2, int(boundary_samples))
    device = torch.device(getattr(velocity_eval, "device", "cpu"))
    dtype = getattr(velocity_eval, "dtype", torch.float32)
    x = torch.linspace(xmin, xmax, samples, device=device, dtype=dtype)
    y = torch.linspace(ymin, ymax, samples, device=device, dtype=dtype)

    boundary_points = torch.stack([
        torch.stack([x, torch.full_like(x, ymin)], dim=-1),
        torch.stack([x, torch.full_like(x, ymax)], dim=-1),
        torch.stack([torch.full_like(y, xmax), y], dim=-1),
        torch.stack([torch.full_like(y, xmin), y], dim=-1),
    ])
    with torch.no_grad():
        boundary_velocity = velocity_eval(boundary_points, time)
    bottom, top, right, left = boundary_velocity
    circulation = (
        _trapezoid_1d(bottom[:, 0], x)
        + _trapezoid_1d(right[:, 1], y)
        - _trapezoid_1d(top[:, 0], x)
        - _trapezoid_1d(left[:, 1], y)
    )
    area = (xmax - xmin) * (ymax - ymin)
    return float((circulation / area).detach().cpu())


def _autograd_ivd_at_points(velocity_eval, points, time, boundary_samples=64):
    """Evaluate IVD against the spatial mean over the full ``[-1, 1]^2`` domain.

    Shared by the seeding monitor and pathline features so the mean vorticity
    is independent of the candidate distribution or the selected seeds.
    """
    vorticity = _autograd_vorticity_at_points(velocity_eval, points, time)
    mean_vorticity = _mean_vorticity_from_boundary(
        velocity_eval, time, boundary_samples)
    return np.abs(vorticity - mean_vorticity)


def _compute_ivd_values_autograd(
        velocity_eval, points, step_ids, times, boundary_samples):
    """Compute continuous IVD at scattered pathline points."""
    ivd_values = np.empty(len(points), dtype=np.float64)
    for step in np.unique(step_ids):
        at_step = step_ids == step
        time = float(times[step])
        ivd_values[at_step] = _autograd_ivd_at_points(
            velocity_eval, points[at_step], time, boundary_samples)
    return ivd_values


def collect_pathline_info(
        unsteady_eval, _sampled_field, label_eval, pcfg, ocfg, rng):
    """Port of PathlineIntegrationInfoCollect2D.

    Returns [num_lines, length, 7], where features are
    [x, y, time, IVD, distance_or_seed_label, u, v].

    IVD is evaluated continuously from the differentiable Vatistas/Tobias
    velocity formula.  No sampled-grid finite differences or IVD interpolation
    are used.
    """
    length = int(pcfg["length"])
    boundary_samples = pcfg.get("ivd_boundary_samples", ocfg.get("grid", 64))
    seeds = _pathline_seedings(
        {**pcfg, "ivd_boundary_samples": boundary_samples}, rng,
        velocity_eval=unsteady_eval, time=ocfg["tmin"])
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

    differentiable_eval = getattr(unsteady_eval, "torch_eval", None)
    if differentiable_eval is None:
        raise ValueError(
            "Autograd IVD requires an unsteady evaluator with a differentiable "
            "torch_eval counterpart. Build it from make_steady_eval() and the "
            "Tobias observer transformation."
        )
    ivd_values = _compute_ivd_values_autograd(
        differentiable_eval,
        valid_points,
        step_ids,
        times,
        boundary_samples,
    )
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
        tmin=0.0, tmax=1.0, rng=None, sampling=GRID_CROSS_SEEDING):
    """C++-named convenience wrapper with selectable seed sampling."""
    if rng is None:
        rng = np.random.default_rng()
    pcfg = {
        "count_k": int(count_k),
        "length": int(output_pathline_length),
        "sampling": _normalize_pathline_sampling(sampling),
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


def _write_cereal_uint8_vector(file_path, values):
    """Write a Cereal-compatible ``std::vector<uint8_t>``.

    ``SteadyVelocityGridSegmentationMVUnet`` reads segmentation files as an
    eight-byte vector length followed by uint8 values, so MVUnet labels use
    this layout instead of the legacy float32 segmentation-binary layout.
    """
    flat = np.ascontiguousarray(values, dtype=np.uint8).reshape(-1)
    with open(file_path, "wb") as handle:
        handle.write(np.asarray([flat.size], dtype="<u8").tobytes())
        handle.write(flat.tobytes(order="C"))


def _build_mvunet_velocity_features(
        velocity_slice, domain=(-1.0, 1.0, -1.0, 1.0)):
    """Return dense MVUnet channels ``[u, v, curl, IVD]`` for one 2D frame.

    Curl and IVD match ``CppProjects/src/VectorFieldCompute.cpp``: centered
    differences on interior grid points, zero-valued boundaries, and
    ``IVD = abs(curl - mean_interior_curl)`` on the interior.  The generator's
    segmentation is expressed in the t0 coordinate system, so callers must
    pass the first Tobias-observed frame.
    """
    velocity = np.asarray(velocity_slice, dtype=np.float32)
    if velocity.ndim != 3 or velocity.shape[-1] != 2:
        raise ValueError(
            "MVUnet velocity input must be shaped [Y, X, 2], got "
            f"{velocity.shape}.")
    height, width = velocity.shape[:2]
    if height < 3 or width < 3:
        raise ValueError(
            "MVUnet curl/IVD computation requires at least a 3x3 grid.")

    xmin, xmax, ymin, ymax = [float(value) for value in domain]
    dx = (xmax - xmin) / float(width - 1)
    dy = (ymax - ymin) / float(height - 1)
    if dx == 0.0 or dy == 0.0:
        raise ValueError("MVUnet velocity domain must have non-zero extent.")

    curl = np.zeros((height, width), dtype=np.float32)
    dv_dx = (
        velocity[1:-1, 2:, 1] - velocity[1:-1, :-2, 1]
    ) * (0.5 / dx)
    du_dy = (
        velocity[2:, 1:-1, 0] - velocity[:-2, 1:-1, 0]
    ) * (0.5 / dy)
    curl[1:-1, 1:-1] = dv_dx - du_dy

    mean_curl = float(np.mean(curl[1:-1, 1:-1], dtype=np.float64))
    ivd = np.zeros_like(curl)
    ivd[1:-1, 1:-1] = np.abs(
        curl[1:-1, 1:-1] - mean_curl)
    return np.ascontiguousarray(
        np.concatenate(
            [velocity, curl[..., None], ivd[..., None]], axis=-1),
        dtype=np.float32,
    )


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


def _lic_mesh_output_path(lic_path):
    """Return the one-to-one companion filename for a LIC mesh overlay."""
    stem, extension = os.path.splitext(lic_path)
    return f"{stem}_mesh{extension or '.png'}"


def _lic_mesh_edges(seeding_points):
    """Build unique edges of the Delaunay mesh induced by seeding points."""
    points = np.asarray(seeding_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("LIC mesh seeding_points must be shaped [N, 2].")
    finite_ids = np.flatnonzero(np.all(np.isfinite(points), axis=1))
    if len(finite_ids) < 3:
        return np.empty((0, 2), dtype=np.int64)
    finite_points = points[finite_ids]
    try:
        from scipy.spatial import Delaunay, QhullError
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SciPy is required when lic.mesh_overlay_enabled is true.") from exc

    try:
        triangles = Delaunay(finite_points).simplices
    except QhullError:
        # QJ resolves otherwise valid but nearly collinear/duplicate inputs.
        try:
            triangles = Delaunay(finite_points, qhull_options="QJ").simplices
        except QhullError:
            logging.warning(
                "[LIC mesh] seeding points are degenerate; drawing points "
                "without triangle edges.")
            return np.empty((0, 2), dtype=np.int64)

    local_edges = np.concatenate([
        triangles[:, [0, 1]],
        triangles[:, [1, 2]],
        triangles[:, [2, 0]],
    ], axis=0)
    local_edges.sort(axis=1)
    local_edges = np.unique(local_edges, axis=0)
    return finite_ids[local_edges].astype(np.int64, copy=False)


def _lic_mesh_color(lcfg, key, default):
    """Validate an RGB color stored in the LIC configuration."""
    color = np.asarray(lcfg.get(key, default), dtype=np.int64)
    if color.shape != (3,) or np.any((color < 0) | (color > 255)):
        raise ValueError(f"lic.{key} must contain three integers in [0, 255].")
    return tuple(int(channel) for channel in color)


def render_lic_mesh_overlay(
        lic_image, seeding_points, lcfg, mesh_edges=None):
    """Draw adaptive seed points and their unstructured mesh over a LIC image.

    Computational coordinates and LIC raster rows both run from -1 to 1, so
    no vertical flip is needed.  Large outlined discs deliberately keep the
    seeding points conspicuous above the Delaunay edges and LIC texture.
    """
    try:
        from PIL import Image, ImageDraw
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Pillow is required when lic.mesh_overlay_enabled is true.") from exc

    image = np.asarray(lic_image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("LIC mesh overlay expects an RGB image shaped [H, W, 3].")
    points = np.asarray(seeding_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("LIC mesh seeding_points must be shaped [N, 2].")
    if mesh_edges is None:
        mesh_edges = _lic_mesh_edges(points)
    mesh_edges = np.asarray(mesh_edges, dtype=np.int64)

    height, width = image.shape[:2]
    pixels = np.empty_like(points)
    pixels[:, 0] = (points[:, 0] + 1.0) * 0.5 * width
    pixels[:, 1] = (points[:, 1] + 1.0) * 0.5 * height
    pixels[:, 0] = np.clip(pixels[:, 0], 0.0, width - 1.0)
    pixels[:, 1] = np.clip(pixels[:, 1], 0.0, height - 1.0)

    base = Image.fromarray(image, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, mode="RGBA")
    line_rgb = _lic_mesh_color(lcfg, "mesh_line_color", (0, 255, 255))
    point_rgb = _lic_mesh_color(lcfg, "mesh_point_color", (255, 55, 35))
    outline_rgb = _lic_mesh_color(
        lcfg, "mesh_point_outline_color", (255, 255, 255))
    line_alpha = int(np.clip(lcfg.get("mesh_line_alpha", 210), 0, 255))
    point_alpha = int(np.clip(lcfg.get("mesh_point_alpha", 245), 0, 255))
    line_width = max(1, int(lcfg.get("mesh_line_width", 1)))
    point_radius = max(1, int(lcfg.get("mesh_point_radius", 2)))
    outline_width = max(0, int(lcfg.get("mesh_point_outline_width", 1)))

    for first, second in mesh_edges:
        if not (0 <= first < len(pixels) and 0 <= second < len(pixels)):
            raise ValueError("LIC mesh contains an out-of-range point index.")
        if np.all(np.isfinite(pixels[[first, second]])):
            draw.line(
                [tuple(pixels[first]), tuple(pixels[second])],
                fill=(*line_rgb, line_alpha), width=line_width)

    # Draw points after edges so the deliberately large markers remain visible.
    for x_pixel, y_pixel in pixels:
        if not np.isfinite(x_pixel) or not np.isfinite(y_pixel):
            continue
        bounds = [
            x_pixel - point_radius, y_pixel - point_radius,
            x_pixel + point_radius, y_pixel + point_radius,
        ]
        draw.ellipse(
            bounds,
            fill=(*point_rgb, point_alpha),
            outline=(*outline_rgb, 255) if outline_width else None,
            width=outline_width)
    return np.asarray(Image.alpha_composite(base, overlay).convert("RGB"))


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
        "mesh_overlay_enabled": bool(
            lic.get("mesh_overlay_enabled", True)),
        "mesh_line_color": list(
            lic.get("mesh_line_color", [0, 255, 255])),
        "mesh_line_alpha": int(lic.get("mesh_line_alpha", 210)),
        "mesh_line_width": int(lic.get("mesh_line_width", 1)),
        "mesh_point_color": list(
            lic.get("mesh_point_color", [255, 55, 35])),
        "mesh_point_alpha": int(lic.get("mesh_point_alpha", 245)),
        "mesh_point_radius": int(lic.get("mesh_point_radius", 2)),
        "mesh_point_outline_color": list(
            lic.get("mesh_point_outline_color", [255, 255, 255])),
        "mesh_point_outline_width": int(
            lic.get("mesh_point_outline_width", 1)),
        "mesh_seed_offset": int(lic.get("mesh_seed_offset", 1733)),
    }


def save_steady_lic_images(
        steady_fields, velocity_evals, label_evals, observers_per_field,
        out_dir, lcfg, pcfg, ocfg, seed):
    """Write steady LIC images and paired adaptive-mesh overlay images."""
    if not lcfg["enabled"]:
        return 0, 0
    if lcfg["mesh_overlay_enabled"] and not pcfg["enabled"]:
        raise ValueError(
            "lic.mesh_overlay_enabled requires pathline.enabled: true.")
    lic_dir = os.path.join(out_dir, "velocity_mix", "LIC")
    os.makedirs(lic_dir, exist_ok=True)
    noise = _make_lic_noise(lcfg, seed)
    mesh_rng = np.random.default_rng(
        int(seed) + int(lcfg["mesh_seed_offset"]))
    mesh_count = 0
    for steady_index, field in enumerate(steady_fields):
        first_id = steady_index * observers_per_field
        last_id = first_id + observers_per_field - 1
        image = render_lic_image(
            field, lcfg, noise, label_evals[steady_index])
        lic_path = os.path.join(
            lic_dir, f"{first_id}_{last_id}_steady_lic.png")
        _save_png(image, lic_path)
        if lcfg["mesh_overlay_enabled"]:
            seeds = _pathline_seedings(
                pcfg, mesh_rng, velocity_eval=velocity_evals[steady_index],
                time=ocfg["tmin"])
            mesh_image = render_lic_mesh_overlay(image, seeds, lcfg)
            _save_png(mesh_image, _lic_mesh_output_path(lic_path))
            mesh_count += 1
    logging.info(
        f"[LIC] wrote {len(steady_fields)} steady LIC images and "
        f"{mesh_count} paired mesh overlays to {lic_dir}.")
    return len(steady_fields), mesh_count


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
    sampling = _normalize_pathline_sampling(
        p.get("sampling", GRID_CROSS_SEEDING))
    return {
        "enabled": bool(p.get("enabled", True)),
        "count_k": int(p.get("count_k", 16)),
        "length": int(p.get("length", 16)),
        "sampling": sampling,
        "jitter_std": float(p.get("jitter_std", 0.05)),
        # Accept legacy vorticity option names; all monitors now use IVD.
        "ivd_candidate_multiplier": int(p.get(
            "ivd_candidate_multiplier", p.get("vorticity_candidate_multiplier", 8))),
        "ivd_uniform_fraction": float(p.get(
            "ivd_uniform_fraction", p.get("vorticity_uniform_fraction", 0.2))),
        "ivd_importance_power": float(p.get(
            "ivd_importance_power", p.get("vorticity_importance_power", 1.5))),
        "ivd_clip_percentile": float(p.get(
            "ivd_clip_percentile", p.get("vorticity_clip_percentile", 99.0))),
        "ivd_boundary_margin": float(p.get(
            "ivd_boundary_margin", p.get("vorticity_boundary_margin", 1e-6))),
        "feature_high_fraction": float(
            p.get("feature_high_fraction", 0.8)),
        "feature_high_ivd_percentile": float(p.get(
            "feature_high_ivd_percentile",
            p.get("feature_high_vorticity_percentile", 70.0))),
        "feature_mesh_resolution": int(
            p.get("feature_mesh_resolution", 64)),
        "feature_mesh_size_ratio": float(
            p.get("feature_mesh_size_ratio", 4.0)),
        "feature_monitor_power": float(
            p.get("feature_monitor_power", 1.5)),
        "feature_mesh_iterations": int(
            p.get("feature_mesh_iterations", 5)),
        "feature_clip_percentile": float(
            p.get("feature_clip_percentile", 99.0)),
        "feature_boundary_margin": float(
            p.get("feature_boundary_margin", 1e-6)),
        "padding_value": float(p.get("padding_value", -1000.0)),
        "max_reseed_attempts": int(p.get("max_reseed_attempts", 32)),
        "ivd_boundary_samples": max(
            2, int(p.get("ivd_boundary_samples", cfg.generate.grid))),
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
    save_velocity_bin = bool(ucfg.get("save_velocity_bin", False))
    velocity_bin_dir = os.path.join(
        out_dir, str(ucfg.get("velocity_bin_dir", "mvunet")))
    if save_velocity_bin:
        os.makedirs(velocity_bin_dir, exist_ok=True)
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
        if lcfg["mesh_overlay_enabled"] and not pcfg["enabled"]:
            raise ValueError(
                "lic.mesh_overlay_enabled requires pathline.enabled: true.")
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
    mvunet_channel_min = np.full(4, np.inf, dtype=np.float64)
    mvunet_channel_max = np.full(4, -np.inf, dtype=np.float64)
    deformed_lic_count = 0
    deformed_lic_mesh_count = 0

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
            if save_velocity_bin:
                # Tobias uses Q(0)=I and c(0)=0, so the steady-coordinate
                # segmentation is spatially aligned with the first frame.
                mvunet_velocity = _build_mvunet_velocity_features(field[0])
                _write_cereal_float_vector(
                    os.path.join(velocity_bin_dir, f"{sample_tag}.bin"),
                    mvunet_velocity)
                _write_cereal_uint8_vector(
                    os.path.join(
                        velocity_bin_dir,
                        f"{sample_tag}_segmentation.bin"),
                    segmentation)
                flattened_features = mvunet_velocity.reshape(-1, 4)
                mvunet_channel_min = np.minimum(
                    mvunet_channel_min,
                    np.min(flattened_features, axis=0))
                mvunet_channel_max = np.maximum(
                    mvunet_channel_max,
                    np.max(flattened_features, axis=0))
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
                seeding_points = pathlines[:, 0, 0:2].astype(
                    np.float64, copy=False)
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
                mesh_edges = (
                    _lic_mesh_edges(seeding_points)
                    if lcfg["mesh_overlay_enabled"] else None)
                for frame_index in range(
                        0, field.shape[0], int(lcfg["frame_stride"])):
                    image = render_lic_image(
                        field[frame_index], lcfg, lic_noise, label_eval)
                    lic_path = os.path.join(
                        lic_dir,
                        f"{sample_tag}deformed_{frame_index}lic.png")
                    _save_png(image, lic_path)
                    deformed_lic_count += 1
                    if lcfg["mesh_overlay_enabled"]:
                        mesh_image = render_lic_mesh_overlay(
                            image, seeding_points, lcfg, mesh_edges)
                        _save_png(
                            mesh_image, _lic_mesh_output_path(lic_path))
                        deformed_lic_mesh_count += 1
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
            _pathline_seed_count(pcfg) if pcfg["enabled"] else 0),
        "pathline_feature": 7 if pcfg["enabled"] else 0,
        "velocity_bin_count": written if save_velocity_bin else 0,
        "velocity_bin_dir": (
            os.path.relpath(velocity_bin_dir, out_dir)
            if save_velocity_bin else None),
        "velocity_bin_channel_min": (
            mvunet_channel_min.tolist() if save_velocity_bin else None),
        "velocity_bin_channel_max": (
            mvunet_channel_max.tolist() if save_velocity_bin else None),
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
        "deformed_lic_mesh_count": deformed_lic_mesh_count,
        "lic_mesh_overlay_enabled": bool(
            lcfg["enabled"] and lcfg["mesh_overlay_enabled"]),
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
        "pathlineIVDImportance": ({
            "candidateMultiplier": int(
                pcfg["ivd_candidate_multiplier"]),
            "uniformFraction": float(
                pcfg["ivd_uniform_fraction"]),
            "importancePower": float(
                pcfg["ivd_importance_power"]),
            "clipPercentile": float(
                pcfg["ivd_clip_percentile"]),
            "boundaryMargin": float(
                pcfg["ivd_boundary_margin"]),
            "ivdBoundarySamples": int(pcfg["ivd_boundary_samples"]),
            "samplingTime": "tmin",
            "criterion": "ivd_autograd",
        } if pcfg["enabled"] and pcfg["sampling"] ==
            IVD_IMPORTANCE_SEEDING else None),
        "pathlineFeatureAdaptiveMesh": ({
            "highIVDSeedFraction": float(
                pcfg["feature_high_fraction"]),
            "highIVDPercentile": float(
                pcfg["feature_high_ivd_percentile"]),
            "meshResolution": int(pcfg["feature_mesh_resolution"]),
            "meshSizeRatio": float(pcfg["feature_mesh_size_ratio"]),
            "monitorPower": float(pcfg["feature_monitor_power"]),
            "optimization": "weighted_centroidal_voronoi",
            "optimizationIterations": int(
                pcfg["feature_mesh_iterations"]),
            "clipPercentile": float(pcfg["feature_clip_percentile"]),
            "boundaryMargin": float(pcfg["feature_boundary_margin"]),
            "ivdBoundarySamples": int(pcfg["ivd_boundary_samples"]),
            "samplingTime": "tmin",
            "criterion": "ivd_autograd",
        } if pcfg["enabled"] and pcfg["sampling"] ==
            FEATURE_ADAPTIVE_MESH_SEEDING else None),
        "outputPathlineLength": int(pcfg["length"]) if pcfg["enabled"] else 0,
        "outputPathlinesCountK": int(pcfg["count_k"]) if pcfg["enabled"] else 0,
        "PathlineCount": int(stats["pathline_count"]),
        "PathlineFeature": int(stats["pathline_feature"]),
        "pathlineFeatureOrder": [
            "position_x", "position_y", "time", "ivd",
            "distance_or_seed_label", "velocity_u", "velocity_v",
        ] if pcfg["enabled"] else [],
        "ivdMethod": (
            "autograd_vorticity_green_boundary_mean"
            if pcfg["enabled"] else None
        ),
        "ivdBoundarySamples": (
            int(pcfg["ivd_boundary_samples"]) if pcfg["enabled"] else 0
        ),
        "velocityBinaryDirectory": stats["velocity_bin_dir"],
        "velocityBinaryCount": int(stats["velocity_bin_count"]),
        "velocityBinaryFrame": 0 if stats["velocity_bin_count"] else None,
        "velocityBinaryChannels": [
            "velocity_u", "velocity_v", "curl", "ivd",
        ] if stats["velocity_bin_count"] else [],
        "velocityBinaryChannelMin": stats["velocity_bin_channel_min"],
        "velocityBinaryChannelMax": stats["velocity_bin_channel_max"],
        "velocityBinaryLayout": (
            "cereal uint64 count + float32[y,x,channel]"
            if stats["velocity_bin_count"] else None),
        "velocitySegmentationBinaryLayout": (
            "cereal uint64 count + uint8[y,x]"
            if stats["velocity_bin_count"] else None),
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
        "steadyLicMeshCount": int(stats["steady_lic_mesh_count"]),
        "deformedLicMeshCount": int(stats["deformed_lic_mesh_count"]),
        "licMeshOverlayEnabled": bool(
            stats["lic_mesh_overlay_enabled"]),
        "licMeshCompanionSuffix": (
            "_mesh.png" if stats["lic_mesh_overlay_enabled"] else None),
        "targetUnsteady": int(target_total),
        "writtenUnsteady": int(written),
    }
    meta_paths = [
        os.path.join(out_dir, "meta.json"),
        os.path.join(out_dir, "velocity_mix", "meta.json"),
    ]
    if stats["velocity_bin_dir"]:
        meta_paths.append(
            os.path.join(out_dir, stats["velocity_bin_dir"], "meta.json"))
    os.makedirs(os.path.dirname(meta_paths[1]), exist_ok=True)
    for meta_path in dict.fromkeys(meta_paths):
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
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

    cfg = EasyConfig()
    cfg.load(args.config, recursive=False)
    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
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
    grid, steady_fields, steady_labels, source, evals, label_evals = render_steady_set(
        specs, ocfg, bounds, device)

    out_dir = args.out_dir or cfg.output.get("out_dir", "outputs/vatistas_unsteady")
    os.makedirs(out_dir, exist_ok=True)
    if bool(cfg.output.get("save_steady", True)):
        save_steady(out_dir, specs, grid, steady_fields, steady_labels, source, counts)
    steady_lic_count, steady_lic_mesh_count = save_steady_lic_images(
        steady_fields, evals, label_evals, ocfg["per_field"], out_dir,
        lcfg, pcfg, ocfg, seed)

    # 2. Tobias observation + pathline integration, mirroring
    # generateMixUnsteadyFieldPathline (60000 target at the paper scale).
    target_total, written, abc_all, stats = generate_unsteady(
        specs, evals, label_evals, steady_labels, grid, ocfg, pcfg, lcfg, cfg,
        out_dir, args.full)
    stats["steady_lic_count"] = steady_lic_count
    stats["steady_lic_mesh_count"] = steady_lic_mesh_count

    np.savez_compressed(os.path.join(out_dir, "dataset_manifest.npz"),
                        target_steady=np.int64(len(specs)), target_unsteady=np.int64(target_total),
                        written_unsteady=np.int64(written), observers_per_field=np.int64(ocfg["per_field"]),
                        source_counts=np.array([counts[SRC_RANDOM], counts[SRC_FITTED], counts[SRC_KILLING]], np.int64),
                        written_observers=abc_all,
                        observer_transformation=np.asarray(ocfg["transformation"]),
                        pathline_count=np.int64(stats["pathline_count"]),
                        pathline_length=np.int64(pcfg["length"] if pcfg["enabled"] else 0),
                        pathline_feature=np.int64(stats["pathline_feature"]),
                        velocity_bin_count=np.int64(
                            stats["velocity_bin_count"]),
                        velocity_bin_dir=np.asarray(
                            stats["velocity_bin_dir"] or ""),
                        velocity_bin_channels=np.asarray(
                            ["velocity_u", "velocity_v", "curl", "ivd"]),
                        velocity_bin_channel_min=np.asarray(
                            stats["velocity_bin_channel_min"]
                            if stats["velocity_bin_channel_min"] is not None
                            else [], dtype=np.float32),
                        velocity_bin_channel_max=np.asarray(
                            stats["velocity_bin_channel_max"]
                            if stats["velocity_bin_channel_max"] is not None
                            else [], dtype=np.float32),
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
                        steady_lic_mesh_count=np.int64(
                            stats["steady_lic_mesh_count"]),
                        deformed_lic_mesh_count=np.int64(
                            stats["deformed_lic_mesh_count"]),
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
    if stats["velocity_bin_count"]:
        logging.info(
            f"[totals] MVUNET VELOCITY BIN pairs = "
            f"{stats['velocity_bin_count']} in "
            f"{os.path.join(out_dir, stats['velocity_bin_dir'])}")
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
        if lcfg["mesh_overlay_enabled"]:
            logging.info(
                f"[totals] LIC MESH companions = "
                f"{stats['steady_lic_mesh_count']} steady + "
                f"{stats['deformed_lic_mesh_count']} deformed "
                f"(suffix: _mesh.png)")
    logging.info("=" * 78)

    if bool(cfg.output.get("save_plots", True)) and not args.no_plots:
        save_plots(out_dir, specs, grid, steady_fields, steady_labels, source, ocfg, evals, cfg)
    logging.info("[main] done.")

if __name__ == "__main__":
    main()
