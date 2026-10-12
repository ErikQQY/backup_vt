"""Generate noisy Vatistas/Tobias training data for MVUNet.

The clean generator remains the source of Vatistas parameters, analytical
velocity samples, observer motion, and analytical vortex labels.  This module
adds independent Gaussian noise to the sampled ``u`` and ``v`` arrays and then
switches every downstream operation to :class:`NoisyGridVelocityEvaluator`:

* pathline RK4 stages use space-time interpolation of the noisy velocity grid;
* velocity gradients and vorticity use finite differences on that grid;
* IVD, the vorticity monitor, adaptive mesh, and pathline descriptors use those
  finite-difference quantities; and
* segmentation and seed labels still use the clean analytical Vatistas label.

The MVUNet pair written for every sample is a Cereal-compatible
``float32[Y,X,4]`` tensor ordered as ``[noisy_u, noisy_v, curl, IVD]`` plus a
paired clean ``uint8[Y,X]`` segmentation.  Pathlines are also required and
persist all 18 noisy-grid descriptors so the vorticity monitor and adaptive
mesh cannot be accidentally bypassed by a configuration change.

Run:
    python VatistasFlowDatasetGeneratorNoiseMVUNet.py
    python VatistasFlowDatasetGeneratorNoiseMVUNet.py \
        --config config/VatistasDatasetNoiseMVUNet.yaml
    python SplitMVUNet.py --source outputs/vatistas_noise_mvunet_raw \
        --target outputs/vatistas_noise_mvunet_trainable
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

import numpy as np

import VatistasFlowDatasetGenerator as clean_generator


LEGACY_PATHLINE_FEATURE_NAMES = (
    "position_x",
    "position_y",
    "time",
    "ivd",
    "distance_or_seed_label",
    "velocity_u",
    "velocity_v",
)

EXTENDED_PATHLINE_FEATURE_NAMES = LEGACY_PATHLINE_FEATURE_NAMES + (
    "lavd",
    "vorticity",
    "du_dx",
    "du_dy",
    "dv_dx",
    "dv_dy",
    "divergence",
    "strain_magnitude",
    "velocity_gradient_determinant",
    "swirl_strength",
    "q_criterion",
)

# Backward-compatible public name for callers that imported the original
# 18-feature tuple before the configurable legacy output was added.
PATHLINE_FEATURE_NAMES = EXTENDED_PATHLINE_FEATURE_NAMES

_ORIGINAL_READ_OBSERVER_CFG = clean_generator.read_observer_cfg
_ORIGINAL_READ_PATHLINE_CFG = clean_generator.read_pathline_cfg
_ORIGINAL_TOBIAS_TRANSFORMATION = clean_generator.tobias_observer_transformation
_ORIGINAL_GENERATE_UNSTEADY = clean_generator.generate_unsteady
_ORIGINAL_SAVE_STEADY = clean_generator.save_steady
_ORIGINAL_SAVE_STEADY_LIC_IMAGES = clean_generator.save_steady_lic_images
_ORIGINAL_SAVE_DATASET_META = clean_generator._save_dataset_meta

_RUNTIME: dict[str, Any] = {
    "noise": None,
    "unsteady_noise_index": 0,
}

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "config",
    "VatistasDatasetNoiseMVUNet.yaml",
)


def _pair(value, name):
    """Normalize a scalar, [u, v], or {u: ..., v: ...} setting."""
    if isinstance(value, dict):
        if "u" not in value or "v" not in value:
            raise ValueError(f"{name} mapping must contain both 'u' and 'v'.")
        pair = (float(value["u"]), float(value["v"]))
    elif isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != 2:
            raise ValueError(f"{name} sequence must contain [u, v].")
        pair = (float(value[0]), float(value[1]))
    else:
        pair = (float(value), float(value))
    if not np.all(np.isfinite(pair)):
        raise ValueError(f"{name} must be finite, got {pair}.")
    return pair


def read_noise_cfg(cfg):
    """Read the noise controls, with top-level ``noise_level`` taking priority.

    ``noise_level`` may be a scalar (same relative level for both components),
    ``[u_level, v_level]``, or ``{u: ..., v: ...}``.
    """
    section = cfg.get("noise", {}) or {}
    level = cfg.get("noise_level", section.get("noise_level", 0.0))
    level_u, level_v = _pair(level, "noise_level")
    if level_u < 0.0 or level_v < 0.0:
        raise ValueError("noise_level values must be non-negative.")

    mean_u, mean_v = _pair(section.get("mean", 0.0), "noise.mean")
    scale = str(section.get("scale", "component_rms")).strip().lower()
    if scale not in {"component_rms", "velocity_rms", "absolute"}:
        raise ValueError(
            "noise.scale must be 'component_rms', 'velocity_rms', or "
            f"'absolute', got {scale!r}.")
    return {
        "level_u": level_u,
        "level_v": level_v,
        "mean_u": mean_u,
        "mean_v": mean_v,
        "scale": scale,
        "seed": int(cfg.get("seed", 0)),
        "seed_offset": int(section.get("seed_offset", 104729)),
    }


def _config_path_from_argv(argv):
    """Resolve ``--config`` without consuming the clean generator's CLI."""
    for index, argument in enumerate(argv):
        if argument.startswith("--config="):
            return argument.split("=", 1)[1]
        if argument == "--config":
            if index + 1 >= len(argv):
                raise ValueError("--config requires a YAML file path.")
            return argv[index + 1]
    return DEFAULT_CONFIG_PATH


def validate_mvunet_config(cfg):
    """Reject settings that would bypass a required noisy-grid computation."""
    if "noise_level" not in cfg:
        raise ValueError(
            "VatistasDatasetNoiseMVUNet.yaml must define top-level "
            "noise_level so u/v Gaussian noise is explicit.")
    read_noise_cfg(cfg)

    observer = cfg.get("observer", {}) or {}
    transformation = str(
        observer.get("transformation", "tobias")).strip().lower()
    if transformation != "tobias":
        raise ValueError(
            "NoiseMVUNet requires observer.transformation: tobias so a clean "
            "analytical field is sampled before grid noise is added.")
    if not bool(observer.get("make_labels", False)):
        raise ValueError(
            "NoiseMVUNet requires observer.make_labels: true for clean "
            "analytical Vatistas segmentation labels.")
    if int(observer.get("timesteps", 0)) < 2:
        raise ValueError("observer.timesteps must be at least 2.")
    if float(observer.get("tmax", 1.0)) <= float(observer.get("tmin", 0.0)):
        raise ValueError("observer.tmax must be greater than observer.tmin.")

    pathline = cfg.get("pathline", {}) or {}
    sampling = clean_generator._normalize_pathline_sampling(
        pathline.get("sampling", clean_generator.GRID_CROSS_SEEDING))
    if not bool(pathline.get("enabled", False)):
        raise ValueError(
            "NoiseMVUNet requires pathline.enabled: true so pathlines and "
            "noisy-grid descriptors are computed.")
    if sampling != clean_generator.FEATURE_ADAPTIVE_MESH_SEEDING:
        raise ValueError(
            "NoiseMVUNet requires pathline.sampling: "
            "FEATURE_ADAPTIVE_MESH_SEEDING so the noisy vorticity monitor and "
            "adaptive mesh are constructed.")
    if not bool(pathline.get("save_all_features", False)):
        raise ValueError(
            "NoiseMVUNet requires pathline.save_all_features: true to compute "
            "LAVD, velocity gradients, strain and divergence in memory.")
    if int(pathline.get("length", 0)) < 2:
        raise ValueError("pathline.length must be at least 2.")

    output = cfg.get("output", {}) or {}
    if not bool(output.get("save_velocity_bin", False)):
        raise ValueError(
            "NoiseMVUNet requires output.save_velocity_bin: true for "
            "[noisy_u,noisy_v,curl,IVD] training tensors.")
    save_pathline_bin = bool(output.get("save_pathline_bin", False))
    save_sample_meta = bool(output.get("save_sample_meta", False))
    velocity_dir = str(output.get("velocity_bin_dir", "mvunet"))
    pathline_dir = str(output.get("pathline_bin_dir", "pathlines"))
    sample_meta_dir = str(output.get("sample_meta_dir", pathline_dir))
    if save_pathline_bin and (
            os.path.normpath(velocity_dir) == os.path.normpath(pathline_dir)):
        raise ValueError(
            "output.velocity_bin_dir and output.pathline_bin_dir must differ; "
            "the MVUNet loader treats every non-segmentation .bin in its "
            "folder as a dense four-channel sample.")
    if save_sample_meta and not save_pathline_bin:
        raise ValueError(
            "output.save_sample_meta requires output.save_pathline_bin: true; "
            "otherwise the per-sample JSON files have no pathline payload.")
    if save_sample_meta and (
            os.path.normpath(sample_meta_dir) != os.path.normpath(pathline_dir)):
        raise ValueError(
            "output.sample_meta_dir must equal output.pathline_bin_dir so "
            "sample_*meta.json resolves its paired *_pathline.bin.")

    grid_size = int((cfg.get("generate", {}) or {}).get("grid", 0))
    if grid_size < 8 or grid_size % 8:
        raise ValueError(
            "generate.grid must be at least 8 and divisible by 8 for the "
            "three-level MVUNet encoder/decoder.")
    observers_per_field = int(observer.get("per_field", 0))
    subset = int(output.get("unsteady_subset", 0))
    if observers_per_field <= 0:
        raise ValueError("observer.per_field must be positive.")
    if subset <= 0 or subset % observers_per_field:
        raise ValueError(
            "output.unsteady_subset must be positive and divisible by "
            "observer.per_field so SplitMVUNet.py keeps complete observer "
            "groups in each split.")


def read_pathline_cfg(cfg):
    """Read and enforce the adaptive 18-channel noisy pathline contract."""
    pcfg = _ORIGINAL_READ_PATHLINE_CFG(cfg)
    section = cfg.get("pathline", {}) or {}
    save_all_features = section.get("save_all_features", True)
    if not isinstance(save_all_features, (bool, np.bool_)):
        raise ValueError(
            "pathline.save_all_features must be true or false, got "
            f"{save_all_features!r}.")
    pcfg["save_all_features"] = bool(save_all_features)
    if not pcfg["enabled"]:
        raise ValueError("NoiseMVUNet pathline generation cannot be disabled.")
    if pcfg["sampling"] != clean_generator.FEATURE_ADAPTIVE_MESH_SEEDING:
        raise ValueError(
            "NoiseMVUNet pathlines require FEATURE_ADAPTIVE_MESH_SEEDING.")
    if not pcfg["save_all_features"]:
        raise ValueError(
            "NoiseMVUNet pathlines require save_all_features: true.")
    feature_names = _pathline_feature_names(pcfg)
    logging.info(
        "[noise] pathline feature set: %d channels (%s)",
        len(feature_names),
        "extended" if pcfg["save_all_features"] else "legacy",
    )
    return pcfg


def _pathline_feature_names(pcfg):
    """Return the ordered feature names selected for persisted pathlines."""
    if bool(pcfg.get("save_all_features", True)):
        return EXTENDED_PATHLINE_FEATURE_NAMES
    return LEGACY_PATHLINE_FEATURE_NAMES


def _noise_sigmas(field, noise_cfg):
    velocity = np.asarray(field, dtype=np.float64)
    levels = np.asarray(
        [noise_cfg["level_u"], noise_cfg["level_v"]], dtype=np.float64)
    if noise_cfg["scale"] == "absolute":
        reference = np.ones(2, dtype=np.float64)
    elif noise_cfg["scale"] == "velocity_rms":
        speed_rms = float(np.sqrt(np.mean(np.sum(velocity * velocity, axis=-1))))
        reference = np.full(2, speed_rms, dtype=np.float64)
    else:
        reference = np.sqrt(np.mean(velocity * velocity, axis=(0, 1, 2)))
    return levels * reference


def add_gaussian_noise(field, noise_cfg, sample_index=0):
    """Add deterministic, statistically independent Gaussian noise to u and v."""
    clean = np.asarray(field, dtype=np.float32)
    if clean.ndim != 4 or clean.shape[-1] != 2:
        raise ValueError(
            "Noise input must be shaped [T,Y,X,2], got "
            f"{clean.shape}.")
    sigmas = _noise_sigmas(clean, noise_cfg)
    rng = np.random.default_rng(
        int(noise_cfg["seed"])
        + int(noise_cfg["seed_offset"])
        + int(sample_index)
    )
    noisy = clean.copy()
    noisy[..., 0] += rng.normal(
        float(noise_cfg["mean_u"]), float(sigmas[0]),
        size=clean.shape[:-1]).astype(np.float32)
    noisy[..., 1] += rng.normal(
        float(noise_cfg["mean_v"]), float(sigmas[1]),
        size=clean.shape[:-1]).astype(np.float32)
    if not np.all(np.isfinite(noisy)):
        raise ValueError("Gaussian noise produced non-finite velocity values.")
    return noisy, sigmas.astype(np.float64, copy=False)


def _bilinear_sample(data, points):
    """Sample one [Y,X,...] array at normalized points in [-1,1]^2."""
    array = np.asarray(data)
    points = np.asarray(points, dtype=np.float64)
    original_shape = points.shape[:-1]
    flat = points.reshape(-1, 2)
    height, width = array.shape[:2]
    x = np.clip(0.5 * (flat[:, 0] + 1.0) * (width - 1), 0.0, width - 1.0)
    y = np.clip(0.5 * (flat[:, 1] + 1.0) * (height - 1), 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    tail = array.shape[2:]
    weight_shape = (len(flat),) + (1,) * len(tail)
    wx = (x - x0).reshape(weight_shape)
    wy = (y - y0).reshape(weight_shape)
    top = array[y0, x0] * (1.0 - wx) + array[y0, x1] * wx
    bottom = array[y1, x0] * (1.0 - wx) + array[y1, x1] * wx
    sampled = top * (1.0 - wy) + bottom * wy
    return sampled.reshape(original_shape + tail)


class NoisyGridVelocityEvaluator:
    """Space-time interpolator plus finite-difference noisy-grid diagnostics."""

    def __init__(self, field, tmin=0.0, tmax=1.0):
        field = np.ascontiguousarray(field, dtype=np.float32)
        if field.ndim != 4 or field.shape[-1] != 2:
            raise ValueError(
                "Noisy grid must be shaped [T,Y,X,2], got "
                f"{field.shape}.")
        time_steps, height, width, _ = field.shape
        if height < 2 or width < 2:
            raise ValueError("Noisy grid needs at least 2x2 spatial samples.")
        self.field = field
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.time_steps = int(time_steps)

        dx = 2.0 / float(width - 1)
        dy = 2.0 / float(height - 1)
        edge_order = 2 if min(height, width) >= 3 else 1
        u = field[..., 0].astype(np.float64)
        v = field[..., 1].astype(np.float64)
        du_dy = np.gradient(u, dy, axis=1, edge_order=edge_order)
        du_dx = np.gradient(u, dx, axis=2, edge_order=edge_order)
        dv_dy = np.gradient(v, dy, axis=1, edge_order=edge_order)
        dv_dx = np.gradient(v, dx, axis=2, edge_order=edge_order)

        self.gradient = np.empty(
            (time_steps, height, width, 2, 2), dtype=np.float32)
        self.gradient[..., 0, 0] = du_dx
        self.gradient[..., 0, 1] = du_dy
        self.gradient[..., 1, 0] = dv_dx
        self.gradient[..., 1, 1] = dv_dy
        self.vorticity = np.asarray(dv_dx - du_dy, dtype=np.float32)
        mean_vorticity = np.mean(
            self.vorticity, axis=(1, 2), dtype=np.float64)
        self.mean_vorticity = mean_vorticity.astype(np.float32)
        self.ivd = np.abs(
            self.vorticity - self.mean_vorticity[:, None, None]
        ).astype(np.float32)

        # The inherited samplers look for ``torch_eval`` before requesting
        # vorticity.  Point it back to this finite-difference object; the
        # patched vorticity dispatcher below never invokes torch/autograd.
        self.torch_eval = self

    def _time_indices(self, time):
        if self.time_steps == 1 or self.tmax == self.tmin:
            return 0, 0, 0.0
        floating = (
            (float(time) - self.tmin)
            / (self.tmax - self.tmin)
            * (self.time_steps - 1)
        )
        floating = float(np.clip(floating, 0.0, self.time_steps - 1.0))
        lower = int(np.floor(floating))
        upper = min(lower + 1, self.time_steps - 1)
        return lower, upper, floating - lower

    def _sample(self, data, points, time):
        lower, upper, ratio = self._time_indices(time)
        first = _bilinear_sample(data[lower], points)
        if upper == lower:
            return first
        second = _bilinear_sample(data[upper], points)
        return first * (1.0 - ratio) + second * ratio

    def __call__(self, points, time):
        return np.asarray(
            self._sample(self.field, points, time), dtype=np.float64)

    def sample_vorticity(self, points, time):
        return np.asarray(
            self._sample(self.vorticity, points, time), dtype=np.float64)

    def sample_ivd(self, points, time):
        # Interpolate signed vorticity and its spatial mean first, then take
        # the absolute deviation.  Interpolating the already-absolute IVD
        # frames would be inconsistent at sign changes between time samples.
        vorticity = self.sample_vorticity(points, time)
        lower, upper, ratio = self._time_indices(time)
        mean_vorticity = (
            float(self.mean_vorticity[lower]) * (1.0 - ratio)
            + float(self.mean_vorticity[upper]) * ratio)
        return np.abs(vorticity - mean_vorticity)

    def sample_gradient(self, points, time):
        return np.asarray(
            self._sample(self.gradient, points, time), dtype=np.float64)


def _finite_difference_vorticity_at_points(velocity_eval, points, time):
    """Replacement for the clean generator's autograd vorticity helper."""
    sampler = getattr(velocity_eval, "sample_vorticity", None)
    if sampler is None:
        raise ValueError(
            "Noise generation requires a NoisyGridVelocityEvaluator; "
            "analytical/autograd vorticity is disabled.")
    return sampler(points, time)


def _pathline_lavd(ivd, valid_lengths, times):
    lavd = np.zeros_like(ivd, dtype=np.float64)
    for line_index, valid_length in enumerate(valid_lengths):
        for step in range(1, int(valid_length)):
            dt = float(times[step] - times[step - 1])
            lavd[line_index, step] = (
                lavd[line_index, step - 1]
                + 0.5 * (ivd[line_index, step - 1] + ivd[line_index, step]) * dt
            )
    return lavd


def collect_pathline_info(
        unsteady_eval, _sampled_field, label_eval, pcfg, ocfg, rng):
    """Integrate and describe paths exclusively from the noisy sampled grid."""
    if not isinstance(unsteady_eval, NoisyGridVelocityEvaluator):
        raise TypeError(
            "Noise pathlines must use NoisyGridVelocityEvaluator, not an "
            "analytical Vatistas callable.")

    length = int(pcfg["length"])
    seeds = clean_generator._pathline_seedings(
        pcfg, rng, velocity_eval=unsteady_eval, time=ocfg["tmin"])
    positions, valid_lengths, times = clean_generator._integrate_pathline_positions(
        unsteady_eval, seeds, ocfg["tmin"], ocfg["tmax"], length)

    for _ in range(int(pcfg["max_reseed_attempts"])):
        failed = valid_lengths <= 2
        if not np.any(failed):
            break
        shifts = rng.uniform(1e-5, 0.5, int(failed.sum()))
        seeds[failed] *= 1.0 - shifts[:, None]
        positions, valid_lengths, times = clean_generator._integrate_pathline_positions(
            unsteady_eval, seeds, ocfg["tmin"], ocfg["tmax"], length)

    padding = float(pcfg["padding_value"])
    save_all_features = bool(pcfg.get("save_all_features", True))
    feature_count = len(_pathline_feature_names(pcfg))
    features = np.full(
        (len(seeds), length, feature_count), padding, dtype=np.float32)
    valid = np.isfinite(positions[..., 0])
    line_ids, step_ids = np.nonzero(valid)
    valid_points = positions[line_ids, step_ids]
    valid_times = times[step_ids]

    velocities = np.empty_like(valid_points)
    ivd_values = np.empty(len(valid_points), dtype=np.float64)
    if save_all_features:
        vorticity = np.empty(len(valid_points), dtype=np.float64)
        gradients = np.empty((len(valid_points), 2, 2), dtype=np.float64)
    for step in np.unique(step_ids):
        at_step = step_ids == step
        time = float(times[step])
        points_at_step = valid_points[at_step]
        velocities[at_step] = unsteady_eval(points_at_step, time)
        ivd_values[at_step] = unsteady_eval.sample_ivd(points_at_step, time)
        if save_all_features:
            vorticity[at_step] = unsteady_eval.sample_vorticity(
                points_at_step, time)
            gradients[at_step] = unsteady_eval.sample_gradient(
                points_at_step, time)

    distance = np.linalg.norm(valid_points - seeds[line_ids], axis=1)

    features[line_ids, step_ids, 0:2] = valid_points
    features[line_ids, step_ids, 2] = valid_times
    features[line_ids, step_ids, 3] = ivd_values
    features[line_ids, step_ids, 4] = distance
    features[line_ids, step_ids, 5:7] = velocities
    if save_all_features:
        du_dx = gradients[:, 0, 0]
        du_dy = gradients[:, 0, 1]
        dv_dx = gradients[:, 1, 0]
        dv_dy = gradients[:, 1, 1]
        divergence = du_dx + dv_dy
        strain_normal = du_dx - dv_dy
        strain_shear = du_dy + dv_dx
        strain_magnitude = np.sqrt(
            strain_normal * strain_normal + strain_shear * strain_shear)
        determinant = du_dx * dv_dy - du_dy * dv_dx
        discriminant = divergence * divergence - 4.0 * determinant
        swirl_strength = 0.5 * np.sqrt(np.maximum(-discriminant, 0.0))
        symmetric_norm = (
            du_dx * du_dx + dv_dy * dv_dy
            + 0.5 * strain_shear * strain_shear)
        antisymmetric_norm = 0.5 * vorticity * vorticity
        q_criterion = 0.5 * (antisymmetric_norm - symmetric_norm)

        ivd_by_path = np.zeros((len(seeds), length), dtype=np.float64)
        ivd_by_path[line_ids, step_ids] = ivd_values
        lavd = _pathline_lavd(ivd_by_path, valid_lengths, times)
        features[line_ids, step_ids, 7] = lavd[line_ids, step_ids]
        features[line_ids, step_ids, 8] = vorticity
        features[line_ids, step_ids, 9] = du_dx
        features[line_ids, step_ids, 10] = du_dy
        features[line_ids, step_ids, 11] = dv_dx
        features[line_ids, step_ids, 12] = dv_dy
        features[line_ids, step_ids, 13] = divergence
        features[line_ids, step_ids, 14] = strain_magnitude
        features[line_ids, step_ids, 15] = determinant
        features[line_ids, step_ids, 16] = swirl_strength
        features[line_ids, step_ids, 17] = q_criterion

    # Labels deliberately bypass the noisy evaluator.
    seed_sd = label_eval(seeds.reshape(-1, 1, 2)).reshape(-1)
    seed_labels = (seed_sd > 0.0).astype(np.float32)
    features[:, 0, 4] = seed_labels
    return features, seed_labels, valid_lengths


def read_observer_cfg(cfg):
    ocfg = _ORIGINAL_READ_OBSERVER_CFG(cfg)
    noise_cfg = read_noise_cfg(cfg)
    ocfg["noise"] = noise_cfg
    _RUNTIME["noise"] = noise_cfg
    logging.info(
        "[noise] levels u=%.6g, v=%.6g; scale=%s; seed=%d+%d",
        noise_cfg["level_u"], noise_cfg["level_v"], noise_cfg["scale"],
        noise_cfg["seed"], noise_cfg["seed_offset"])
    return ocfg


def tobias_observer_transformation(
        vel_eval, abc, abc_dot, ocfg, coords_np, return_evaluator=False):
    """Sample the clean analytical field, add noise, then expose only its grid."""
    clean_field = _ORIGINAL_TOBIAS_TRANSFORMATION(
        vel_eval, abc, abc_dot, ocfg, coords_np, return_evaluator=False)
    sample_index = int(_RUNTIME["unsteady_noise_index"])
    _RUNTIME["unsteady_noise_index"] = sample_index + 1
    noisy_field, sigmas = add_gaussian_noise(
        clean_field, ocfg["noise"], sample_index=sample_index)
    evaluator = NoisyGridVelocityEvaluator(
        noisy_field, tmin=ocfg["tmin"], tmax=ocfg["tmax"])
    evaluator.noise_sigma = sigmas
    return (noisy_field, evaluator) if return_evaluator else noisy_field


def generate_unsteady(*args, **kwargs):
    result = _ORIGINAL_GENERATE_UNSTEADY(*args, **kwargs)
    target_total, written, observers, stats = result
    if stats["pathline_count"]:
        pcfg = args[6] if len(args) > 6 else kwargs["pcfg"]
        stats["pathline_feature"] = len(_pathline_feature_names(pcfg))
    return target_total, written, observers, stats


def _noisy_steady_fields(steady_fields):
    noise_cfg = _RUNTIME.get("noise")
    if noise_cfg is None:
        raise RuntimeError("Noise configuration has not been initialized.")
    noisy = np.empty_like(steady_fields, dtype=np.float32)
    for index, field in enumerate(steady_fields):
        noisy_field, _ = add_gaussian_noise(
            field[None], noise_cfg, sample_index=10_000_000 + index)
        noisy[index] = noisy_field[0]
    return noisy


def save_steady(out_dir, specs, grid, fields, labels, source, counts):
    """Save noisy steady velocities with unchanged clean analytical labels."""
    return _ORIGINAL_SAVE_STEADY(
        out_dir, specs, grid, _noisy_steady_fields(fields), labels, source,
        counts)


def save_steady_lic_images(
        steady_fields, _velocity_evals, label_evals, observers_per_field,
        out_dir, lcfg, pcfg, ocfg, seed):
    if not lcfg["enabled"]:
        return 0, 0
    noisy_fields = _noisy_steady_fields(steady_fields)
    noisy_evals = [
        NoisyGridVelocityEvaluator(
            field[None], tmin=ocfg["tmin"], tmax=ocfg["tmin"])
        for field in noisy_fields
    ]
    return _ORIGINAL_SAVE_STEADY_LIC_IMAGES(
        noisy_fields, noisy_evals, label_evals, observers_per_field,
        out_dir, lcfg, pcfg, ocfg, seed)


def _noise_metadata(ocfg):
    noise = ocfg["noise"]
    return {
        "noiseDistribution": "independent_gaussian_per_velocity_component",
        "noiseLevel": {
            "u": float(noise["level_u"]),
            "v": float(noise["level_v"]),
        },
        "noiseMean": {
            "u": float(noise["mean_u"]),
            "v": float(noise["mean_v"]),
        },
        "noiseScale": noise["scale"],
        "noiseSeed": int(noise["seed"]),
        "noiseSeedOffset": int(noise["seed_offset"]),
        "generationOrder": [
            "analytical_vatistas_tobias_velocity",
            "sample_to_discrete_grid",
            "independent_gaussian_noise_on_u_and_v",
            "finite_difference_noisy_grid_diagnostics",
            "adaptive_mesh_and_noisy_grid_pathline_integration",
        ],
        "velocityEvaluation": "bilinear_space_linear_time_from_noisy_grid",
        "pathlineIntegrationVelocity": (
            "rk4_bilinear_space_linear_time_noisy_grid"),
        "spatialDerivativeMethod": (
            "numpy_gradient_second_order_finite_difference_noisy_grid"),
        "vorticityMonitorSource": "finite_difference_noisy_grid_vorticity",
        "labels": "clean_analytical_vatistas_signed_distance",
    }


def _save_dataset_meta(out_dir, ocfg, pcfg, stats, target_total, written):
    _ORIGINAL_SAVE_DATASET_META(
        out_dir, ocfg, pcfg, stats, target_total, written)
    metadata = _noise_metadata(ocfg)
    feature_names = _pathline_feature_names(pcfg)
    channel_min = np.asarray(
        stats.get("velocity_bin_channel_min") or (), dtype=np.float64)
    channel_max = np.asarray(
        stats.get("velocity_bin_channel_max") or (), dtype=np.float64)
    if channel_min.shape != (4,) or channel_max.shape != (4,):
        raise ValueError(
            "NoiseMVUNet metadata requires four finite MVUNet channel bounds.")
    if not np.all(np.isfinite(channel_min)) or not np.all(np.isfinite(channel_max)):
        raise ValueError("NoiseMVUNet channel bounds must be finite.")
    normalization_min = float(np.min(channel_min))
    normalization_max = float(np.max(channel_max))
    metadata.update({
        # MVUNet's scalar MinMaxNormalization is applied to all four channels.
        "minV": normalization_min,
        "maxV": normalization_max,
        "mvunetNormalizationRange": {
            "min": normalization_min,
            "max": normalization_max,
            "source": "global_minmax_over_saved_noisy_u_v_curl_ivd",
        },
        "PathlineFeature": int(stats["pathline_feature"]),
        "pathlineFeatureOrder": (
            list(feature_names) if pcfg["enabled"] else []),
        "pathlineFeatureSet": (
            "extended_18" if pcfg.get("save_all_features", True)
            else "legacy_7"),
        "saveAllPathlineFeatures": bool(
            pcfg.get("save_all_features", True)),
        "ivdMethod": (
            "finite_difference_vorticity_spatial_grid_mean_noisy_grid"
            if pcfg["enabled"] else None),
        "pathlineDescriptorMethod": (
            "finite_difference_gradient_interpolated_from_noisy_grid"
            if pcfg["enabled"] and pcfg.get("save_all_features", True)
            else None),
        "mvunetDenseFeatureMethod": (
            "finite_difference_from_noisy_velocity_frame_0"),
    })
    meta_paths = [
        os.path.join(out_dir, "meta.json"),
        os.path.join(out_dir, "velocity_mix", "meta.json"),
    ]
    if stats.get("velocity_bin_dir"):
        meta_paths.append(os.path.join(
            out_dir, stats["velocity_bin_dir"], "meta.json"))
    if stats.get("pathline_bin_dir"):
        meta_paths.append(os.path.join(
            out_dir, stats["pathline_bin_dir"], "meta.json"))
    root_meta_path = os.path.join(out_dir, "meta.json")
    with open(root_meta_path, "r", encoding="utf-8") as handle:
        root_payload = json.load(handle)
    for meta_path in dict.fromkeys(meta_paths):
        # Base metadata is identical across its output folders.  Start from the
        # root copy so the dedicated noisy_pathlines directory also receives a
        # self-describing meta.json even when per-sample JSON is disabled.
        payload = json.loads(json.dumps(root_payload))
        payload.update(metadata)
        if payload.get("pathlineVorticityImportance") is not None:
            payload["pathlineVorticityImportance"]["criterion"] = (
                "absolute_vorticity_finite_difference_noisy_grid")
        if payload.get("pathlineFeatureAdaptiveMesh") is not None:
            payload["pathlineFeatureAdaptiveMesh"]["criterion"] = (
                "absolute_vorticity_finite_difference_noisy_grid")
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    manifest_path = os.path.join(out_dir, "dataset_manifest.npz")
    if os.path.isfile(manifest_path):
        with np.load(manifest_path, allow_pickle=False) as archive:
            manifest = {key: archive[key] for key in archive.files}
        manifest.update({
            "noise_level": np.asarray(
                [ocfg["noise"]["level_u"], ocfg["noise"]["level_v"]],
                dtype=np.float32),
            "noise_scale": np.asarray(ocfg["noise"]["scale"]),
            "pathline_feature_order": np.asarray(
                feature_names if pcfg["enabled"] else ()),
            "pathline_feature_set": np.asarray(
                "extended_18" if pcfg.get("save_all_features", True)
                else "legacy_7"),
            "save_all_pathline_features": np.asarray(
                bool(pcfg.get("save_all_features", True))),
            "derivative_method": np.asarray(
                "finite_difference_noisy_grid"),
            "mvunet_normalization_min": np.float32(normalization_min),
            "mvunet_normalization_max": np.float32(normalization_max),
        })
        np.savez_compressed(manifest_path, **manifest)


def install_noise_mvunet_pipeline():
    """Patch the clean orchestration so no post-noise analytic path remains."""
    _RUNTIME["unsteady_noise_index"] = 0
    clean_generator.read_observer_cfg = read_observer_cfg
    clean_generator.read_pathline_cfg = read_pathline_cfg
    clean_generator.tobias_observer_transformation = (
        tobias_observer_transformation)
    clean_generator._autograd_vorticity_at_points = (
        _finite_difference_vorticity_at_points)
    clean_generator.collect_pathline_info = collect_pathline_info
    clean_generator.generate_unsteady = generate_unsteady
    clean_generator.save_steady = save_steady
    clean_generator.save_steady_lic_images = save_steady_lic_images
    clean_generator._save_dataset_meta = _save_dataset_meta


# Public compatibility alias for small validation scripts.
install_noise_pipeline = install_noise_mvunet_pipeline


def main():
    has_config = any(
        arg == "--config" or arg.startswith("--config=")
        for arg in sys.argv[1:])
    if not has_config:
        sys.argv[1:1] = ["--config", DEFAULT_CONFIG_PATH]

    # Validate before fitting/rendering thousands of fields.  The clean entry
    # point parses the same arguments again after the noisy hooks are installed.
    config_path = _config_path_from_argv(sys.argv[1:])
    cfg = clean_generator.EasyConfig()
    cfg.load(config_path, recursive=False)
    validate_mvunet_config(cfg)

    install_noise_mvunet_pipeline()
    logging.info(
        "[noise-mvunet] noisy-grid finite-difference pipeline installed; "
        "analytical velocity is used only to create the clean sampled grid, "
        "and analytical labels remain noise-free.")
    clean_generator.main()


if __name__ == "__main__":
    main()
