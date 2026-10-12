"""Generate Vatistas data whose physics is derived only from a noisy grid.

The clean generator remains the source of Vatistas parameters, analytical
velocity samples, observer motion, and analytical vortex labels.  This module
adds independent Gaussian noise to the sampled ``u`` and ``v`` arrays and then
switches every downstream operation to :class:`NoisyGridVelocityEvaluator`:

* pathline RK4 stages use space-time interpolation of the noisy velocity grid;
* velocity gradients and vorticity use finite differences on that grid;
* IVD, the vorticity monitor, adaptive mesh, and pathline descriptors use those
  finite-difference quantities; and
* segmentation and seed labels still use the clean analytical Vatistas label.

Run:
    python VatistasFlowDatasetGeneratorNoise.py
    python VatistasFlowDatasetGeneratorNoise.py \
        --config config/VatistasDatasetNoise.yaml
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


def read_pathline_cfg(cfg):
    """Add the noisy generator's 7/18-channel choice to the base config."""
    pcfg = _ORIGINAL_READ_PATHLINE_CFG(cfg)
    section = cfg.get("pathline", {}) or {}
    save_all_features = section.get("save_all_features", True)
    if not isinstance(save_all_features, (bool, np.bool_)):
        raise ValueError(
            "pathline.save_all_features must be true or false, got "
            f"{save_all_features!r}.")
    pcfg["save_all_features"] = bool(save_all_features)
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
        return np.asarray(
            self._sample(self.ivd, points, time), dtype=np.float64)

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
        "velocityEvaluation": "bilinear_space_linear_time_from_noisy_grid",
        "spatialDerivativeMethod": "numpy_gradient_finite_difference_noisy_grid",
        "labels": "clean_analytical_vatistas_signed_distance",
    }


def _save_dataset_meta(out_dir, ocfg, pcfg, stats, target_total, written):
    _ORIGINAL_SAVE_DATASET_META(
        out_dir, ocfg, pcfg, stats, target_total, written)
    metadata = _noise_metadata(ocfg)
    feature_names = _pathline_feature_names(pcfg)
    metadata.update({
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
    })
    meta_paths = [
        os.path.join(out_dir, "meta.json"),
        os.path.join(out_dir, "velocity_mix", "meta.json"),
    ]
    if stats.get("velocity_bin_dir"):
        meta_paths.append(os.path.join(
            out_dir, stats["velocity_bin_dir"], "meta.json"))
    for meta_path in dict.fromkeys(meta_paths):
        if not os.path.isfile(meta_path):
            continue
        with open(meta_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.update(metadata)
        if payload.get("pathlineVorticityImportance") is not None:
            payload["pathlineVorticityImportance"]["criterion"] = (
                "absolute_vorticity_finite_difference_noisy_grid")
        if payload.get("pathlineFeatureAdaptiveMesh") is not None:
            payload["pathlineFeatureAdaptiveMesh"]["criterion"] = (
                "absolute_vorticity_finite_difference_noisy_grid")
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
        })
        np.savez_compressed(manifest_path, **manifest)


def install_noise_pipeline():
    """Patch only the orchestration hooks needed by the clean generator."""
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


def main():
    install_noise_pipeline()
    has_config = any(
        arg == "--config" or arg.startswith("--config=")
        for arg in sys.argv[1:])
    if not has_config:
        default_config = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "config", "VatistasDatasetNoise.yaml")
        sys.argv[1:1] = ["--config", default_config]
    logging.info(
        "[noise] noisy-grid finite-difference pipeline installed; analytical "
        "velocity is used only before grid noise is added.")
    clean_generator.main()


if __name__ == "__main__":
    main()
