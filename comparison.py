#!/usr/bin/env python3
"""Compare eight vortex detectors on the cylinder wake.

All eight methods share the same ``cylinder2d.nc`` snapshot and spatial grid,
while each learned model keeps the preprocessing contract for which its
checkpoint was trained:

* IVD, Q, two-dimensional Delta, and lambda-2 are evaluated
  instantaneously at ``--time-index``;
* V0 uses ``detect.py``'s recovered ``legacy-grid`` adapter, including regular
  16-by-16 seeds, legacy IVD, and the default 0.72 physical-time horizon;
* V1 uses ``detect_test.py``'s ``FEATURE_ADAPTIVE_MESH_SEEDING`` adapter and
  the default normalized one-unit physical-time horizon.
* MVUNet uses the Vatistas generator's dense t0 ``[u,v,curl,IVD]`` input;
  its paired training data used ``GRID_CROSS_SEEDING``.
* VortexUNet uses the generator's dense t0 ``[u,v]`` velocity grid; its
  paired training data also used ``GRID_CROSS_SEEDING``.

The two-dimensional Delta field is the signed complex-eigenvalue
discriminant ``det(J) - trace(J)**2 / 4``.  This is the meaningful planar
specialization of the Delta criterion; inserting a 2-by-2 determinant into
the three-dimensional cubic formula gives dimensionally inconsistent and
nearly full-domain detections.

All learned-model panels use one fixed probability scale from zero to one. The
script also writes per-seed probabilities for V0/V1 and a combined detection
CSV so the visual comparison can be audited numerically.

Example
-------
MPLCONFIGDIR=/tmp/pyflowvis-mpl .venv/bin/python comparison.py \
    --input dataset/cylinder2d.nc --time-index 1000 --device cpu
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import numpy as np

import detect as v0_common
import detect_test as v1_common
import comparison_plot_style as plot_style
import mvunet_comparison as mvunet
import vortexunet_comparison as vortexunet

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter, label


@dataclass
class ModelDetectionResult:
    """Projected predictions and metadata for one learned detector."""

    label: str
    preprocessing: str
    epoch: int
    duration: float
    patch_count: int
    threshold: float
    smooth_sigma: float
    min_component_pixels: int
    points: np.ndarray
    scores: np.ndarray
    source_patches: np.ndarray
    probability: np.ndarray
    support: np.ndarray
    detections: list[v0_common.RegionDetection]


@dataclass(frozen=True)
class EulerianCriterionResult:
    """One instantaneous velocity-gradient criterion and its regions."""

    label: str
    symbol: str
    prefix: str
    definition: str
    values: np.ndarray
    threshold: float
    lower_is_vortex: bool
    smooth_sigma: float
    min_component_pixels: int
    detections: list[v0_common.RegionDetection]

    @property
    def decision_rule(self) -> str:
        operator = "<" if self.lower_is_vortex else ">"
        return f"{self.symbol} {operator} {self.threshold:.8g}"


def compute_velocity_gradient_criteria(
    snapshot_u: np.ndarray,
    snapshot_v: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Q, planar Delta, and lambda-2 from a 2-D velocity snapshot.

    ``J = [[du/dx, du/dy], [dv/dx, dv/dy]]`` is decomposed as
    ``J = S + Omega``.  Lambda-2 is the larger eigenvalue of
    ``S @ S + Omega @ Omega`` in this planar analogue, so its vortex
    condition remains ``lambda_2 < 0`` (two negative eigenvalues).
    """

    snapshot_u = np.asarray(snapshot_u, dtype=np.float64)
    snapshot_v = np.asarray(snapshot_v, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    expected_shape = (len(y), len(x))
    if snapshot_u.shape != expected_shape or snapshot_v.shape != expected_shape:
        raise ValueError(
            "Velocity snapshots must both have shape "
            f"{expected_shape}; got {snapshot_u.shape} and {snapshot_v.shape}."
        )
    if len(x) < 3 or len(y) < 3:
        raise ValueError("At least three grid points per axis are required.")

    du_dx = np.gradient(snapshot_u, x, axis=1, edge_order=2)
    du_dy = np.gradient(snapshot_u, y, axis=0, edge_order=2)
    dv_dx = np.gradient(snapshot_v, x, axis=1, edge_order=2)
    dv_dy = np.gradient(snapshot_v, y, axis=0, edge_order=2)

    symmetric_xy = 0.5 * (du_dy + dv_dx)
    rotation_xy = 0.5 * (du_dy - dv_dx)
    strain_norm_squared = (
        du_dx * du_dx
        + dv_dy * dv_dy
        + 2.0 * symmetric_xy * symmetric_xy
    )
    rotation_norm_squared = 2.0 * rotation_xy * rotation_xy
    q_criterion = 0.5 * (rotation_norm_squared - strain_norm_squared)

    jacobian_trace = du_dx + dv_dy
    jacobian_determinant = du_dx * dv_dy - du_dy * dv_dx
    delta_2d = jacobian_determinant - 0.25 * jacobian_trace**2

    # A = S^2 + Omega^2 is symmetric.  Its larger eigenvalue is obtained
    # analytically to avoid constructing one 2-by-2 matrix per grid cell.
    a11 = du_dx**2 + symmetric_xy**2 - rotation_xy**2
    a22 = dv_dy**2 + symmetric_xy**2 - rotation_xy**2
    a12 = symmetric_xy * (du_dx + dv_dy)
    half_trace = 0.5 * (a11 + a22)
    half_gap = np.sqrt(np.maximum(0.0, 0.25 * (a11 - a22) ** 2 + a12**2))
    lambda_2 = half_trace + half_gap

    return q_criterion, delta_2d, lambda_2


def _classic_valid_mask(
    flow: v0_common.FlowWindow,
    fields: Sequence[np.ndarray],
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
    obstacle_margin: float,
    border_cells: int,
) -> np.ndarray:
    """Build one common fluid mask for all velocity-gradient criteria."""

    valid = np.isfinite(np.asarray(flow.vorticity[0], dtype=np.float64))
    for field in fields:
        valid &= np.isfinite(field)

    if obstacle_radius > 0.0:
        grid_x, grid_y = np.meshgrid(flow.x, flow.y)
        distance = np.hypot(
            grid_x - obstacle_center[0], grid_y - obstacle_center[1]
        )
        valid &= distance > obstacle_margin * obstacle_radius

    if border_cells:
        if 2 * border_cells >= min(valid.shape):
            raise ValueError(
                "--classic-border-cells removes the entire flow grid."
            )
        valid[:border_cells, :] = False
        valid[-border_cells:, :] = False
        valid[:, :border_cells] = False
        valid[:, -border_cells:] = False
    if not np.any(valid):
        raise ValueError(
            "No valid fluid cells remain for velocity-gradient criteria."
        )
    return valid


def _detect_eulerian_regions(
    values: np.ndarray,
    threshold: float,
    lower_is_vortex: bool,
    valid: np.ndarray,
    vorticity: np.ndarray,
    mean_vorticity: float,
    x: np.ndarray,
    y: np.ndarray,
    min_pixels: int,
) -> list[v0_common.RegionDetection]:
    """Threshold a criterion and compute rotation-aware weighted centroids."""

    active = values < threshold if lower_is_vortex else values > threshold
    detections: list[v0_common.RegionDetection] = []
    for rotation_sign, sign_mask in (
        (1, vorticity >= mean_vorticity),
        (-1, vorticity < mean_vorticity),
    ):
        components, component_count = label(valid & sign_mask & active)
        for component_id in range(1, component_count + 1):
            yy, xx = np.nonzero(components == component_id)
            if len(xx) < min_pixels:
                continue
            component_values = values[yy, xx]
            if lower_is_vortex:
                weights = np.maximum(threshold - component_values, 1e-12)
                peak = float(np.min(component_values))
            else:
                weights = np.maximum(component_values - threshold, 1e-12)
                peak = float(np.max(component_values))
            detections.append(
                v0_common.RegionDetection(
                    x=float(np.average(x[xx], weights=weights)),
                    y=float(np.average(y[yy], weights=weights)),
                    peak=peak,
                    pixels=len(xx),
                    rotation_sign=rotation_sign,
                )
            )
    detections.sort(key=lambda item: (item.x, item.y))
    return detections


def build_eulerian_criteria(
    flow: v0_common.FlowWindow,
    snapshot_u: np.ndarray,
    snapshot_v: np.ndarray,
    args: argparse.Namespace,
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
) -> list[EulerianCriterionResult]:
    """Compute, mask, and detect Q/Delta/lambda-2 regions consistently."""

    raw_fields = compute_velocity_gradient_criteria(
        snapshot_u, snapshot_v, flow.x, flow.y
    )
    if args.classic_smooth_sigma > 0.0:
        fields = tuple(
            gaussian_filter(
                field,
                sigma=args.classic_smooth_sigma,
                mode="nearest",
            )
            for field in raw_fields
        )
    else:
        fields = raw_fields

    valid = _classic_valid_mask(
        flow,
        fields,
        obstacle_center,
        obstacle_radius,
        args.classic_obstacle_margin,
        args.classic_border_cells,
    )
    vorticity = np.asarray(flow.vorticity[0], dtype=np.float64)
    mean_vorticity = float(np.mean(vorticity[valid]))
    specifications = (
        (
            "Q criterion",
            "Q",
            "Q",
            r"$Q=\frac{1}{2}(\Vert\Omega\Vert_F^2-\Vert S\Vert_F^2)$",
            fields[0],
            args.q_threshold,
            False,
        ),
        (
            "Delta criterion (2-D)",
            "Delta_2D",
            "D",
            r"$\Delta_{2D}=\det J-\frac{1}{4}(\mathrm{tr}\,J)^2$",
            fields[1],
            args.delta_threshold,
            False,
        ),
        (
            "lambda2 criterion",
            "lambda_2",
            "L",
            r"$\lambda_2(S^2+\Omega^2)$ (larger planar eigenvalue)",
            fields[2],
            args.lambda2_threshold,
            True,
        ),
    )

    results: list[EulerianCriterionResult] = []
    for (
        label_text,
        symbol,
        prefix,
        definition,
        field,
        threshold,
        lower,
    ) in specifications:
        detections = _detect_eulerian_regions(
            field,
            threshold,
            lower,
            valid,
            vorticity,
            mean_vorticity,
            flow.x,
            flow.y,
            args.classic_min_component_pixels,
        )
        results.append(
            EulerianCriterionResult(
                label=label_text,
                symbol=symbol,
                prefix=prefix,
                definition=definition,
                values=np.where(valid, field, np.nan),
                threshold=threshold,
                lower_is_vortex=lower,
                smooth_sigma=args.classic_smooth_sigma,
                min_component_pixels=args.classic_min_component_pixels,
                detections=detections,
            )
        )
    return results


def _project_predictions(
    model,
    prepared: v0_common.PreparedInputs,
    flow: v0_common.FlowWindow,
    device: str,
    batch_size: int,
    threshold: float,
    smooth_sigma: float,
    min_component_pixels: int,
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[v0_common.RegionDetection],
]:
    """Run the common model-output projection and connected components."""

    score_blocks = v0_common.predict_batches(
        model, prepared.features, device, batch_size
    )
    points, scores, source_patches = v0_common.merge_duplicate_seeds(
        prepared.seeds, score_blocks, prepared.display_valid
    )
    probability, support = v0_common.build_probability_map(
        flow.x, flow.y, points, scores, smooth_sigma
    )
    v0_common.mask_obstacle(
        probability,
        support,
        flow.x,
        flow.y,
        obstacle_center,
        obstacle_radius,
    )
    detections = v0_common.detect_probability_regions(
        probability,
        support,
        flow.x,
        flow.y,
        threshold,
        min_component_pixels,
    )
    return points, scores, source_patches, probability, support, detections


def run_v0(
    flow: v0_common.FlowWindow,
    start_time: float,
    args: argparse.Namespace,
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
    device: str,
) -> ModelDetectionResult:
    """Run the V0 checkpoint through detect.py's legacy-grid contract."""

    prepared = v0_common.prepare_model_inputs(
        flow=flow,
        mode=v0_common.LEGACY_GRID_MODE,
        start_time=start_time,
        duration=args.v0_duration,
        obstacle_center=obstacle_center,
        obstacle_radius=obstacle_radius,
        patch_points=args.v0_patch_points,
        patch_stride_points=args.v0_patch_stride_points,
        tile_size=None,
        tile_stride=None,
        legacy_time_mode=v0_common.LEGACY_TIME_ANCHORED,
        model_time_start=args.v0_model_time_start,
    )
    model, epoch = v0_common.load_model(
        args.v0_checkpoint, args.v0_torchscript, device
    )
    (
        points,
        scores,
        source_patches,
        probability,
        support,
        detections,
    ) = _project_predictions(
        model,
        prepared,
        flow,
        device,
        args.v0_batch_size,
        args.v0_threshold,
        args.v0_smooth_sigma,
        args.v0_min_component_pixels,
        obstacle_center,
        obstacle_radius,
    )
    return ModelDetectionResult(
        label="Vortex Transformer",
        preprocessing="legacy-grid",
        epoch=epoch,
        duration=args.v0_duration,
        patch_count=len(prepared.patches),
        threshold=args.v0_threshold,
        smooth_sigma=args.v0_smooth_sigma,
        min_component_pixels=args.v0_min_component_pixels,
        points=points,
        scores=scores,
        source_patches=source_patches,
        probability=probability,
        support=support,
        detections=detections,
    )


def _v1_preparation_args(args: argparse.Namespace) -> SimpleNamespace:
    """Map comparison CLI names to detect_test.prepare_adaptive_inputs."""

    return SimpleNamespace(
        duration=args.v1_duration,
        seed=args.v1_seed,
        feature_high_fraction=args.v1_feature_high_fraction,
        feature_high_vorticity_percentile=(
            args.v1_feature_high_vorticity_percentile
        ),
        feature_mesh_resolution=args.v1_feature_mesh_resolution,
        feature_mesh_size_ratio=args.v1_feature_mesh_size_ratio,
        feature_monitor_power=args.v1_feature_monitor_power,
        feature_mesh_iterations=args.v1_feature_mesh_iterations,
        feature_clip_percentile=args.v1_feature_clip_percentile,
        feature_boundary_margin=args.v1_feature_boundary_margin,
        obstacle_seeding_margin=args.v1_obstacle_seeding_margin,
        max_reseed_attempts=args.v1_max_reseed_attempts,
    )


def run_v1(
    flow: v0_common.FlowWindow,
    start_time: float,
    args: argparse.Namespace,
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
    device: str,
) -> tuple[ModelDetectionResult, int, int, str]:
    """Run V1 with the feature-adaptive training-data convention."""

    tile_size = (
        float(flow.y[-1] - flow.y[0])
        if args.v1_tile_size is None
        else args.v1_tile_size
    )
    tile_stride = (
        0.5 * tile_size
        if args.v1_tile_stride is None
        else args.v1_tile_stride
    )
    patches = v0_common.make_cross_patches(
        flow.x, flow.y, tile_size, tile_stride
    )
    prepared, reseed_passes, invalid_t0_count = (
        v1_common.prepare_adaptive_inputs(
            flow,
            patches,
            _v1_preparation_args(args),
            start_time,
            obstacle_center,
            obstacle_radius,
        )
    )
    model, epoch, backend = v1_common.load_v1_model(
        args.v1_checkpoint,
        args.v1_torchscript,
        args.v1_config,
        device,
        args.v1_backend,
    )
    (
        points,
        scores,
        source_patches,
        probability,
        support,
        detections,
    ) = _project_predictions(
        model,
        prepared,
        flow,
        device,
        args.v1_batch_size,
        args.v1_threshold,
        args.v1_smooth_sigma,
        args.v1_min_component_pixels,
        obstacle_center,
        obstacle_radius,
    )
    result = ModelDetectionResult(
        label="FGMA-VT",
        preprocessing="feature-adaptive-mesh",
        epoch=epoch,
        duration=args.v1_duration,
        patch_count=len(prepared.patches),
        threshold=args.v1_threshold,
        smooth_sigma=args.v1_smooth_sigma,
        min_component_pixels=args.v1_min_component_pixels,
        points=points,
        scores=scores,
        source_patches=source_patches,
        probability=probability,
        support=support,
        detections=detections,
    )
    return result, reseed_passes, invalid_t0_count, backend


def save_comparison_summary(
    output_path: Path,
    ivd_threshold: float,
    ivd_smooth_sigma: float,
    ivd_min_component_pixels: int,
    ivd_detections: Sequence[v0_common.RegionDetection],
    eulerian_results: Sequence[EulerianCriterionResult],
    v0_result: ModelDetectionResult,
    v1_result: ModelDetectionResult,
    mvunet_result: ModelDetectionResult,
    vortexunet_result: ModelDetectionResult,
) -> None:
    """Write all eight methods' region detections into one long-form CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "id",
                "x",
                "y",
                "peak_value",
                "component_pixels",
                "rotation_sign",
                "decision_threshold",
                "integration_horizon",
                "smoothing_sigma",
                "min_component_pixels_setting",
                "epoch",
                "preprocessing",
                "decision_rule",
                "definition",
            ]
        )
        for detection_id, detection in enumerate(ivd_detections, 1):
            writer.writerow(
                [
                    "IVD",
                    detection_id,
                    f"{detection.x:.8g}",
                    f"{detection.y:.8g}",
                    f"{detection.peak:.8g}",
                    detection.pixels,
                    detection.rotation_sign,
                    f"{ivd_threshold:.8g}",
                    0.0,
                    f"{ivd_smooth_sigma:.8g}",
                    ivd_min_component_pixels,
                    "",
                    "instantaneous",
                    f"IVD >= {ivd_threshold:.8g}",
                    "abs(vorticity - spatial_mean_vorticity)",
                ]
            )
        for result in eulerian_results:
            for detection_id, detection in enumerate(result.detections, 1):
                writer.writerow(
                    [
                        result.label,
                        detection_id,
                        f"{detection.x:.8g}",
                        f"{detection.y:.8g}",
                        f"{detection.peak:.8g}",
                        detection.pixels,
                        detection.rotation_sign,
                        f"{result.threshold:.8g}",
                        0.0,
                        f"{result.smooth_sigma:.8g}",
                        result.min_component_pixels,
                        "",
                        "instantaneous-velocity-gradient",
                        result.decision_rule,
                        result.definition.replace("$", ""),
                    ]
                )
        for result in (
            mvunet_result,
            vortexunet_result,
            v0_result,
            v1_result,
        ):
            for detection_id, detection in enumerate(result.detections, 1):
                writer.writerow(
                    [
                        result.label,
                        detection_id,
                        f"{detection.x:.8g}",
                        f"{detection.y:.8g}",
                        f"{detection.peak:.8g}",
                        detection.pixels,
                        getattr(detection, "rotation_sign", ""),
                        f"{result.threshold:.8g}",
                        f"{result.duration:.8g}",
                        f"{result.smooth_sigma:.8g}",
                        result.min_component_pixels,
                        result.epoch,
                        result.preprocessing,
                        f"probability >= {result.threshold:.8g}",
                        (
                            "dense-grid U-Net probability"
                            if result.label in {"MVUNet", "VortexUNet"}
                            else "pathline-transformer probability"
                        ),
                    ]
                )


def plot_comparison(
    output_path: Path,
    input_path: Path,
    flow: v0_common.FlowWindow,
    snapshot_u: np.ndarray,
    snapshot_v: np.ndarray,
    start_time: float,
    ivd: np.ndarray,
    ivd_threshold: float,
    ivd_detections: Sequence[v0_common.RegionDetection],
    eulerian_results: Sequence[EulerianCriterionResult],
    v0_result: ModelDetectionResult,
    v1_result: ModelDetectionResult,
    mvunet_result: ModelDetectionResult,
    vortexunet_result: ModelDetectionResult,
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
    show_positive_seeds: bool,
    criterion_display_percentile: float,
) -> tuple[Path, Path]:
    """Render aligned classical/MVUNet/VortexUNet/V0/V1 panels."""

    if len(eulerian_results) != 3:
        raise ValueError("Exactly Q, Delta, and lambda-2 results are required.")
    figure = plt.figure(figsize=(17.5, 21.8), constrained_layout=True)
    grid = figure.add_gridspec(
        8,
        2,
        width_ratios=(1.0, 0.018),
        height_ratios=(1.0,) * 8,
    )
    axes = []
    for row in range(8):
        axes.append(
            figure.add_subplot(
                grid[row, 0],
                sharex=axes[0] if axes else None,
                sharey=axes[0] if axes else None,
            )
        )
    ivd_color_axis = figure.add_subplot(grid[0, 1])
    criterion_color_axis = figure.add_subplot(grid[1:4, 1])
    probability_color_axis = figure.add_subplot(grid[4:, 1])
    figure.suptitle(
        f"Vortex detection comparison — {input_path.name}, t={start_time:g}",
        fontsize=15,
        weight="bold",
    )
    stream_skip = max(1, len(flow.x) // 320)
    stream_x = np.linspace(flow.x[0], flow.x[-1], len(flow.x))
    stream_y = np.linspace(flow.y[0], flow.y[-1], len(flow.y))
    field_cmap = plot_style.FIELD_CMAP
    background_color = plot_style.BACKGROUND_COLOR
    threshold_color = plot_style.THRESHOLD_COLOR
    streamline_sets: list[object] = []
    for axis in axes:
        axis.set_facecolor(background_color)

    def draw_streamlines(
        axis,
        linewidth: float = 0.42,
        alpha: float = 1.0,
    ) -> None:
        streamline_sets.append(axis.streamplot(
            stream_x[::stream_skip],
            stream_y,
            snapshot_u[:, ::stream_skip],
            snapshot_v[:, ::stream_skip],
            color=plot_style.streamline_color(alpha),
            density=(4.0, 1.15),
            linewidth=linewidth,
            arrowsize=0.5,
            minlength=0.05,
            zorder=4,
        ))

    def draw_obstacle(axis) -> None:
        if obstacle_radius > 0.0:
            axis.add_patch(
                Circle(
                    obstacle_center,
                    obstacle_radius,
                    facecolor=plot_style.OBSTACLE_FACE_COLOR,
                    edgecolor=plot_style.OBSTACLE_EDGE_COLOR,
                    linewidth=1.0,
                    zorder=8,
                )
            )

    def finish_axis(axis) -> None:
        draw_obstacle(axis)
        axis.set_xlim(flow.x[0], flow.x[-1])
        axis.set_ylim(flow.y[0], flow.y[-1])
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])

    ivd_axis = axes[0]
    finite_ivd_values = np.asarray(ivd)[np.isfinite(ivd)]
    if finite_ivd_values.size == 0:
        raise ValueError("IVD has no finite display values.")
    ivd_norm = Normalize(
        vmin=0.0,
        vmax=max(
            float(np.max(finite_ivd_values)),
            np.finfo(np.float64).eps,
        ),
    )
    ivd_image = ivd_axis.pcolormesh(
        flow.x,
        flow.y,
        ivd,
        shading="auto",
        cmap=field_cmap,
        norm=ivd_norm,
        rasterized=True,
    )
    draw_streamlines(ivd_axis)
    finite_ivd = np.ma.masked_invalid(ivd)
    if (
        finite_ivd_values.size
        and float(np.min(finite_ivd_values)) < ivd_threshold
        < float(np.max(finite_ivd_values))
    ):
        ivd_axis.contour(
            flow.x,
            flow.y,
            finite_ivd,
            levels=[ivd_threshold],
            colors=[threshold_color],
            linewidths=1.2,
            zorder=5,
        )
    ivd_axis.set_title(
        f"IVD | threshold={ivd_threshold:.3g} | "
        f"regions={len(ivd_detections)}",
        fontsize=11,
    )
    ivd_axis.legend(
        handles=[
            Line2D(
                [], [], color=threshold_color, linewidth=1.4,
                label=f"threshold={ivd_threshold:.3g} contour",
            ),
        ],
        loc="upper right",
        fontsize=7,
        framealpha=0.85,
    )
    finish_axis(ivd_axis)
    ivd_colorbar = figure.colorbar(ivd_image, cax=ivd_color_axis)
    ivd_colorbar.set_label("IVD")

    criterion_score_fields = [
        -result.values if result.lower_is_vortex else result.values
        for result in eulerian_results
    ]
    finite_score_blocks = [
        values[np.isfinite(values)]
        for values in criterion_score_fields
        if np.any(np.isfinite(values))
    ]
    if not finite_score_blocks:
        raise ValueError("Q/Delta/lambda-2 have no finite display values.")
    finite_criterion_scores = np.concatenate(finite_score_blocks)
    criterion_display_limit = float(
        np.percentile(
            np.abs(finite_criterion_scores),
            criterion_display_percentile,
        )
    )
    criterion_display_limit = max(
        criterion_display_limit,
        np.finfo(np.float64).eps,
    )
    criterion_norm = Normalize(vmin=0.0, vmax=criterion_display_limit)

    def draw_eulerian_criterion(
        axis,
        result: EulerianCriterionResult,
    ) -> None:
        criterion_score = (
            -result.values if result.lower_is_vortex else result.values
        )
        score_threshold = (
            -result.threshold if result.lower_is_vortex else result.threshold
        )
        finite_scores = criterion_score[np.isfinite(criterion_score)]
        if finite_scores.size == 0:
            raise ValueError(f"{result.label} has no finite display values.")
        criterion_display = np.maximum(criterion_score, 0.0)
        axis.pcolormesh(
            flow.x,
            flow.y,
            criterion_display,
            shading="auto",
            cmap=field_cmap,
            norm=criterion_norm,
            rasterized=True,
        )
        draw_streamlines(axis, linewidth=0.30, alpha=0.48)
        finite_min = float(np.min(finite_scores))
        finite_max = float(np.max(finite_scores))
        if finite_min < score_threshold < finite_max:
            axis.contour(
                flow.x,
                flow.y,
                np.ma.masked_invalid(criterion_score),
                levels=[score_threshold],
                colors=[threshold_color],
                linewidths=1.2,
                zorder=5,
            )
        operator = r"$<$" if result.lower_is_vortex else r"$>$"
        display_symbol = {
            "Q": r"$Q$",
            "Delta_2D": r"$\Delta_{2D}$",
            "lambda_2": r"$\lambda_2$",
        }[result.symbol]
        axis.set_title(
            f"{result.label} | {result.definition} | "
            f"{display_symbol} {operator} {result.threshold:.3g} | "
            f"regions={len(result.detections)}",
            fontsize=10.5,
        )
        finish_axis(axis)

    for row, result in enumerate(eulerian_results, start=1):
        draw_eulerian_criterion(
            axes[row],
            result,
        )

    criterion_mappable = ScalarMappable(
        norm=criterion_norm,
        cmap=field_cmap,
    )
    criterion_mappable.set_array([])
    criterion_colorbar = figure.colorbar(
        criterion_mappable,
        cax=criterion_color_axis,
        extend="max",
    )
    criterion_colorbar.set_label(
        r"positive vortex strength $[t^{-2}]$: $Q$, $\Delta_{2D}$, $-\lambda_2$"
    )

    probability_norm = Normalize(vmin=0.0, vmax=1.0)
    probability_cmap = field_cmap

    def draw_model(axis, result: ModelDetectionResult) -> None:
        probability = np.ma.masked_invalid(result.probability)
        axis.pcolormesh(
            flow.x,
            flow.y,
            probability,
            shading="auto",
            cmap=probability_cmap,
            norm=probability_norm,
            rasterized=True,
            zorder=2,
        )
        draw_streamlines(axis)
        finite_probability = probability.compressed()
        if (
            finite_probability.size
            and float(np.min(finite_probability)) < result.threshold
            < float(np.max(finite_probability))
        ):
            axis.contour(
                flow.x,
                flow.y,
                probability,
                levels=[result.threshold],
                colors=[threshold_color],
                linewidths=1.25,
                zorder=5,
            )
        if show_positive_seeds:
            positive = result.scores >= result.threshold
            axis.scatter(
                result.points[positive, 0],
                result.points[positive, 1],
                s=4,
                facecolors="none",
                edgecolors="white",
                linewidths=0.3,
                alpha=0.55,
                zorder=5,
            )
        axis.set_title(
            f"{result.label} (epoch {result.epoch}) | "
            f"{result.preprocessing}, horizon={result.duration:g}, "
            f"threshold={result.threshold:g} | "
            f"regions={len(result.detections)}",
            fontsize=11,
        )
        handles = [
            Line2D(
                [], [], color=threshold_color, linewidth=1.4,
                label=f"p={result.threshold:g} contour",
            ),
        ]
        if show_positive_seeds:
            handles.append(
                Line2D(
                    [], [], marker="o", linestyle="none", color="white",
                    markerfacecolor="none", markersize=3,
                    label="positive seed",
                )
            )
        axis.legend(
            handles=handles,
            loc="upper right",
            fontsize=7,
            framealpha=0.85,
        )
        finish_axis(axis)

    draw_model(axes[4], mvunet_result)
    draw_model(axes[5], vortexunet_result)
    draw_model(axes[6], v0_result)
    draw_model(axes[7], v1_result)
    probability_mappable = ScalarMappable(
        norm=probability_norm, cmap=probability_cmap
    )
    probability_mappable.set_array([])
    probability_colorbar = figure.colorbar(
        probability_mappable,
        cax=probability_color_axis,
    )
    probability_colorbar.set_label("vortex-membership probability")

    if output_path.suffix.lower() == ".pdf":
        raster_output_path = output_path.with_suffix(".png")
        pdf_output_path = output_path
    elif output_path.suffix:
        raster_output_path = output_path
        pdf_output_path = output_path.with_suffix(".pdf")
    else:
        raster_output_path = output_path.with_suffix(".png")
        pdf_output_path = output_path.with_suffix(".pdf")

    raster_output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(raster_output_path, dpi=220)
    plot_style.thin_streamplots_for_pdf(streamline_sets)
    figure.savefig(pdf_output_path, dpi=220, format="pdf")
    plt.close(figure)
    return raster_output_path, pdf_output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare instantaneous IVD/Q/Delta/lambda-2, "
            "PathlineTransformerV0, PathlineTransformerV1, MVUNet, and "
            "VortexUNet on one "
            "NetCDF snapshot."
        )
    )
    parser.add_argument(
        "--input", type=Path, default=Path("dataset/cylinder2d.nc")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/vortex_detection_comparison.png"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/vortex_detection_comparison.csv"),
    )
    parser.add_argument(
        "--v0-scores-output",
        type=Path,
        default=Path("outputs/comparison_v0_seed_scores.csv"),
    )
    parser.add_argument(
        "--v1-scores-output",
        type=Path,
        default=Path("outputs/comparison_v1_seed_scores.csv"),
    )
    parser.add_argument("--time-index", type=int, default=1000)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu"],
        default="auto",
        help=(
            "Both traced graphs contain CPU constants; auto therefore resolves "
            "to CPU for reproducible comparison."
        ),
    )

    v0_group = parser.add_argument_group("PathlineTransformerV0")
    v0_group.add_argument(
        "--v0-checkpoint",
        type=Path,
        default=Path("trainedVortexTransformer/best_checkpoint.pth.tar"),
    )
    v0_group.add_argument(
        "--v0-torchscript",
        type=Path,
        default=Path(
            "trainedVortexTransformer/BackUp_best_checkpoint.pthtraced.pt"
        ),
    )
    v0_group.add_argument("--v0-duration", type=float, default=0.72)
    v0_group.add_argument("--v0-patch-points", type=int, default=32)
    v0_group.add_argument("--v0-patch-stride-points", type=int, default=32)
    v0_group.add_argument(
        "--v0-model-time-start",
        type=float,
        default=v0_common.LEGACY_DEFAULT_MODEL_TIME_START,
    )
    v0_group.add_argument(
        "--v0-ivd-reference-start-index", type=int, default=800
    )
    v0_group.add_argument(
        "--v0-ivd-reference-end-index", type=int, default=1000
    )
    v0_group.add_argument(
        "--v0-ivd-reference-mean",
        type=float,
        default=None,
        help="Optional precomputed signed-curl mean; overrides frame indices.",
    )
    v0_group.add_argument("--v0-threshold", type=float, default=0.5)
    v0_group.add_argument("--v0-smooth-sigma", type=float, default=1.0)
    v0_group.add_argument(
        "--v0-min-component-pixels", type=int, default=50
    )
    v0_group.add_argument("--v0-batch-size", type=int, default=4)

    v1_group = parser.add_argument_group("PathlineTransformerV1")
    v1_group.add_argument(
        "--v1-checkpoint",
        type=Path,
        default=Path("trainedVortexTransformerV1/best_checkpoint.pth.tar"),
    )
    v1_group.add_argument(
        "--v1-torchscript",
        type=Path,
        default=Path(
            "trainedVortexTransformerV1/BackUp_best_checkpoint.pthtraced.pt"
        ),
    )
    v1_group.add_argument(
        "--v1-config",
        type=Path,
        default=Path("config/segmentation/pathline_transformer_v1.yaml"),
    )
    v1_group.add_argument(
        "--v1-backend",
        choices=["torchscript", "eager"],
        default="torchscript",
    )
    v1_group.add_argument("--v1-duration", type=float, default=1.0)
    v1_group.add_argument("--v1-tile-size", type=float, default=None)
    v1_group.add_argument("--v1-tile-stride", type=float, default=None)
    v1_group.add_argument("--v1-seed", type=int, default=7068)
    v1_group.add_argument(
        "--v1-max-reseed-attempts", type=int, default=32
    )
    v1_group.add_argument(
        "--v1-feature-high-fraction", type=float, default=0.80
    )
    v1_group.add_argument(
        "--v1-feature-high-vorticity-percentile", type=float, default=70.0
    )
    v1_group.add_argument(
        "--v1-feature-mesh-resolution", type=int, default=64
    )
    v1_group.add_argument(
        "--v1-feature-mesh-size-ratio", type=float, default=4.0
    )
    v1_group.add_argument(
        "--v1-feature-monitor-power", type=float, default=1.5
    )
    v1_group.add_argument(
        "--v1-feature-mesh-iterations", type=int, default=5
    )
    v1_group.add_argument(
        "--v1-feature-clip-percentile", type=float, default=99.0
    )
    v1_group.add_argument(
        "--v1-feature-boundary-margin", type=float, default=1e-6
    )
    v1_group.add_argument(
        "--v1-obstacle-seeding-margin",
        type=float,
        default=0.0,
        help="Set 1.5-2 to reduce adaptive seeds concentrated at the wall.",
    )
    v1_group.add_argument("--v1-threshold", type=float, default=0.5)
    v1_group.add_argument("--v1-smooth-sigma", type=float, default=3.0)
    v1_group.add_argument(
        "--v1-min-component-pixels", type=int, default=8
    )
    v1_group.add_argument("--v1-batch-size", type=int, default=1)

    ivd_group = parser.add_argument_group("instantaneous IVD")
    ivd_group.add_argument("--ivd-threshold", type=float, default=None)
    ivd_group.add_argument("--ivd-percentile", type=float, default=91.0)
    ivd_group.add_argument("--ivd-smooth-sigma", type=float, default=1.0)
    ivd_group.add_argument("--ivd-min-component-pixels", type=int, default=8)
    ivd_group.add_argument("--ivd-obstacle-margin", type=float, default=1.5)
    ivd_group.add_argument("--ivd-border-cells", type=int, default=2)

    classic_group = parser.add_argument_group(
        "instantaneous velocity-gradient criteria"
    )
    classic_group.add_argument(
        "--q-threshold",
        type=float,
        default=0.0,
        help="Detect Q > this value (standard boundary: 0).",
    )
    classic_group.add_argument(
        "--delta-threshold",
        type=float,
        default=0.0,
        help=(
            "Detect planar Delta > this value, where Delta is "
            "det(J)-trace(J)^2/4 (standard boundary: 0)."
        ),
    )
    classic_group.add_argument(
        "--lambda2-threshold",
        type=float,
        default=0.0,
        help="Detect lambda_2 < this value (standard boundary: 0).",
    )
    classic_group.add_argument(
        "--classic-smooth-sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma applied to each velocity-gradient scalar field.",
    )
    classic_group.add_argument(
        "--classic-min-component-pixels", type=int, default=8
    )
    classic_group.add_argument(
        "--classic-obstacle-margin", type=float, default=1.5
    )
    classic_group.add_argument("--classic-border-cells", type=int, default=2)
    classic_group.add_argument(
        "--criterion-display-percentile",
        type=float,
        default=99.0,
        help=(
            "Symmetric robust color limit percentile for Q/Delta/lambda-2; "
            "this changes only the visualization, not detections."
        ),
    )

    geometry = parser.add_argument_group("geometry and display")
    geometry.add_argument("--obstacle-x", type=float, default=0.0)
    geometry.add_argument("--obstacle-y", type=float, default=0.0)
    geometry.add_argument("--obstacle-radius", type=float, default=None)
    geometry.add_argument(
        "--show-positive-seeds",
        action="store_true",
        help="Overlay threshold-positive samples in all learned-model panels.",
    )
    mvunet.add_arguments(parser)
    vortexunet.add_arguments(parser)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    mvunet.validate_arguments(args)
    vortexunet.validate_arguments(args)
    args.mvunet_checkpoint = mvunet.resolve_artifact(
        args.mvunet_checkpoint, mvunet.DEFAULT_CHECKPOINT
    )
    args.mvunet_torchscript = mvunet.resolve_artifact(
        args.mvunet_torchscript, mvunet.DEFAULT_TORCHSCRIPT
    )
    args.vortexunet_checkpoint = vortexunet.resolve_artifact(
        args.vortexunet_checkpoint, vortexunet.DEFAULT_CHECKPOINT
    )
    args.vortexunet_torchscript = vortexunet.resolve_artifact(
        args.vortexunet_torchscript, vortexunet.DEFAULT_TORCHSCRIPT
    )
    required_files = [
        args.input,
        args.v0_checkpoint,
        args.v0_torchscript,
        args.v1_checkpoint,
        args.v1_config,
        args.mvunet_checkpoint,
        args.mvunet_torchscript,
        args.vortexunet_checkpoint,
        args.vortexunet_torchscript,
    ]
    if args.v1_backend == "torchscript":
        required_files.append(args.v1_torchscript)
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required file(s) not found: " + ", ".join(missing))

    if args.v0_duration <= 0.0 or args.v1_duration <= 0.0:
        raise ValueError("Both model durations must be positive.")
    if args.v0_patch_points < 2 or args.v0_patch_stride_points < 1:
        raise ValueError("V0 patch points must be >=2 and stride must be positive.")
    if not np.isfinite(args.v0_model_time_start):
        raise ValueError("--v0-model-time-start must be finite.")
    if args.v0_ivd_reference_mean is None:
        if not (
            0
            <= args.v0_ivd_reference_start_index
            <= args.v0_ivd_reference_end_index
        ):
            raise ValueError("Invalid V0 legacy IVD reference frame range.")
    elif not np.isfinite(args.v0_ivd_reference_mean):
        raise ValueError("--v0-ivd-reference-mean must be finite.")
    if args.v1_tile_size is not None and args.v1_tile_size <= 0.0:
        raise ValueError("--v1-tile-size must be positive.")
    if args.v1_tile_stride is not None and args.v1_tile_stride <= 0.0:
        raise ValueError("--v1-tile-stride must be positive.")
    if args.v1_max_reseed_attempts < 0:
        raise ValueError("--v1-max-reseed-attempts cannot be negative.")
    if not 0.0 <= args.v1_feature_high_fraction <= 1.0:
        raise ValueError("--v1-feature-high-fraction must be in [0,1].")
    if not 0.0 < args.v1_feature_high_vorticity_percentile < 100.0:
        raise ValueError(
            "--v1-feature-high-vorticity-percentile must be in (0,100)."
        )
    if args.v1_feature_mesh_resolution < 8:
        raise ValueError("--v1-feature-mesh-resolution must be at least 8.")
    if args.v1_feature_mesh_size_ratio < 1.0:
        raise ValueError("--v1-feature-mesh-size-ratio must be at least 1.")
    if args.v1_feature_monitor_power <= 0.0:
        raise ValueError("--v1-feature-monitor-power must be positive.")
    if args.v1_feature_mesh_iterations < 0:
        raise ValueError("--v1-feature-mesh-iterations cannot be negative.")
    if not 0.0 < args.v1_feature_clip_percentile <= 100.0:
        raise ValueError("--v1-feature-clip-percentile must be in (0,100].")
    if not 0.0 <= args.v1_feature_boundary_margin < 1.0:
        raise ValueError("--v1-feature-boundary-margin must be in [0,1).")
    if args.v1_obstacle_seeding_margin < 0.0:
        raise ValueError("--v1-obstacle-seeding-margin cannot be negative.")
    if args.v0_batch_size < 1 or args.v1_batch_size < 1:
        raise ValueError("Both model batch sizes must be positive.")
    for name, value in (
        ("--v0-threshold", args.v0_threshold),
        ("--v1-threshold", args.v1_threshold),
    ):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    if args.v0_smooth_sigma < 0.0 or args.v1_smooth_sigma < 0.0:
        raise ValueError("Model smoothing sigmas cannot be negative.")
    if (
        args.v0_min_component_pixels < 1
        or args.v1_min_component_pixels < 1
    ):
        raise ValueError("Model component sizes must be positive.")
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
    for name, value in (
        ("--q-threshold", args.q_threshold),
        ("--delta-threshold", args.delta_threshold),
        ("--lambda2-threshold", args.lambda2_threshold),
    ):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if args.classic_smooth_sigma < 0.0:
        raise ValueError("--classic-smooth-sigma cannot be negative.")
    if args.classic_min_component_pixels < 1:
        raise ValueError("--classic-min-component-pixels must be positive.")
    if args.classic_obstacle_margin < 1.0:
        raise ValueError("--classic-obstacle-margin must be at least 1.")
    if args.classic_border_cells < 0:
        raise ValueError("--classic-border-cells cannot be negative.")
    if not 0.0 < args.criterion_display_percentile <= 100.0:
        raise ValueError(
            "--criterion-display-percentile must be in (0,100]."
        )
    if args.obstacle_radius is not None and args.obstacle_radius < 0.0:
        raise ValueError("--obstacle-radius cannot be negative.")


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    # V0's deterministic trace and V1's paired trace contain CPU tensor
    # constants.  Keeping both on CPU is the only like-for-like backend.
    device = "cpu"

    reference_indices = (
        None
        if args.v0_ivd_reference_mean is not None
        else (
            args.v0_ivd_reference_start_index,
            args.v0_ivd_reference_end_index,
        )
    )
    window_duration = max(args.v0_duration, args.v1_duration)
    flow, start_time, snapshot_u, snapshot_v, constants = (
        v0_common.load_flow_window(
            args.input,
            args.time_index,
            window_duration,
            legacy_reference_indices=reference_indices,
            legacy_reference_mean=args.v0_ivd_reference_mean,
        )
    )
    obstacle_center = (args.obstacle_x, args.obstacle_y)
    obstacle_radius = (
        constants.get("radius", 0.0)
        if args.obstacle_radius is None
        else args.obstacle_radius
    )

    print(
        f"Reading {args.input}: grid={len(flow.x)}x{len(flow.y)}, "
        f"snapshot t={start_time:g}; shared window ends at "
        f"t={start_time + window_duration:g}."
    )
    print("Running PathlineTransformerV0 (legacy-grid)...")
    v0_result = run_v0(
        flow,
        start_time,
        args,
        obstacle_center,
        obstacle_radius,
        device,
    )
    print("Running PathlineTransformerV1 (feature-adaptive-mesh)...")
    v1_result, reseed_passes, invalid_t0_count, v1_backend = run_v1(
        flow,
        start_time,
        args,
        obstacle_center,
        obstacle_radius,
        device,
    )
    ivd, ivd_threshold, mean_vorticity, ivd_detections = (
        v0_common.detect_ivd_regions(
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
    eulerian_results = build_eulerian_criteria(
        flow,
        snapshot_u,
        snapshot_v,
        args,
        obstacle_center,
        obstacle_radius,
    )
    grid_x, grid_y = np.meshgrid(flow.x, flow.y)
    mvunet_valid = np.isfinite(snapshot_u) & np.isfinite(snapshot_v)
    if obstacle_radius > 0.0:
        mvunet_valid &= np.hypot(
            grid_x - obstacle_center[0], grid_y - obstacle_center[1]
        ) > args.classic_obstacle_margin * obstacle_radius
    print(f"Running MVUNet with {mvunet.TRAINING_SEEDING} training data...")
    mv = mvunet.run(
        snapshot_u,
        snapshot_v,
        flow.x,
        flow.y,
        args,
        valid_mask=mvunet_valid,
        border_cells=args.classic_border_cells,
        device=device,
    )
    mv_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    mvunet_result = ModelDetectionResult(
        label="MVUNet",
        preprocessing=mvunet.PREPROCESSING + "; " + mvunet.TRAINING_SEEDING,
        epoch=mv.epoch,
        duration=0.0,
        patch_count=1,
        threshold=args.mvunet_threshold,
        smooth_sigma=args.mvunet_smooth_sigma,
        min_component_pixels=args.mvunet_min_component_pixels,
        points=mv_points,
        scores=mv.probability.ravel(),
        source_patches=np.zeros(len(mv_points), dtype=np.int64),
        probability=mv.probability,
        support=np.isfinite(mv.probability).astype(np.float64),
        detections=mv.detections,
    )
    print(
        f"Running VortexUNet with {vortexunet.TRAINING_SEEDING} "
        "training data..."
    )
    vu = vortexunet.run(
        snapshot_u,
        snapshot_v,
        flow.x,
        flow.y,
        args,
        valid_mask=mvunet_valid,
        border_cells=args.classic_border_cells,
        device=device,
    )
    vortexunet_result = ModelDetectionResult(
        label="VortexUNet",
        preprocessing=(
            vortexunet.PREPROCESSING + "; " + vortexunet.TRAINING_SEEDING
        ),
        epoch=vu.epoch,
        duration=0.0,
        patch_count=1,
        threshold=args.vortexunet_threshold,
        smooth_sigma=args.vortexunet_smooth_sigma,
        min_component_pixels=args.vortexunet_min_component_pixels,
        points=mv_points,
        scores=vu.probability.ravel(),
        source_patches=np.zeros(len(mv_points), dtype=np.int64),
        probability=vu.probability,
        support=np.isfinite(vu.probability).astype(np.float64),
        detections=vu.detections,
    )

    v0_common.save_seed_scores(
        args.v0_scores_output,
        v0_result.points,
        v0_result.scores,
        v0_result.source_patches,
        v0_result.threshold,
    )
    v0_common.save_seed_scores(
        args.v1_scores_output,
        v1_result.points,
        v1_result.scores,
        v1_result.source_patches,
        v1_result.threshold,
    )
    save_comparison_summary(
        args.summary_output,
        ivd_threshold,
        args.ivd_smooth_sigma,
        args.ivd_min_component_pixels,
        ivd_detections,
        eulerian_results,
        v0_result,
        v1_result,
        mvunet_result,
        vortexunet_result,
    )
    figure_output, pdf_output = plot_comparison(
        args.output,
        args.input,
        flow,
        snapshot_u,
        snapshot_v,
        start_time,
        ivd,
        ivd_threshold,
        ivd_detections,
        eulerian_results,
        v0_result,
        v1_result,
        mvunet_result,
        vortexunet_result,
        obstacle_center,
        obstacle_radius,
        args.show_positive_seeds,
        args.criterion_display_percentile,
    )

    v0_positive = int(
        np.count_nonzero(v0_result.scores >= v0_result.threshold)
    )
    v1_positive = int(
        np.count_nonzero(v1_result.scores >= v1_result.threshold)
    )
    print(
        f"IVD: mean vorticity={mean_vorticity:.6g}, "
        f"threshold={ivd_threshold:.6g}, regions={len(ivd_detections)}."
    )
    for result in eulerian_results:
        operator = "<" if result.lower_is_vortex else ">"
        print(
            f"{result.label}: {result.symbol} {operator} "
            f"{result.threshold:.6g}, regions={len(result.detections)}."
        )
    print(
        f"V0 epoch {v0_result.epoch}: patches={v0_result.patch_count}, "
        f"positive seeds={v0_positive}/{len(v0_result.scores)}, "
        f"regions={len(v0_result.detections)}."
    )
    print(
        f"V1 epoch {v1_result.epoch} via {v1_backend}: "
        f"patches={v1_result.patch_count}, "
        f"positive seeds={v1_positive}/{len(v1_result.scores)}, "
        f"regions={len(v1_result.detections)}, "
        f"reseed passes={reseed_passes}, invalid t0 seeds={invalid_t0_count}."
    )
    print(
        f"MVUNet epoch {mvunet_result.epoch}: "
        f"positive cells={np.count_nonzero(mvunet_result.probability >= mvunet_result.threshold)}/"
        f"{np.count_nonzero(np.isfinite(mvunet_result.probability))}, "
        f"regions={len(mvunet_result.detections)}, "
        f"training seeding={mvunet.TRAINING_SEEDING}."
    )
    print(
        f"VortexUNet epoch {vortexunet_result.epoch}: "
        f"positive cells={np.count_nonzero(vortexunet_result.probability >= vortexunet_result.threshold)}/"
        f"{np.count_nonzero(np.isfinite(vortexunet_result.probability))}, "
        f"regions={len(vortexunet_result.detections)}, "
        f"training seeding={vortexunet.TRAINING_SEEDING}."
    )
    print(f"Comparison figure: {figure_output}")
    print(f"Comparison PDF: {pdf_output}")
    print(f"Detection summary: {args.summary_output}")
    print(f"V0 seed scores: {args.v0_scores_output}")
    print(f"V1 seed scores: {args.v1_scores_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
