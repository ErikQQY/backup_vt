#!/usr/bin/env python3
"""Run the eight-method vortex comparison on ``fourcenters2d.nc``.

This is a dataset-specific entry point built on :mod:`comparison`.  It keeps
the IVD, Q, Delta-2D, lambda-2, PathlineTransformerV0,
PathlineTransformerV1, MVUNet, and VortexUNet pipelines identical to
``comparison.py``, while
replacing cylinder-specific defaults with values valid for the rotating
four-centres (RFC) dataset.

The RFC file has 512 periodic frames on ``[-2, 2] x [-2, 2]`` and contains no
solid obstacle.  Frame 256 is a near-symmetric mid-period snapshot.  V0 uses
the complete period (frames 0--511) for its signed-vorticity reference mean.
V1 uses an objective ``|omega - domain_mean(omega)|`` adaptive monitor and
overlapping 3-by-3 physical tiles so that the two circulation signs receive
comparable seed support.

All options from ``comparison.py`` remain available and command-line values
override the defaults below.  For example::

    .venv/bin/python comparison_rfc.py --time-index 300

The plot is written as both PNG and PDF by ``comparison.py``.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import comparison as comparison_base
import comparison_plot_style as plot_style
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

_BASE_BUILD_PARSER = comparison_base.build_parser


def build_parser() -> argparse.ArgumentParser:
    """Return ``comparison.py``'s parser with RFC-specific defaults."""

    parser = _BASE_BUILD_PARSER()
    parser.description = (
        "Compare IVD, Q, Delta-2D, lambda-2, Vortex Transformer, "
        "FGMA-VT, MVUNet, and VortexUNet on the rotating four-centres "
        "NetCDF dataset."
    )
    parser.epilog = (
        "RFC defaults: frame 256, V0 vorticity reference frames 0:511, "
        "V1 tile-size 3 with stride 1, an IVD seeding monitor, and no "
        "obstacle. All options can "
        "be overridden explicitly."
    )
    rfc_v1_group = parser.add_argument_group("RFC-specific FGMA-VT")
    rfc_v1_group.add_argument(
        "--v1-monitor",
        "--v1-seeding-monitor",
        dest="v1_monitor",
        choices=["ivd", "raw-vorticity"],
        default="ivd",
        help=(
            "Adaptive-mesh monitor. 'ivd' removes the instantaneous global "
            "mean rotation and uses |omega-domain_mean(omega)|; "
            "'raw-vorticity' "
            "restores the training generator's original |omega| monitor."
        ),
    )
    parser.set_defaults(
        input=Path("dataset/fourcenters2d.nc"),
        output=Path("outputs/fourcenters2d_comparison.png"),
        summary_output=Path("outputs/fourcenters2d_comparison.csv"),
        v0_scores_output=Path("outputs/fourcenters2d_v0_seed_scores.csv"),
        v1_scores_output=Path("outputs/fourcenters2d_v1_seed_scores.csv"),
        time_index=256,
        v0_ivd_reference_start_index=0,
        v0_ivd_reference_end_index=511,
        v0_ivd_reference_mean=None,
        v1_tile_size=3.0,
        v1_tile_stride=1.0,
        v1_threshold=0.515,
        v1_smooth_sigma=3.25,
        obstacle_x=0.0,
        obstacle_y=0.0,
        obstacle_radius=0.0,
        v1_obstacle_seeding_margin=0.0,
    )
    return parser


class _ReferencedVorticityFlow:
    """Expose vorticity samples relative to one global rotation reference."""

    def __init__(self, flow: Any, reference: float) -> None:
        self.vorticity = flow.vorticity
        self._flow = flow
        self._reference = float(reference)

    def sample(
        self,
        field: np.ndarray,
        points: np.ndarray,
        time: float,
    ) -> np.ndarray:
        return self._flow.sample(field, points, time) - self._reference


def _prepare_rfc_adaptive_inputs(
    flow: Any,
    patches: list[Any],
    args: argparse.Namespace,
    start_time: float,
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
) -> tuple[Any, int, int]:
    """Prepare V1 inputs with an optional objective adaptive-mesh monitor."""

    preparation_args = comparison_base._v1_preparation_args(args)
    if args.v1_monitor == "raw-vorticity":
        return comparison_base.v1_common.prepare_adaptive_inputs(
            flow,
            patches,
            preparation_args,
            start_time,
            obstacle_center,
            obstacle_radius,
        )

    rng = np.random.default_rng(preparation_args.seed)
    feature_blocks: list[np.ndarray] = []
    seed_blocks: list[np.ndarray] = []
    valid_blocks: list[np.ndarray] = []
    total_reseed_passes = 0
    invalid_t0_count = 0

    domain_mean_vorticity = np.mean(
        flow.vorticity,
        axis=(1, 2),
        dtype=np.float64,
    )
    reference = float(
        np.interp(start_time, flow.times, domain_mean_vorticity)
    )
    monitor_flow = _ReferencedVorticityFlow(flow, reference)

    for patch in patches:
        seeds = comparison_base.v1_common.feature_adaptive_mesh_seeds(
            flow=monitor_flow,
            patch=patch,
            start_time=start_time,
            rng=rng,
            high_fraction=preparation_args.feature_high_fraction,
            high_vorticity_percentile=(
                preparation_args.feature_high_vorticity_percentile
            ),
            mesh_resolution=preparation_args.feature_mesh_resolution,
            mesh_size_ratio=preparation_args.feature_mesh_size_ratio,
            monitor_power=preparation_args.feature_monitor_power,
            mesh_iterations=preparation_args.feature_mesh_iterations,
            clip_percentile=preparation_args.feature_clip_percentile,
            boundary_margin=preparation_args.feature_boundary_margin,
            obstacle_center=obstacle_center,
            obstacle_radius=obstacle_radius,
            obstacle_seeding_margin=(
                preparation_args.obstacle_seeding_margin
            ),
        )
        features, final_seeds, initial_valid, reseed_passes = (
            comparison_base.v1_common.integrate_adaptive_patch(
                flow,
                patch,
                seeds,
                start_time,
                preparation_args.duration,
                obstacle_center,
                obstacle_radius,
                rng,
                preparation_args.max_reseed_attempts,
            )
        )
        feature_blocks.append(features)
        seed_blocks.append(final_seeds)
        valid_blocks.append(initial_valid)
        total_reseed_passes += reseed_passes
        invalid_t0_count += int(np.count_nonzero(~initial_valid))

    prepared = comparison_base.v0_common.PreparedInputs(
        list(patches),
        feature_blocks,
        seed_blocks,
        valid_blocks,
    )
    return prepared, total_reseed_passes, invalid_t0_count


def run_rfc_v1(
    flow: Any,
    start_time: float,
    args: argparse.Namespace,
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
    device: str,
) -> tuple[Any, int, int, str]:
    """Run FGMA-VT using the RFC-specific adaptive monitor."""

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
    patches = comparison_base.v0_common.make_cross_patches(
        flow.x,
        flow.y,
        tile_size,
        tile_stride,
    )
    prepared, reseed_passes, invalid_t0_count = (
        _prepare_rfc_adaptive_inputs(
            flow,
            patches,
            args,
            start_time,
            obstacle_center,
            obstacle_radius,
        )
    )
    model, epoch, backend = comparison_base.v1_common.load_v1_model(
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
    ) = comparison_base._project_predictions(
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
    monitor_label = (
        "IVD-monitor" if args.v1_monitor == "ivd" else "raw-vorticity-monitor"
    )
    result = comparison_base.ModelDetectionResult(
        label="FGMA-VT",
        preprocessing=f"feature-adaptive-mesh/{monitor_label}",
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
    if detections:
        component_pixels = np.asarray(
            [detection.pixels for detection in detections],
            dtype=np.float64,
        )
        area_cv = float(component_pixels.std() / component_pixels.mean())
        area_ratio = float(component_pixels.max() / component_pixels.min())
        print(
            "FGMA-VT area balance: "
            f"pixels={component_pixels.astype(int).tolist()}, "
            f"CV={100.0 * area_cv:.2f}%, max/min={area_ratio:.3f}."
        )
    return result, reseed_passes, invalid_t0_count, backend


def plot_rfc_comparison(
    output_path: Path,
    input_path: Path,
    flow: Any,
    snapshot_u: np.ndarray,
    snapshot_v: np.ndarray,
    start_time: float,
    ivd: np.ndarray,
    ivd_threshold: float,
    ivd_detections: list[Any],
    eulerian_results: list[Any],
    v0_result: Any,
    v1_result: Any,
    mvunet_result: Any,
    vortexunet_result: Any,
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
    show_positive_seeds: bool,
    criterion_display_percentile: float,
    *,
    figure_size: tuple[float, float] = (19.5, 9.5),
) -> tuple[Path, Path]:
    """Render a compact 2-by-4 comparison with a configurable canvas."""

    if len(eulerian_results) != 3:
        raise ValueError("Exactly Q, Delta, and lambda-2 results are required.")

    figure = plt.figure(figsize=figure_size)
    grid = figure.add_gridspec(
        2,
        5,
        width_ratios=(1.0, 1.0, 1.0, 1.0, 0.045),
        height_ratios=(1.0, 1.0),
        left=0.055,
        right=0.94,
        bottom=0.075,
        top=0.89,
        wspace=0.18,
        hspace=0.28,
    )
    panel_specs = [
        grid[0, 0],
        grid[0, 1],
        grid[0, 2],
        grid[0, 3],
        grid[1, 0],
        grid[1, 1],
        grid[1, 2],
        grid[1, 3],
    ]
    axes = []
    for panel_spec in panel_specs:
        axes.append(
            figure.add_subplot(
                panel_spec,
                sharex=axes[0] if axes else None,
                sharey=axes[0] if axes else None,
            )
        )
    color_grid = grid[:, 4].subgridspec(3, 1, hspace=0.52)
    ivd_color_axis = figure.add_subplot(color_grid[0, 0])
    criterion_color_axis = figure.add_subplot(color_grid[1, 0])
    probability_color_axis = figure.add_subplot(color_grid[2, 0])

    figure.suptitle(
        f"Vortex detection comparison — {input_path.name}, t={start_time:g}",
        fontsize=15,
        weight="bold",
    )
    field_cmap = plot_style.FIELD_CMAP
    threshold_color = plot_style.THRESHOLD_COLOR
    stream_skip = max(1, len(flow.x) // 256)
    stream_x = np.linspace(flow.x[0], flow.x[-1], len(flow.x))
    stream_y = np.linspace(flow.y[0], flow.y[-1], len(flow.y))
    streamline_sets: list[object] = []

    for axis in axes:
        axis.set_facecolor(plot_style.BACKGROUND_COLOR)

    def draw_streamlines(
        axis: Any,
        linewidth: float = 0.42,
        alpha: float = 0.9,
    ) -> None:
        streamline_sets.append(axis.streamplot(
            stream_x[::stream_skip],
            stream_y,
            snapshot_u[:, ::stream_skip],
            snapshot_v[:, ::stream_skip],
            color=plot_style.streamline_color(alpha),
            density=(1.35, 1.35),
            linewidth=linewidth,
            arrowsize=0.55,
            minlength=0.05,
            zorder=4,
        ))

    def finish_axis(axis: Any) -> None:
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
        axis.set_xlim(flow.x[0], flow.x[-1])
        axis.set_ylim(flow.y[0], flow.y[-1])
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])

    finite_ivd_values = np.asarray(ivd)[np.isfinite(ivd)]
    if finite_ivd_values.size == 0:
        raise ValueError("IVD has no finite display values.")
    ivd_norm = Normalize(
        vmin=0.0,
        vmax=max(float(np.max(finite_ivd_values)), np.finfo(float).eps),
    )
    ivd_image = axes[0].pcolormesh(
        flow.x,
        flow.y,
        ivd,
        shading="auto",
        cmap=field_cmap,
        norm=ivd_norm,
        rasterized=True,
    )
    draw_streamlines(axes[0])
    if (
        float(np.min(finite_ivd_values))
        < ivd_threshold
        < float(np.max(finite_ivd_values))
    ):
        axes[0].contour(
            flow.x,
            flow.y,
            np.ma.masked_invalid(ivd),
            levels=[ivd_threshold],
            colors=[threshold_color],
            linewidths=1.2,
            zorder=5,
        )
    axes[0].set_title(
        f"IVD\nthreshold={ivd_threshold:.3g} | "
        f"regions={len(ivd_detections)}",
        fontsize=10.5,
    )
    axes[0].legend(
        handles=[
            Line2D(
                [],
                [],
                color=threshold_color,
                linewidth=1.4,
                label=f"threshold={ivd_threshold:.3g} contour",
            )
        ],
        loc="upper right",
        fontsize=7,
        framealpha=0.85,
    )
    finish_axis(axes[0])
    ivd_colorbar = figure.colorbar(ivd_image, cax=ivd_color_axis)
    ivd_colorbar.set_label("IVD")

    criterion_score_fields = [
        -result.values if result.lower_is_vortex else result.values
        for result in eulerian_results
    ]
    finite_criterion_blocks = [
        values[np.isfinite(values)]
        for values in criterion_score_fields
        if np.any(np.isfinite(values))
    ]
    if not finite_criterion_blocks:
        raise ValueError("Q/Delta/lambda-2 have no finite display values.")
    finite_criterion_scores = np.concatenate(finite_criterion_blocks)
    criterion_limit = float(
        np.percentile(
            np.abs(finite_criterion_scores),
            criterion_display_percentile,
        )
    )
    criterion_limit = max(criterion_limit, np.finfo(float).eps)
    criterion_norm = Normalize(vmin=0.0, vmax=criterion_limit)

    def draw_criterion(axis: Any, result: Any) -> None:
        score = -result.values if result.lower_is_vortex else result.values
        score_threshold = (
            -result.threshold if result.lower_is_vortex else result.threshold
        )
        finite_scores = score[np.isfinite(score)]
        if finite_scores.size == 0:
            raise ValueError(f"{result.label} has no finite display values.")
        score_display = np.maximum(score, 0.0)
        axis.pcolormesh(
            flow.x,
            flow.y,
            score_display,
            shading="auto",
            cmap=field_cmap,
            norm=criterion_norm,
            rasterized=True,
        )
        draw_streamlines(axis, linewidth=0.30, alpha=0.48)
        if (
            float(np.min(finite_scores))
            < score_threshold
            < float(np.max(finite_scores))
        ):
            axis.contour(
                flow.x,
                flow.y,
                np.ma.masked_invalid(score),
                levels=[score_threshold],
                colors=[threshold_color],
                linewidths=1.2,
                zorder=5,
            )
        operator = "<" if result.lower_is_vortex else ">"
        symbol = {
            "Q": r"$Q$",
            "Delta_2D": r"$\Delta_{2D}$",
            "lambda_2": r"$\lambda_2$",
        }[result.symbol]
        axis.set_title(
            f"{result.label}\n{symbol} {operator} {result.threshold:.3g} | "
            f"regions={len(result.detections)}",
            fontsize=10.5,
        )
        finish_axis(axis)

    # Row-major order: IVD, Q, Delta, lambda-2, MVUNet, VortexUNet, V0, V1.
    draw_criterion(axes[1], eulerian_results[0])
    draw_criterion(axes[2], eulerian_results[1])
    draw_criterion(axes[3], eulerian_results[2])
    criterion_mappable = ScalarMappable(norm=criterion_norm, cmap=field_cmap)
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

    def draw_model(axis: Any, result: Any) -> None:
        probability = np.ma.masked_invalid(result.probability)
        axis.pcolormesh(
            flow.x,
            flow.y,
            probability,
            shading="auto",
            cmap=field_cmap,
            norm=probability_norm,
            rasterized=True,
            zorder=2,
        )
        draw_streamlines(axis)
        finite_probability = probability.compressed()
        if (
            finite_probability.size
            and float(np.min(finite_probability))
            < result.threshold
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
        preprocessing_label = {
            "feature-adaptive-mesh/IVD-monitor": "IVD-adaptive mesh",
            "feature-adaptive-mesh/raw-vorticity-monitor": (
                "raw-vorticity adaptive mesh"
            ),
        }.get(result.preprocessing, result.preprocessing)
        if result.label == "MVUNet":
            preprocessing_label = "dense [u,v,curl,IVD] grid"
        elif result.label == "VortexUNet":
            preprocessing_label = "dense [u,v] grid"
        balance_label = ""
        if result.label == "FGMA-VT" and len(result.detections) > 1:
            component_pixels = np.asarray(
                [detection.pixels for detection in result.detections],
                dtype=np.float64,
            )
            area_cv = float(component_pixels.std() / component_pixels.mean())
            balance_label = f" | area CV={100.0 * area_cv:.1f}%"
        axis.set_title(
            f"{result.label} (epoch {result.epoch}) | "
            f"regions={len(result.detections)}{balance_label}\n"
            f"{preprocessing_label} | h={result.duration:g} | "
            f"p >= {result.threshold:g}",
            fontsize=10.0,
        )
        handles = [
            Line2D(
                [],
                [],
                color=threshold_color,
                linewidth=1.4,
                label=f"p={result.threshold:g} contour",
            )
        ]
        if show_positive_seeds:
            handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    color="white",
                    markerfacecolor="none",
                    markersize=3,
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
        norm=probability_norm,
        cmap=field_cmap,
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


def main(argv: Iterable[str] | None = None) -> int:
    """Run the shared comparison pipeline with the RFC parser defaults."""

    # comparison.main() deliberately owns the complete orchestration.  Swap
    # only its parser for this call so computation, CSV output, plotting, and
    # checkpoint loading cannot drift from the canonical implementation.
    original_build_parser = comparison_base.build_parser
    original_plot_comparison = comparison_base.plot_comparison
    original_run_v1 = comparison_base.run_v1
    comparison_base.build_parser = build_parser
    comparison_base.plot_comparison = plot_rfc_comparison
    comparison_base.run_v1 = run_rfc_v1
    try:
        return comparison_base.main(argv)
    finally:
        comparison_base.build_parser = original_build_parser
        comparison_base.plot_comparison = original_plot_comparison
        comparison_base.run_v1 = original_run_v1


if __name__ == "__main__":
    raise SystemExit(main())
