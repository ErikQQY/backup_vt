"""Compare the four Vatistas pathline-seeding methods on one labelled LIC.

The sampling implementations, LIC renderer, and Vatistas field evaluator are
imported directly from ``VatistasFlowDatasetGenerator.py``.  The generator's
computational spatial domain is ``[-1, 1]^2`` and its time interval is
``[0, 1]``.  For the requested 1 x 1 x 1 schematic, this script maps x and y
to ``[0, 1]`` and plots the t=0 slice of that unit spatio-temporal domain.

Example
-------
python plot_vatistas_sampling_methods.py \
    --config config/VatistasDataset.yaml \
    --output outputs/vatistas_sampling_methods.png

The command creates both ``vatistas_sampling_methods.png`` and
``vatistas_sampling_methods.pdf``, plus the standalone mesh panel
``vatistas_sampling_methods_feature_adaptive_mesh_seeding.{pdf,svg}``, its
initial CVT state and five relaxation snapshots
``vatistas_sampling_methods_feature_adaptive_mesh_seeding_iter0.svg`` through
``vatistas_sampling_methods_feature_adaptive_mesh_seeding_iter5.svg``, and the
annotation-free ``vatistas_sampling_methods_lic_label.svg`` and
``vatistas_sampling_methods_blurred_flow.svg``, plus the point-only
``vatistas_sampling_methods_feature_adaptive_mesh_points.svg`` and
``vatistas_sampling_methods_original_flow_grid_points.svg``.
An eight-stage initialization diagram is also saved as
``vatistas_sampling_methods_feature_adaptive_mesh_initialization.{png,svg}``,
showing the actual structured candidate grid through the exact iter0 seeds,
with separate calculation sheets for high- and low-IVD farthest insertion.
Its monitor uses IVD relative to the spatial mean over the full generator
domain, matching the dataset generator's adaptive sampler.
The illustration selects the top 15% IVD candidates for a compact central
ellipse; use ``--high-ivd-percentile`` to adjust this plotting-only threshold.
Each initialization stage is also exported separately as PNG and SVG in
``outputs/adaptive_seeds/`` (beside the chosen output stem), without titles
or captions. Colorbars are placed outside, below their plots.
The exact points from stage 08 are also saved on a transparent canvas as
``adaptive_seeds/08_initial_adaptive_mesh_points.{png,svg}``.
Stage 06 is exported as a detailed calculation sheet, with its individual
views in ``06a_*`` through ``06h_*`` and the compact ``06_insertion_overview``.
The low-IVD continuation is saved as ``06_low_ivd_farthest_insertion`` and
``06i_low_ivd_*`` through ``06p_low_ivd_*``. Its orange points show the completed
high-IVD seeds, while all distances and insertion scores use only low-IVD seeds.
The iteration snapshots reuse the exact panel-08 canvas and initial seeds;
matching PNG previews are also saved beside the six iteration SVGs.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MatplotlibPath
from matplotlib.text import Text
from matplotlib.transforms import Affine2D, Bbox
import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from VatistasFlowDatasetGenerator import (
    FEATURE_ADAPTIVE_MESH_SEEDING,
    GRID_CROSS_SEEDING,
    RECTANGULAR_CLUSTER4_RANDOM,
    IVD_IMPORTANCE_SEEDING,
    EasyConfig,
    _autograd_ivd_at_points,
    _common_existing_seed_count,
    _feature_mesh_candidates,
    _lic_mesh_edges,
    _make_lic_noise,
    _pathline_seedings,
    _weighted_cvt_mesh_points,
    build_bounds,
    make_local_grid,
    make_steady_eval,
    read_lic_cfg,
    read_pathline_cfg,
    render_lic_image,
)


SAMPLING_METHODS = (
    GRID_CROSS_SEEDING,
    RECTANGULAR_CLUSTER4_RANDOM,
    IVD_IMPORTANCE_SEEDING,
    FEATURE_ADAPTIVE_MESH_SEEDING,
)

SAMPLING_TITLES = {
    GRID_CROSS_SEEDING: "Cross Seeding",
    RECTANGULAR_CLUSTER4_RANDOM: "Random Seeding",
    IVD_IMPORTANCE_SEEDING: "IVD Importance Seeding",
    FEATURE_ADAPTIVE_MESH_SEEDING: "Feature-Adaptive Seeding",
}

ADAPTIVE_CVT_ITERATIONS = tuple(range(0, 6))


@dataclass
class AdaptiveInitialization:
    """The candidate monitor and insertion order before any CVT relaxation."""

    config: dict
    structured: np.ndarray
    candidates: np.ndarray
    ivd: np.ndarray
    high_ids: np.ndarray
    low_ids: np.ndarray
    feature: np.ndarray
    density: np.ndarray
    high_seeds: np.ndarray
    low_seeds: np.ndarray
    iter0: np.ndarray


@dataclass
class InsertionStep:
    """Distances and eligible scores after exactly k fixed seeds are inserted."""

    selected: np.ndarray
    distance_sq: np.ndarray
    scores: np.ndarray
    next_id: int | None


def trace_farthest_insertion(candidates, density, insertion):
    """Replay and check every argmax against the sampler's insertion order."""
    point_ids = {tuple(point): i for i, point in enumerate(candidates)}
    selected = np.asarray([point_ids[tuple(point)] for point in insertion])
    available = np.ones(len(candidates), dtype=bool)
    distance_sq = np.full(len(candidates), np.inf)
    steps = {}
    for k, chosen in enumerate(selected, start=1):
        if k > 1:
            scores = distance_sq * density
            scores[~available] = -1.0
            if int(np.argmax(scores)) != chosen:
                raise RuntimeError(f"Weighted insertion replay differs at seed {k}.")
        available[chosen] = False
        distance_sq = np.minimum(
            distance_sq, np.sum((candidates - candidates[chosen]) ** 2, axis=1)
        )
        scores = distance_sq * density
        scores[~available] = -1.0
        steps[k] = InsertionStep(
            selected[:k].copy(), distance_sq.copy(), scores.copy(),
            int(np.argmax(scores)) if k < len(insertion) else None,
        )
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw the four VatistasDatasetGenerator seeding methods over the "
            "same yellow-labelled LIC image."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/VatistasDataset.yaml"),
        help="Vatistas dataset YAML used for sampling and LIC parameters.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/vatistas_sampling_methods.png"),
        help=(
            "Output path stem; matching PNG and PDF files are created "
            "(default: outputs/vatistas_sampling_methods.png)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed; defaults to the YAML seed.",
    )
    parser.add_argument(
        "--count-k",
        type=int,
        default=None,
        help="Override pathline.count_k; 16 gives 256 points per panel.",
    )
    parser.add_argument(
        "--high-ivd-percentile",
        type=float,
        default=85.0,
        help=(
            "High-IVD region threshold for the illustration; "
            "85 selects the top 15% of candidates (default: 85)."
        ),
    )
    parser.add_argument(
        "--field-grid",
        type=int,
        default=128,
        help="Resolution used to sample the analytical velocity field.",
    )
    parser.add_argument(
        "--lic-size",
        type=int,
        default=384,
        help="LIC raster width and height in pixels.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Raster output resolution.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used by IVD-based samplers (default: cpu).",
    )
    return parser.parse_args()


def representative_vatistas_spec() -> dict[str, object]:
    """Return a centred elliptical vortex with one dominant high-IVD region."""
    # Parameter order in the generator is:
    # [theta, sx, sy, rc, n, tx, ty].
    # A compact core with a steep outer transition keeps the top 15% IVD
    # candidates in one central ellipse instead of selecting outer lobes.
    params = np.asarray(
        [[0.30, 1.15, 0.85, 0.45, 12.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    return {
        "type": "vatistas",
        "params": params,
        "shapes": np.asarray([2], dtype=np.int64),  # S3_center_ccw
        "mask": np.asarray([1.0], dtype=np.float32),
        "source": "sampling_schematic",
    }


def unit_coordinates(points: np.ndarray) -> np.ndarray:
    """Map generator coordinates [-1, 1]^2 to the displayed unit square."""
    return 0.5 * (np.asarray(points, dtype=np.float64) + 1.0)


def unit_domain_mesh(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate adaptive points together with the full unit-square boundary.

    A Delaunay mesh only covers the convex hull of its vertices.  Adaptive
    seeding deliberately keeps points just inside the computational boundary,
    so triangulating the seeds alone leaves an unmeshed strip around the plot.
    Add evenly spaced boundary-support vertices to make the convex hull exactly
    ``[0, 1]^2``.  These vertices close the visualization mesh and are not
    sampling points.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Mesh points must be shaped [N, 2].")

    # Match the boundary-edge length to the nominal spacing of the interior
    # points.  Using more than the four corners avoids large, skinny triangles
    # between a corner and the nearest adaptive seed.
    boundary_segments = max(2, int(np.ceil(np.sqrt(max(len(points), 1)))))
    edge_coordinates = np.linspace(0.0, 1.0, boundary_segments + 1)
    boundary_points = np.concatenate(
        [
            np.column_stack(
                [edge_coordinates, np.zeros_like(edge_coordinates)]
            ),
            np.column_stack(
                [np.ones_like(edge_coordinates[1:]), edge_coordinates[1:]]
            ),
            np.column_stack(
                [edge_coordinates[-2::-1], np.ones_like(edge_coordinates[:-1])]
            ),
            np.column_stack(
                [np.zeros_like(edge_coordinates[-2:0:-1]),
                 edge_coordinates[-2:0:-1]]
            ),
        ],
        axis=0,
    )
    mesh_points = np.concatenate([points, boundary_points], axis=0)
    return mesh_points, _lic_mesh_edges(mesh_points)


def draw_adaptive_mesh_edges(ax, seeds: np.ndarray) -> list[Line2D]:
    """Draw the full-domain Delaunay mesh and return its line artists."""
    mesh_points, mesh_edges = unit_domain_mesh(seeds)
    return [
        ax.plot(
            mesh_points[[first, second], 0],
            mesh_points[[first, second], 1],
            color="#E8F8FF",
            linewidth=0.2,
            alpha=0.52,
            zorder=1.3,
        )[0]
        for first, second in mesh_edges
    ]


def build_shared_lic(
    cfg: EasyConfig,
    bounds: dict[str, object],
    device: torch.device,
    seed: int,
    field_grid: int,
    lic_size: int,
):
    """Build the common Vatistas field, label evaluator, and LIC background."""
    spec = representative_vatistas_spec()
    # The schematic deliberately uses a sharper profile than the training
    # pool's default n <= 6. Extend only this local evaluator's bounds so its
    # illustrative n is not silently clamped back to the training range.
    schematic_n = float(spec["params"][0, 4])
    schematic_bounds = dict(bounds)
    schematic_bounds["n"] = (
        min(bounds["n"][0], schematic_n),
        max(bounds["n"][1], schematic_n),
    )
    velocity_eval, label_eval = make_steady_eval(
        spec, schematic_bounds, device
    )
    coordinates = make_local_grid(field_grid, device).cpu().numpy()
    field = velocity_eval(coordinates)
    dx = float(coordinates[0, 1, 0] - coordinates[0, 0, 0])
    dy = float(coordinates[1, 0, 1] - coordinates[0, 0, 1])
    u, v = field[..., 0], field[..., 1]
    dv_dx = np.gradient(v, dx, axis=1, edge_order=2)
    du_dy = np.gradient(u, dy, axis=0, edge_order=2)
    raw_vorticity_magnitude = np.abs(dv_dx - du_dy)
    smoothing_sigma = max(0.75, field_grid / 64.0)
    vorticity_magnitude = gaussian_filter(
        raw_vorticity_magnitude,
        sigma=smoothing_sigma,
        mode="nearest",
    )
    value_min = float(vorticity_magnitude.min())
    value_max = float(vorticity_magnitude.max())
    if value_max > value_min + 1.0e-12:
        normalized_vorticity = (vorticity_magnitude - value_min) / (
            value_max - value_min
        )
    else:
        normalized_vorticity = np.zeros_like(vorticity_magnitude)
    # Give all very-low-vorticity regions one visual floor so zero-vorticity
    # curves blend into the background instead of appearing as dark arcs.
    background_floor = 0.12
    display_signal = np.maximum(normalized_vorticity, background_floor)
    displayed_vorticity = 0.30 + 0.55 * display_signal
    color_flow_image = np.clip(
        matplotlib.colormaps["viridis"](displayed_vorticity)[..., :3] * 255.0,
        0.0,
        255.0,
    ).astype(np.uint8)

    lic_cfg = read_lic_cfg(cfg)
    lic_cfg["enabled"] = True
    lic_cfg["image_size"] = int(lic_size)
    lic_cfg["noise_size"] = int(lic_size)
    lic_cfg["overlay_segmentation"] = True
    noise = _make_lic_noise(lic_cfg, seed)
    lic_image = render_lic_image(field, lic_cfg, noise, label_eval)

    label_coordinates = make_local_grid(lic_size, device).cpu().numpy()
    label = label_eval(label_coordinates)
    return velocity_eval, lic_image, color_flow_image, label


def sample_all_methods(
    cfg: EasyConfig,
    velocity_eval,
    seed: int,
    count_k: int | None,
) -> tuple[dict[str, np.ndarray], dict[int, np.ndarray]]:
    """Sample every method and retain adaptive CVT states zero through five."""
    base_cfg = read_pathline_cfg(cfg)
    if count_k is not None:
        base_cfg["count_k"] = int(count_k)

    samples = {}
    adaptive_iteration_samples = {}
    for index, method in enumerate(SAMPLING_METHODS):
        method_cfg = deepcopy(base_cfg)
        method_cfg["sampling"] = method
        # Independent deterministic streams prevent one method from changing
        # another method's result when its implementation consumes more RNG.
        method_seed = seed + 1009 * (index + 1)
        if method != FEATURE_ADAPTIVE_MESH_SEEDING:
            samples[method] = _pathline_seedings(
                method_cfg,
                np.random.default_rng(method_seed),
                velocity_eval=velocity_eval,
                time=0.0,
            )
            continue

        # CVT relaxation is deterministic after the initial generators are
        # selected. Restarting from the same RNG stream with an iteration cap
        # of 0, 1, ..., 5 therefore captures the initialization and the exact
        # state after each relaxation round.
        for iteration in ADAPTIVE_CVT_ITERATIONS:
            iteration_cfg = deepcopy(method_cfg)
            iteration_cfg["feature_mesh_iterations"] = iteration
            adaptive_iteration_samples[iteration] = _pathline_seedings(
                iteration_cfg,
                np.random.default_rng(method_seed),
                velocity_eval=velocity_eval,
                time=0.0,
            )

        configured_iterations = int(method_cfg["feature_mesh_iterations"])
        if configured_iterations in adaptive_iteration_samples:
            samples[method] = adaptive_iteration_samples[
                configured_iterations
            ].copy()
        else:
            samples[method] = _pathline_seedings(
                method_cfg,
                np.random.default_rng(method_seed),
                velocity_eval=velocity_eval,
                time=0.0,
            )
    return samples, adaptive_iteration_samples


def trace_adaptive_initialization(
    cfg: EasyConfig,
    velocity_eval,
    seed: int,
    count_k: int | None,
    expected_iter0: np.ndarray,
) -> AdaptiveInitialization:
    """Replay the monitor construction with the adaptive sampler's RNG stream.

    Candidate generation, IVD evaluation and farthest insertion use the
    generator's own functions. The partition/monitor calculations below mirror
    ``_feature_adaptive_mesh_seedings`` to expose its intermediate arrays for
    plotting. Require bitwise equality with the sampler's shuffled iter0 so a
    future generator change cannot silently produce a misleading diagram.
    """
    pcfg = read_pathline_cfg(cfg)
    if count_k is not None:
        pcfg["count_k"] = int(count_k)
    method_index = SAMPLING_METHODS.index(FEATURE_ADAPTIVE_MESH_SEEDING)
    rng = np.random.default_rng(seed + 1009 * (method_index + 1))
    resolution = pcfg["feature_mesh_resolution"]
    margin = pcfg["feature_boundary_margin"]
    lower, upper = -1.0 + margin, 1.0 - margin
    spacing = (upper - lower) / resolution
    axis = lower + (np.arange(resolution) + 0.5) * spacing
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    structured = np.column_stack([xx.ravel(), yy.ravel()])
    candidates = _feature_mesh_candidates(resolution, margin, rng)
    ivd = _autograd_ivd_at_points(
        velocity_eval.torch_eval, candidates, 0.0,
        pcfg["ivd_boundary_samples"],
    )
    ivd = np.nan_to_num(ivd, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.lexsort((rng.random(len(candidates)), ivd))
    high_candidate_count = int(round(
        len(candidates)
        * (1.0 - pcfg["feature_high_ivd_percentile"] / 100.0)
    ))
    high_ids = order[-high_candidate_count:]
    low_ids = order[:-high_candidate_count]
    contrast = np.maximum(ivd - float(np.min(ivd)), 0.0)
    scale = float(np.percentile(contrast, pcfg["feature_clip_percentile"]))
    eps = np.finfo(np.float64).eps
    feature = (
        np.full(len(candidates), 0.5, dtype=np.float64)
        if not np.isfinite(scale) or scale <= eps
        else np.clip(contrast / scale, 0.0, 1.0)
    )
    ratio = pcfg["feature_mesh_size_ratio"]
    power = pcfg["feature_monitor_power"]
    high_size = ratio - (ratio - 1.0) * feature[high_ids] ** power
    low_size = ratio - (ratio - 1.0) * (1.0 - feature[low_ids]) ** power
    density = np.empty(len(candidates), dtype=np.float64)
    for ids, sizes in ((high_ids, high_size), (low_ids, low_size)):
        weights = 1.0 / np.maximum(sizes, eps) ** 2
        density[ids] = weights / max(float(np.max(weights)), eps)

    target_count = _common_existing_seed_count(pcfg["count_k"])
    high_count = int(round(target_count * pcfg["feature_high_fraction"]))
    high_seeds = _weighted_cvt_mesh_points(
        candidates[high_ids], high_size, high_count, rng, iterations=0
    )
    low_seeds = _weighted_cvt_mesh_points(
        candidates[low_ids], low_size, target_count - high_count, rng,
        iterations=0,
    )
    iter0 = np.concatenate([high_seeds, low_seeds], axis=0)
    rng.shuffle(iter0)
    if not np.array_equal(iter0, expected_iter0):
        raise RuntimeError(
            "Initialization diagram does not match the sampler's iter0. "
            "Update the diagnostic replay to match the generator."
        )
    return AdaptiveInitialization(
        pcfg, structured, candidates, ivd, high_ids, low_ids, feature, density,
        high_seeds, low_seeds, iter0,
    )


def save_adaptive_initialization_panels(
    figure, axes, output_dir: Path, dpi: int,
) -> tuple[Path, ...]:
    """Export stages with external colorbars, without titles or captions."""
    names = (
        "01_structured_candidate_grid",
        "02_jitter_candidates",
        "03_measure_ivd",
        "04_split_regions",
        "05_density_weights",
        "06_insertion_overview",
        "07_regional_seed_quotas",
        "08_initial_adaptive_mesh",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    # Main-axes text includes titles and captions; colorbar labels are nested
    # in their own inset axes and remain visible during standalone exports.
    annotations = [artist for ax in axes for artist in panel_annotations(ax)]
    # Hide the annotations before measuring the crop to remove their margins.
    visibility = [(artist, artist.get_visible()) for artist in (
        *figure.texts, *figure.artists, *figure.axes, *annotations,
    )]
    colorbar_layouts = [
        (ax, box, box.get_axes_locator(), box.get_position().frozen(),
         box.get_in_layout())
        for ax in axes for box in ax.child_axes
        if box.get_gid() == "adaptive_initialization_colorbar"
    ]
    paths = []
    try:
        for artist, _ in visibility:
            artist.set_visible(False)
        # Center each colorbar below its panel for standalone exports. The
        # nested bar and its labels follow the containing axes automatically.
        for ax, box, _, _, _ in colorbar_layouts:
            panel = ax.get_position()
            box.set_axes_locator(None)
            box.set_position((
                panel.x0 + 0.10 * panel.width,
                panel.y0 - 0.19 * panel.height,
                0.80 * panel.width,
                0.145 * panel.height,
            ))
            # set_position disables layout participation; include the external
            # legend again so the export crop contains its full extent.
            box.set_in_layout(True)
        for name, ax in zip(names, axes):
            ax.set_visible(True)
            figure.canvas.draw()
            bounds = ax.get_tightbbox(figure.canvas.get_renderer())
            bounds = bounds.transformed(figure.dpi_scale_trans.inverted()).padded(0.10)
            for suffix in ("png", "svg"):
                path = output_dir / f"{name}.{suffix}"
                figure.savefig(path, dpi=dpi, facecolor="white", bbox_inches=bounds)
                paths.append(path)
            ax.set_visible(False)
    finally:
        for _, box, locator, position, in_layout in colorbar_layouts:
            box.set_position(position)
            box.set_axes_locator(locator)
            box.set_in_layout(in_layout)
        for artist, visible in visibility:
            artist.set_visible(visible)
    return tuple(paths)


def save_initial_adaptive_points(figure, ax, scatter, output_dir, dpi):
    """Export only panel 08's initial seeds, with its exact scale and crop."""
    bounds = ax.get_window_extent().transformed(
        figure.dpi_scale_trans.inverted()
    ).padded(0.10)
    visibility = [(artist, artist.get_visible()) for artist in (
        *figure.axes, *figure.texts, *figure.artists, *ax.get_children(),
    )]
    clip_on = scatter.get_clip_on()
    paths = []
    try:
        for artist, _ in visibility:
            artist.set_visible(False)
        ax.set_visible(True)
        scatter.set_visible(True)
        # The transparent padding has room for full circular markers at the
        # domain boundary; do not trim them to the original mesh axes frame.
        scatter.set_clip_on(False)
        for suffix in ("png", "svg"):
            path = output_dir / f"08_initial_adaptive_mesh_points.{suffix}"
            figure.savefig(path, dpi=dpi, transparent=True,
                           facecolor="none", edgecolor="none", bbox_inches=bounds)
            paths.append(path)
    finally:
        scatter.set_clip_on(clip_on)
        for artist, visible in visibility:
            artist.set_visible(visible)
    return tuple(paths)


def panel_annotations(ax):
    """Hide standalone titles/captions while retaining calculation labels."""
    return [ax.title, ax._left_title, ax._right_title] + [
        artist for artist in ax.texts
        if artist.get_gid() == "initialization_caption"
    ]


def draw_insertion_details(figure, trace, *, region="H", y_offset=0.0):
    """Expand a regional insertion, showing completed H seeds when R is L."""
    ink, muted = "#142D43", "#526779"
    cyan, pink = "#00BED5", "#D21EAF"
    use_high = region == "H"
    ids = trace.high_ids if use_high else trace.low_ids
    insertion = trace.high_seeds if use_high else trace.low_seeds
    candidates, density = trace.candidates[ids], trace.density[ids]
    steps = trace_farthest_insertion(candidates, density, insertion)
    # Panel G shows the selection of p16 from the first 15 fixed seeds.
    middle_k = min(15, len(insertion) - 1)
    late_k = min(24, len(insertion) - 1) if use_high else len(insertion) - 1
    points = unit_coordinates(candidates)
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    half_span = float(np.ptp(points, axis=0).max()) * 0.62
    limits = (center - half_span, center + half_span)
    if not use_high:
        # The low-IVD region surrounds the central high-IVD ellipse. Retain
        # the whole domain, with a clear band above it for the equations.
        limits = (np.asarray([-0.16, -0.08]), np.asarray([1.16, 1.24]))
    first, second = steps[1], steps[2]
    weighted_id = first.next_id
    distance_id = int(np.argmax(first.distance_sq))
    # Use an unselected candidate that is closer to p2, to make min() visible.
    reduction = first.distance_sq - second.distance_sq
    probe_id = int(np.argmax(reduction * second.distance_sq))
    changed = reduction > 1.0e-12

    figure_width, figure_height = figure.get_size_inches()

    def ftext(x, y, value, **kwargs):
        return figure.text(x / figure_width, (y + y_offset) / figure_height,
                           value, **kwargs)

    section_title = (
        "06 H  |  Weighted farthest insertion in the high-IVD region"
        if use_high else
        f"06 L  |  Low-IVD insertion after all {len(trace.high_seeds)} H seeds are fixed"
    )
    description = (
        "Actual H-region candidates, enlarged. The density stays fixed; "
        "nearest distances and scores change as seeds are added."
        if use_high else
        "Orange: completed H seeds. Start a new selected set in L; "
        "all p and S below belong to L. Distances use only selected L seeds."
    )
    transition = (
        f"Continue H insertion to its quota of {len(trace.high_seeds)} seeds. "
        "Then fix those positions and start the independent L insertion below."
        if use_high else
        f"Select p{len(insertion)} to finish L: {len(insertion)} L seeds + "
        f"{len(trace.high_seeds)} fixed H seeds. Merge + shuffle -> 07 -> 08 / iter0. "
        "CVT centroid updates have not started."
    )

    header_texts = [
        ftext(0.7, 12.0, section_title,
              fontsize=21, fontweight="semibold", color=ink),
        ftext(0.7, 11.62, description,
              fontsize=11.5, color=muted),
    ]
    shared_texts = [
        ftext(0.7, 11.17,
              rf"$R={region}\subset[-1,1]^2,\quad "
              rf"S_0=\varnothing,\quad S_k\subset {region},\quad "
              r"q_i(k)=\rho_i D_i^2(k),\quad "
              r"p_{k+1}=x_{\arg\max_{i:\,x_i\notin S_k}q_i(k)},\quad "
              r"D_i^2(k+1)=\min\{D_i^2(k),\|x_i-p_{k+1}\|^2\}$",
              fontsize=13.5, color=ink),
        ftext(0.7, 0.55,
              "Star: random first seed    Cyan: already selected    Magenta ring: next seed    "
              "Dashed segment: nearest-seed distance    Diamond B: distance-only maximum"
              + ("" if use_high else f"    Orange: {len(trace.high_seeds)} fixed H seeds"),
              fontsize=10, color=ink),
        ftext(0.7, 0.21, transition,
              fontsize=10, color=muted),
    ]
    # Keep the narrative footer in the combined diagram only. Standalone
    # stage 06 retains the mathematical annotations and its symbol key.
    header_texts.append(shared_texts.pop())
    axes, names = [], []

    def panel(index, name, title, formula, values, cmap, max_value=None):
        col, row = index % 4, index // 4
        left = np.linspace(0.7, 15.2, 4)[col] + 0.225
        bottom = 7.0 if row == 0 else 1.75
        ax = figure.add_axes((left / figure_width, (bottom + y_offset) / figure_height,
                              3.65 / figure_width, 3.65 / figure_height))
        ax.set(xlim=(limits[0][0], limits[1][0]),
               ylim=(limits[0][1], limits[1][1]), aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#F5F8FB")
        for spine in ax.spines.values():
            spine.set_color("#B9C9D5")
        if not use_high:
            name = f"06{chr(105 + index)}_low_ivd_{name.split('_', 1)[1]}"
            title = f"{chr(73 + index)}  {title.split('  ', 1)[1]}"
            # These finished H seeds provide context only. They are never
            # included in the L distance matrix or the eligible candidate set.
            fixed_h = unit_coordinates(trace.high_seeds)
            ax.scatter(fixed_h[:, 0], fixed_h[:, 1], s=7, c="#DB7023",
                       edgecolors="none", alpha=0.70, zorder=2)
            ax.text(0.50, 0.50, rf"$N_H={len(trace.high_seeds)}$", ha="center",
                    fontsize=9, color="#9B4F16",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2),
                    zorder=3)
        ax.set_title(title, loc="left", fontsize=12, fontweight="semibold",
                     color=ink, pad=12)
        visible = np.ma.masked_less(values, 0)
        vmax = max(float(np.max(values)) if max_value is None else max_value,
                   np.finfo(float).eps)
        candidate_size = min(15, 9500 / len(points))
        artist = ax.scatter(points[:, 0], points[:, 1], c=visible, s=candidate_size,
                            cmap=cmap, vmin=0, vmax=vmax, linewidths=0, zorder=1)
        ax.text(0.03, 0.96, formula, va="top", transform=ax.transAxes,
                fontsize=15, color=ink,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.96, pad=8),
                zorder=6)
        # All numerical ranges are explicit, since score magnitudes shrink.
        cax = ax.inset_axes((0.05, -0.10, 0.90, 0.026))
        bar = figure.colorbar(artist, cax=cax, orientation="horizontal")
        bar.set_ticks([0, vmax])
        bar.set_ticklabels(["0", f"{vmax:.4g}"])
        bar.ax.tick_params(labelsize=8, length=2, pad=2)
        bar.outline.set_visible(False)
        axes.append(ax)
        names.append(name)
        return ax

    def metrics(ax, lines):
        box = ax.inset_axes((-0.04, -0.31, 1.13, 0.16))
        box.set_axis_off()
        box.text(0, 1, "\n".join(lines), fontsize=10, va="top", color=ink,
                 linespacing=1.6, transform=box.transAxes)

    def mark(ax, index, label=None, *, color=pink, marker="o", size=135):
        x, y = points[index]
        ax.scatter(x, y, s=size, marker=marker, facecolors="none",
                   edgecolors=color, linewidths=1.8, zorder=5)
        if label:
            # Place labels outward from the region centre, away from its points.
            offset = np.asarray([10 if x >= center[0] else -10,
                                 10 if y >= center[1] else -13])
            ax.annotate(label, (x, y), xytext=tuple(offset),
                        textcoords="offset points", fontsize=11, color=color,
                        ha="left" if offset[0] > 0 else "right", zorder=7,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1))

    def seeds(ax, state, labels=True):
        selected = points[state.selected]
        ax.scatter(selected[:, 0], selected[:, 1], s=35, c=cyan,
                   edgecolors=ink, linewidths=0.7, zorder=4)
        ax.scatter(*selected[0], s=145, marker="*", c=ink,
                   edgecolors="white", linewidths=0.7, zorder=5)
        if labels and len(selected) <= 4:
            for k, index in enumerate(state.selected):
                mark(ax, index, rf"$p_{k + 1}$", color=ink, size=0)

    def segment(ax, a, b, color=muted):
        ax.plot(points[[a, b], 0], points[[a, b], 1], "--",
                color=color, linewidth=1.4, zorder=3)

    probability = density / density.sum()
    ax = panel(0, "06a_first_seed_probability", "A  Draw the first seed",
               r"$P(i)=\rho_i/\sum_{j\in R}\rho_j$", probability, "magma")
    seeds(ax, first)
    p1 = first.selected[0]
    metrics(ax, [rf"$\rho(p_1)={density[p1]:.4f},\quad "
                 rf"\sum_R\rho={density.sum():.3f}$",
                 rf"$P(p_1)={probability[p1]:.5f},\quad |S_1|=1$"])

    ax = panel(1, "06b_nearest_distance", "B  Measure nearest distances",
               r"$D_i^2(1)=\|x_i-p_1\|^2$", first.distance_sq, "Blues")
    seeds(ax, first)
    for index, letter, color, marker in ((weighted_id, "A", pink, "o"),
                                         (distance_id, "B", ink, "D")):
        segment(ax, p1, index, color)
        mark(ax, index, letter, color=color, marker=marker)
    metrics(ax, [rf"$D_A^2={first.distance_sq[weighted_id]:.4f},\quad "
                 rf"D_B^2={first.distance_sq[distance_id]:.4f}$",
                 r"$B=\arg\max_i D_i^2(1)$"])

    ax = panel(2, "06c_weighted_score_comparison", "C  Weight, compare, select p2",
               r"$q_i(1)=\rho_i D_i^2(1)$", first.scores, "YlOrRd")
    seeds(ax, first)
    mark(ax, weighted_id, r"$A=p_2$")
    mark(ax, distance_id, "B", color=ink, marker="D")
    metrics(ax, [
        rf"$A:\ {first.distance_sq[weighted_id]:.4f}\times "
        rf"{density[weighted_id]:.4f}={first.scores[weighted_id]:.4f}$",
        rf"$B:\ {first.distance_sq[distance_id]:.4f}\times "
        rf"{density[distance_id]:.4f}={first.scores[distance_id]:.4f}$",
    ])

    ax = panel(3, "06d_update_nearest_distance", "D  Insert p2; update every distance",
               r"$D_i^2(2)=\min\{D_i^2(1),\|x_i-p_2\|^2\}$",
               second.distance_sq, "Blues", float(first.distance_sq.max()))
    # Pale green squares outline the candidates whose nearest distance shrinks.
    ax.scatter(points[changed, 0], points[changed, 1],
               s=min(27, 16500 / len(points)), marker="s",
               facecolors="none", edgecolors="#26A67C", linewidths=0.5, zorder=2)
    seeds(ax, second)
    segment(ax, probe_id, p1)
    segment(ax, probe_id, weighted_id, pink)
    mark(ax, probe_id, "X", color="#087B57", marker="s", size=95)
    new_distance = float(np.sum((candidates[probe_id] - candidates[weighted_id]) ** 2))
    metrics(ax, [rf"$X:\ \min({first.distance_sq[probe_id]:.4f}, "
                 rf"{new_distance:.4f})={second.distance_sq[probe_id]:.4f}$",
                 f"Green squares: {np.count_nonzero(changed)} distances decrease."])

    for index, k in enumerate((min(2, len(insertion) - 1),
                               min(3, len(insertion) - 1), middle_k, late_k), start=4):
        state = steps[k]
        chosen = state.next_id
        title = (f"{chr(65 + index)}  Re-score with {k} seeds; select p{k + 1}"
                 if index < 7 else f"H  Repeat: {k} seeds -> p{k + 1}")
        if not use_high and index == 7:
            title = f"H  Finish L: {k} seeds -> p{k + 1}"
        ax = panel(index, f"06{chr(97 + index)}_score_after_{k:02d}_seeds", title,
                   rf"$q_i({k})=\rho_i\min_{{p\in S_{{{k}}}}}\|x_i-p\|^2$",
                   state.scores, "YlOrRd")
        seeds(ax, state)
        nearest = state.selected[np.argmin(np.sum(
            (candidates[state.selected] - candidates[chosen]) ** 2, axis=1))]
        segment(ax, chosen, nearest, pink)
        mark(ax, chosen, rf"$p_{{{k + 1}}}$")
        metrics(ax, [rf"$q_{{\max}}={state.distance_sq[chosen]:.4f}\times "
                     rf"{density[chosen]:.4f}={state.scores[chosen]:.4f}$",
                     rf"$|S_{{{k}}}|={k}\ \longrightarrow\ |S_{{{k + 1}}}|={k + 1}$"])

    # Carry the same visual reading order into the standalone calculation sheet.
    for i in (0, 1, 2, 4, 5, 6):
        start, end = axes[i].get_position(), axes[i + 1].get_position()
        arrow = FancyArrowPatch(
            (start.x1 + 0.007, (start.y0 + start.y1) / 2),
            (end.x0 - 0.007, (end.y0 + end.y1) / 2),
            arrowstyle="->", mutation_scale=14, linewidth=1.3,
            color="#7592A7", transform=figure.transFigure,
        )
        figure.add_artist(arrow)
        shared_texts.append(arrow)
    turn_points = np.asarray([
        (19.18, 8.825), (19.48, 8.825), (19.48, 5.88),
        (0.46, 5.88), (0.46, 3.575), (0.82, 3.575),
    ])
    turn_points[:, 1] += y_offset
    turn_points /= (figure_width, figure_height)
    turn = FancyArrowPatch(
        path=MatplotlibPath(turn_points), arrowstyle="->", mutation_scale=14,
        linewidth=1.3, color="#7592A7", transform=figure.transFigure,
    )
    figure.add_artist(turn)
    shared_texts.append(turn)
    return axes, names, header_texts, shared_texts


def save_insertion_details(figure, axes, names, shared_texts, output_dir, dpi,
                           stem="06_weighted_farthest_insertion"):
    """Export the expanded stage 06 and each calculation without captions."""
    annotations = [artist for ax in axes for artist in panel_annotations(ax)]
    visibility = [(artist, artist.get_visible()) for artist in (
        *figure.axes, *figure.texts, *figure.artists, *annotations,
    )]
    outputs = []
    try:
        for artist, _ in visibility:
            artist.set_visible(False)
        # The full stage retains its equation/key, with no overall title.
        for artist in [*axes, *shared_texts]:
            artist.set_visible(True)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        bounds = Bbox.union([artist.get_tightbbox(renderer)
                             for artist in [*axes, *shared_texts]])
        bounds = bounds.transformed(figure.dpi_scale_trans.inverted()).padded(0.10)
        for suffix in ("png", "svg"):
            path = output_dir / f"{stem}.{suffix}"
            figure.savefig(path, dpi=dpi, facecolor="white", bbox_inches=bounds)
            outputs.append(path)
        for artist in [*axes, *shared_texts]:
            artist.set_visible(False)
        for name, ax in zip(names, axes):
            ax.set_visible(True)
            figure.canvas.draw()
            bounds = ax.get_tightbbox(figure.canvas.get_renderer())
            bounds = bounds.transformed(figure.dpi_scale_trans.inverted()).padded(0.10)
            for suffix in ("png", "svg"):
                path = output_dir / f"{name}.{suffix}"
                figure.savefig(path, dpi=dpi, facecolor="white", bbox_inches=bounds)
                outputs.append(path)
            ax.set_visible(False)
    finally:
        for artist, visible in visibility:
            artist.set_visible(visible)
    return tuple(outputs)


def save_adaptive_mesh_iterations(
    figure, mesh_ax, mesh_lines, mesh_scatter,
    iteration_samples: dict[int, np.ndarray], output: Path, dpi: int,
) -> tuple[Path, ...]:
    """Continue from panel 08 using its exact artists, frame and crop."""
    initial_seeds = np.asarray(mesh_scatter.get_offsets()).copy()
    if not np.array_equal(initial_seeds, unit_coordinates(iteration_samples[0])):
        raise ValueError("Iteration zero must match panel 08's initial seeds.")

    annotations = [artist for artist in mesh_ax.get_children()
                   if isinstance(artist, Text)]
    visibility = [(artist, artist.get_visible()) for artist in (
        *figure.axes, *figure.texts, *figure.artists, *annotations,
    )]
    paths = []
    movement_artists = []
    output_stem = output.with_suffix("")
    try:
        for artist, _ in visibility:
            artist.set_visible(False)
        mesh_ax.set_visible(True)
        figure.canvas.draw()
        bounds = mesh_ax.get_tightbbox(figure.canvas.get_renderer())
        bounds = bounds.transformed(figure.dpi_scale_trans.inverted()).padded(0.10)
        previous_seeds = initial_seeds
        for iteration in ADAPTIVE_CVT_ITERATIONS:
            seeds = unit_coordinates(iteration_samples[iteration])
            if iteration > 0:
                for artist in [*mesh_lines, *movement_artists]:
                    artist.remove()
                movement_artists = []
                mesh_lines = draw_adaptive_mesh_edges(mesh_ax, seeds)
                mesh_scatter.set_offsets(seeds)
                displacement = seeds - previous_seeds
                moved = np.linalg.norm(displacement, axis=1) > 1.0e-12
                if np.any(moved):
                    movement_artists.append(mesh_ax.quiver(
                        previous_seeds[moved, 0], previous_seeds[moved, 1],
                        displacement[moved, 0], displacement[moved, 1],
                        angles="xy", scale_units="xy", scale=1.0,
                        color="#FF2D20", width=0.0035, headwidth=4.0,
                        headlength=5.0, headaxislength=4.5,
                        alpha=0.95, zorder=3.7,
                    ))
                    movement_artists.append(mesh_ax.scatter(
                        seeds[moved, 0], seeds[moved, 1], s=28,
                        c="#FF2D20", edgecolors="#FFFFFF", linewidths=0.6,
                        alpha=0.98, zorder=4.0,
                    ))
            for suffix in ("png", "svg"):
                path = output_stem.parent / (
                    f"{output_stem.name}_feature_adaptive_mesh_seeding_"
                    f"iter{iteration}.{suffix}"
                )
                figure.savefig(path, dpi=dpi, facecolor="white", bbox_inches=bounds)
                paths.append(path)
            previous_seeds = seeds
    finally:
        for artist in [*mesh_lines, *movement_artists]:
            artist.remove()
        draw_adaptive_mesh_edges(mesh_ax, initial_seeds)
        mesh_scatter.set_offsets(initial_seeds)
        for artist, visible in visibility:
            artist.set_visible(visible)
    return tuple(paths)


def draw_adaptive_initialization(
    trace: AdaptiveInitialization,
    lic_image: np.ndarray,
    label: np.ndarray,
    output: Path,
    dpi: int,
    adaptive_iteration_samples: dict[int, np.ndarray],
) -> tuple[Path, ...]:
    """Draw eight initialization stages and a quantitative insertion expansion."""
    figure = plt.figure(figsize=(20.0, 12.5), facecolor="white")
    ink, muted = "#142D43", "#526779"
    high_color, low_color = "#DB7023", "#357EB5"
    figure.text(
        0.035, 0.965, "From a structured grid to feature-adaptive seeds",
        fontsize=23, fontweight="semibold", color=ink,
    )
    figure.text(
        0.035, 0.930,
        "Vatistas field at t = 0  |  Follow 1-4 across the top, then 5-8 "
        "back along row 2; H then L details below  |  CVT centroid updates: 0",
        fontsize=12, color=muted,
    )
    # Four square panels per row; the U-shaped order keeps every transition
    # adjacent, including the move from the top row to the bottom row.
    width, height = 0.205, 0.328
    lefts = np.linspace(0.035, 0.760, 4)
    axes = [
        figure.add_axes((x, y, width, height))
        for y, xs in ((0.550, lefts), (0.115, lefts[::-1]))
        for x in xs
    ]
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#F5F8FB")
        for spine in ax.spines.values():
            spine.set_color("#B9C9D5")

    def heading(index, title, caption):
        ax = axes[index]
        ax.set_title(
            f"{index + 1:02d}  {title}", loc="left", fontsize=13,
            fontweight="semibold", color=ink, pad=11,
        )
        caption_artist = ax.text(
            0.0, -0.035, caption, transform=ax.transAxes, va="top",
            fontsize=10.5, color=muted, linespacing=1.5,
        )
        caption_artist.set_gid("initialization_caption")

    def dots(ax, points, **kwargs):
        points = unit_coordinates(points)
        return ax.scatter(
            points[:, 0], points[:, 1], linewidths=0, **kwargs
        )

    def scale_bar(ax, artist, title, ticks):
        # A compact inset includes its own light background for legibility.
        box = ax.inset_axes((0.04, 0.035, 0.61, 0.145), zorder=5)
        box.set_gid("adaptive_initialization_colorbar")
        box.set_facecolor((1, 1, 1, 0.94))
        box.set_xticks([])
        box.set_yticks([])
        for spine in box.spines.values():
            spine.set_visible(False)
        bar = box.inset_axes((0.08, 0.47, 0.84, 0.17))
        colorbar = figure.colorbar(artist, cax=bar, orientation="horizontal")
        colorbar.set_ticks(ticks)
        colorbar.ax.tick_params(labelsize=8, length=2, pad=2)
        colorbar.outline.set_visible(False)
        box.text(0.08, 0.78, title, fontsize=8, color=ink)

    pcfg = trace.config
    resolution = pcfg["feature_mesh_resolution"]
    total = len(trace.iter0)
    high_n, low_n = len(trace.high_seeds), len(trace.low_seeds)
    candidate_count = len(trace.candidates)
    point_size = min(4.0, 7500.0 / candidate_count)

    def candidate_zoom(ax, show_jitter):
        structured = unit_coordinates(trace.structured)
        jittered = unit_coordinates(trace.candidates)
        center = structured[np.argmin(np.sum(
            (structured - (0.72, 0.76)) ** 2, axis=1))]
        delta = (1.0 - pcfg["feature_boundary_margin"]) / resolution
        span = 2.0 * delta
        chosen = np.all(np.abs(structured - center) < 1.6 * delta, axis=1)
        zoom = ax.inset_axes((0.045, 0.65, 0.305, 0.305), zorder=8)
        zoom.set(xlim=(center[0] - span, center[0] + span),
                 ylim=(center[1] - span, center[1] + span), aspect="equal")
        zoom.set_xticks([])
        zoom.set_yticks([])
        zoom.set_facecolor("white")
        for spine in zoom.spines.values():
            spine.set_color("#688DA5")
        for offset in np.arange(-1.5, 2, 1) * delta:
            zoom.axvline(center[0] + offset, color="#DCE4EB", linewidth=0.8)
            zoom.axhline(center[1] + offset, color="#DCE4EB", linewidth=0.8)
        origin = structured[chosen]
        zoom.scatter(origin[:, 0], origin[:, 1], s=22,
                     facecolors="none" if show_jitter else "#279BBB",
                     edgecolors="#8B9EAD" if show_jitter else "#279BBB",
                     linewidths=0.8, zorder=3)
        if show_jitter:
            shifted = jittered[chosen]
            displacement = shifted - origin
            zoom.quiver(origin[:, 0], origin[:, 1], displacement[:, 0],
                        displacement[:, 1], angles="xy", scale_units="xy",
                        scale=1, width=0.012, color="#DB7023", zorder=4)
            zoom.scatter(shifted[:, 0], shifted[:, 1], s=15, c="#279BBB", zorder=5)
        ax.indicate_inset_zoom(zoom, edgecolor="#688DA5", linewidth=1.0)

    dots(axes[0], trace.structured, s=point_size, c="#279BBB")
    candidate_zoom(axes[0], False)
    heading(0, "Structured candidate grid",
            f"{resolution} x {resolution} cell-centred points "
            f"({candidate_count:,} candidates).\n"
            "Inset: one candidate at each background-cell centre.\n"
            "All overview panels use the same unit spatial domain.")

    dots(axes[1], trace.candidates, s=point_size, c="#279BBB")
    candidate_zoom(axes[1], True)
    heading(1, "Jitter the candidates",
            "Independent x/y jitter: +/- 0.20 grid spacings.\n"
            "Inset: open centres -> jittered candidates (arrows).\n"
            "Keep the same count, inside the boundary margin.")

    artist = dots(
        axes[2], trace.candidates, s=point_size * 1.8,
        c=trace.ivd, cmap="viridis", vmin=0,
        vmax=max(float(trace.ivd.max()), np.finfo(float).eps),
    )
    scale_bar(axes[2], artist, "IVD at candidates",
              [0, float(trace.ivd.max())])
    for index, label_position in ((int(np.argmax(trace.ivd)), (0.60, 0.78)),
                                   (int(np.argmin(trace.ivd)), (0.60, 0.18))):
        point = unit_coordinates(trace.candidates[index])
        axes[2].scatter(*point, s=65, facecolors="none", edgecolors=ink, zorder=4)
        axes[2].annotate(
            rf"$\mathrm{{IVD}}_i={trace.ivd[index]:.4f}$", point,
            xytext=label_position, textcoords="axes fraction", fontsize=9,
            color=ink, ha="center", bbox=dict(facecolor="white", edgecolor="none", pad=3),
            arrowprops=dict(arrowstyle="-", color=ink, linewidth=0.8), zorder=4,
        )
    heading(2, "Measure IVD",
            r"$\mathrm{IVD}=|\omega-\bar{\omega}|$"
            " at t = 0.\n"
            r"$\omega=\partial_x v-\partial_y u$ (autograd)." "\n"
            r"$\bar{\omega}=|\Omega|^{-1}\oint_{\partial\Omega}\mathbf{u}\cdot d\mathbf{l}$; "
            r"$\Omega=[-1,1]^2$.")

    for ids, color in ((trace.low_ids, low_color),
                       (trace.high_ids, high_color)):
        dots(axes[3], trace.candidates[ids], s=point_size * 1.4, c=color)
    for y, symbol, count, quota, color in (
        (0.94, "H", len(trace.high_ids), high_n, high_color),
        (0.86, "L", len(trace.low_ids), low_n, low_color),
    ):
        axes[3].text(0.06, y, rf"$|{symbol}|={count}\ \longrightarrow\ N_{symbol}={quota}$",
                      transform=axes[3].transAxes, color=color, fontsize=18, bbox=dict(facecolor="white", edgecolor="none", alpha=0.94, pad=2))
    heading(3, "Split regions / reserve seeds",
            f"Orange: top {100 - pcfg['feature_high_ivd_percentile']:g}% "
            f"IVD ({len(trace.high_ids):,} candidates).\n"
            f"Quotas ({100 * pcfg['feature_high_fraction']:g}% / "
            f"{100 * (1 - pcfg['feature_high_fraction']):g}%): "
            f"{high_n} high + {low_n} low.\n"
            "Regions are ranked by IVD.")

    artist = dots(
        axes[4], trace.candidates, s=point_size * 1.8,
        c=trace.density, cmap="magma", vmin=0, vmax=1,
    )
    scale_bar(axes[4], artist, r"$\rho$ (normalized within each region)", [0, 1])
    transfer_box = axes[4].inset_axes((0.66, 0.04, 0.30, 0.29), zorder=6)
    transfer_box.set_facecolor("white")
    transfer_box.set_xticks([])
    transfer_box.set_yticks([])
    for spine in transfer_box.spines.values():
        spine.set_visible(False)
    transfer = transfer_box.inset_axes((0.22, 0.22, 0.72, 0.70))
    transfer.set_facecolor("white")
    for ids, color in ((trace.high_ids, high_color), (trace.low_ids, low_color)):
        transfer.scatter(trace.feature[ids], trace.density[ids], s=2, c=color)
    transfer.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, 1], yticks=[0, 1])
    transfer.tick_params(labelsize=6, length=2, pad=1)
    transfer.set_xlabel("f", fontsize=8, labelpad=-4)
    transfer.set_ylabel(r"$\rho$", fontsize=8, labelpad=-4)
    for spine in transfer.spines.values():
        spine.set_color("#B9C9D5")
    ratio, power = pcfg["feature_mesh_size_ratio"], pcfg["feature_monitor_power"]
    heading(4, "Build the density weights",
            rf"$h_H={ratio:g}-({ratio:g}-1)f^{{{power:g}}}$; "
            rf"$h_L={ratio:g}-({ratio:g}-1)(1-f)^{{{power:g}}}$" "\n"
            r"$\rho=h^{-2}/\max_{\mathrm{region}}(h^{-2})$; "
            "smaller h gives larger weight.\n"
            "Inset: actual f-to-density values in H / L.")

    # Show a real prefix of the first nonempty regional insertion sequence.
    # The high region is completed before the low region starts in the sampler.
    if high_n:
        active_ids, insertion, region = trace.high_ids, trace.high_seeds, "high"
    else:
        active_ids, insertion, region = trace.low_ids, trace.low_seeds, "low"
    prefix_count = max(1, min(4, len(insertion) - 1))
    prefix = insertion[:prefix_count]
    distance_sq = np.min(np.sum(
        (trace.candidates[active_ids, None, :] - prefix[None, :, :]) ** 2,
        axis=2,
    ), axis=1)
    scores = distance_sq * trace.density[active_ids]
    dots(axes[5], trace.candidates, s=point_size, c="#DCE4EB", alpha=0.6)
    dots(axes[5], trace.candidates[active_ids], s=point_size * 2,
         c=scores, cmap="YlOrRd")
    dots(axes[5], prefix, s=25, c="#00BCD4", zorder=3)
    dots(axes[5], prefix[:1], s=110, marker="*", c=ink, zorder=4)
    for index, point in enumerate(unit_coordinates(prefix), start=1):
        axes[5].annotate(rf"$p_{index}$", point, xytext=(8, 7),
                         textcoords="offset points", fontsize=10, color=ink,
                         bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
    if prefix_count < len(insertion):
        next_point = unit_coordinates(insertion[prefix_count])
        axes[5].scatter(*next_point, s=140, facecolors="none",
                        edgecolors="#D21EAF", linewidths=2, zorder=4)
        axes[5].annotate(rf"$p_{{{prefix_count + 1}}}$", next_point, xytext=(8, 7),
                         textcoords="offset points", fontsize=10, color="#D21EAF")
    heading(5, "Weighted farthest insertion",
            rf"First seed: $P(i)\propto\rho_i$; then "
            r"$i^*=\arg\max_i(d_{i,\min}^2\rho_i)$." "\n"
            f"Actual {region}-region prefix: {prefix_count} seeds; star = first.\n"
            + ("Expanded below: distances, products, updates and repeats."
               if prefix_count < len(insertion)
               else "This region's seed quota is already filled."))

    dots(axes[6], trace.candidates, s=point_size, c="#DCE4EB", alpha=0.45)
    dots(axes[6], trace.high_seeds, s=15, c=high_color, zorder=3)
    dots(axes[6], trace.low_seeds, s=15, c=low_color, zorder=3)
    axes[6].text(0.05, 0.93, rf"$N_H={high_n}$", fontsize=18, color=high_color,
                 transform=axes[6].transAxes)
    axes[6].text(0.60, 0.93, rf"$N_L={low_n}$", fontsize=18, color=low_color,
                 transform=axes[6].transAxes)
    heading(6, "Complete both regional quotas",
            f"Finish high-region insertion ({high_n} orange seeds),\n"
            f"then low-region insertion ({low_n} blue seeds).\n"
            "Merge and shuffle; seed positions stay fixed.")

    axes[7].imshow(
        lic_image, extent=(0, 1, 0, 1), origin="lower",
        interpolation="bilinear", zorder=0,
    )
    initial_mesh_lines = draw_adaptive_mesh_edges(
        axes[7], unit_coordinates(trace.iter0))
    label_axis = np.linspace(0, 1, label.shape[0])
    for color, line_width in (("#171717", 3.5), ("#FFD400", 2.1)):
        axes[7].contour(label_axis, label_axis, label, levels=[0.0],
                        colors=[color], linewidths=[line_width], zorder=2)
    seeds = unit_coordinates(trace.iter0)
    initial_mesh_scatter = axes[7].scatter(
        seeds[:, 0], seeds[:, 1], s=18, c="#00D9FF",
        edgecolors="#071015", linewidths=0.45, alpha=0.94, zorder=3,
    )
    heading(7, "iter0: initial adaptive mesh",
            f"Same {total} seed coordinates as the iter0 SVG.\n"
            "Add Delaunay edges and boundary supports to draw.\n"
            "CVT centroid updates have not started.")

    for index in range(7):
        start, end = axes[index].get_position(), axes[index + 1].get_position()
        if index == 3:
            # Route the row turn in the right margin, clear of captions.
            arrow = FancyArrowPatch(
                (start.x1 + 0.007, start.y0 + height / 2),
                (end.x1 + 0.007, end.y0 + height / 2),
                connectionstyle="bar,fraction=-0.055", arrowstyle="->",
                mutation_scale=16, color="#7592A7", linewidth=1.6,
                transform=figure.transFigure,
            )
        else:
            direction = 1 if index < 3 else -1
            start_x = start.x1 + 0.005 if direction == 1 else start.x0 - 0.005
            end_x = end.x0 - 0.005 if direction == 1 else end.x1 + 0.005
            arrow = FancyArrowPatch(
                (start_x, start.y0 + height / 2),
                (end_x, end.y0 + height / 2),
                arrowstyle="->", mutation_scale=16, color="#7592A7",
                linewidth=1.6, transform=figure.transFigure,
            )
        figure.add_artist(arrow)

    figure.text(
        0.035, 0.020,
        r"IVD monitor: $c=\mathrm{IVD}-\min(\mathrm{IVD})$, "
        rf"$f=\mathrm{{clip}}(c/P_{{{pcfg['feature_clip_percentile']:g}}}(c),0,1)$ "
        r"(flat field: $f=0.5$).  "
        "Boundary supports are drawing vertices, not extra seeds.  "
        "Spatial coordinates are mapped from [-1, 1] to [0, 1].",
        fontsize=10, color=muted,
    )
    # Stack the overview, the H insertion, then the L insertion while keeping
    # every panel's physical size. Regions with fewer than two seeds need no
    # multi-step insertion sheet; their quota is still visible in the overview.
    detail_regions = [region for region, seeds in (
        ("H", trace.high_seeds), ("L", trace.low_seeds),
    ) if len(seeds) >= 2]
    sheet_count = 1 + len(detail_regions)
    overview_offset = len(detail_regions) / sheet_count
    overview_scale = 1 / sheet_count
    overview_positions = [ax.get_position().frozen() for ax in axes]
    figure.set_size_inches(20.0, 12.5 * sheet_count)
    for ax, position in zip(axes, overview_positions):
        ax.set_position((position.x0, overview_offset + overview_scale * position.y0,
                         position.width, overview_scale * position.height))
    for artist in figure.texts:
        x, y = artist.get_position()
        artist.set_position((x, overview_offset + overview_scale * y))
    for artist in figure.artists:
        artist.set_transform(
            Affine2D().scale(1, overview_scale).translate(0, overview_offset)
            + figure.transFigure
        )
    detail_groups = []
    for index, region in enumerate(detail_regions):
        detail_axes, detail_names, _, detail_shared_texts = draw_insertion_details(
            figure, trace, region=region,
            y_offset=12.5 * (len(detail_regions) - index - 1),
        )
        stem = ("06_weighted_farthest_insertion" if region == "H"
                else "06_low_ivd_farthest_insertion")
        detail_groups.append((detail_axes, detail_names, detail_shared_texts, stem))

    output_stem = output.with_suffix("")
    outputs = tuple(
        output_stem.parent / (
            f"{output_stem.name}_feature_adaptive_mesh_initialization.{suffix}"
        )
        for suffix in ("png", "svg")
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for path in outputs:
        figure.savefig(path, dpi=dpi, facecolor="white")
    outputs += save_adaptive_initialization_panels(
        figure, axes, output_stem.parent / "adaptive_seeds", dpi,
    )
    outputs += save_initial_adaptive_points(
        figure, axes[7], initial_mesh_scatter,
        output_stem.parent / "adaptive_seeds", dpi,
    )
    for detail_axes, detail_names, detail_shared_texts, stem in detail_groups:
        outputs += save_insertion_details(
            figure, detail_axes, detail_names, detail_shared_texts,
            output_stem.parent / "adaptive_seeds", dpi, stem=stem,
        )
    outputs += save_adaptive_mesh_iterations(
        figure, axes[7], initial_mesh_lines, initial_mesh_scatter,
        adaptive_iteration_samples, output, dpi,
    )
    plt.close(figure)
    return outputs


def draw_figure(
    lic_image: np.ndarray,
    color_flow_image: np.ndarray,
    label: np.ndarray,
    samples: dict[str, np.ndarray],
    output: Path,
    dpi: int,
) -> tuple[Path, ...]:
    """Draw a one-by-four comparison on the shared labelled LIC."""
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(21.0, 6.0),
        sharex=True,
        sharey=True,
    )
    figure.patch.set_facecolor("#FFFFFF")
    label_axis = np.linspace(0.0, 1.0, label.shape[0])

    for ax, method in zip(axes, SAMPLING_METHODS):
        seeds = unit_coordinates(samples[method])
        ax.imshow(
            lic_image,
            extent=(0.0, 1.0, 0.0, 1.0),
            origin="lower",
            interpolation="bilinear",
            zorder=0,
        )

        if method == GRID_CROSS_SEEDING:
            if len(seeds) % 4:
                raise ValueError(
                    "GRID_CROSS_SEEDING must contain groups of four points."
                )
            # The generator stores every cross as left, down, right, up.
            # Join opposite arms without adding an artificial centre point.
            for left, down, right, up in seeds.reshape(-1, 4, 2):
                ax.plot(
                    [left[0], right[0]],
                    [left[1], right[1]],
                    color="#F2F4F8",
                    linestyle=(0, (2.0, 2.0)),
                    linewidth=0.65,
                    alpha=0.72,
                    zorder=1.2,
                )
                ax.plot(
                    [down[0], up[0]],
                    [down[1], up[1]],
                    color="#F2F4F8",
                    linestyle=(0, (2.0, 2.0)),
                    linewidth=0.65,
                    alpha=0.72,
                    zorder=1.2,
                )

        if method == RECTANGULAR_CLUSTER4_RANDOM:
            ax.axvline(
                0.5,
                color="#F2F4F8",
                linestyle=(0, (5.0, 4.0)),
                linewidth=1.1,
                alpha=0.9,
                zorder=1.2,
            )
            ax.axhline(
                0.5,
                color="#F2F4F8",
                linestyle=(0, (5.0, 4.0)),
                linewidth=1.1,
                alpha=0.9,
                zorder=1.2,
            )

        if method == FEATURE_ADAPTIVE_MESH_SEEDING:
            # Delaunay triangulation of only the seeds stops at their convex
            # hull.  Include non-sampling boundary support vertices so the
            # displayed mesh covers the entire computational domain.
            draw_adaptive_mesh_edges(ax, seeds)

        # A dark under-stroke keeps the yellow ground-truth boundary visible
        # against bright LIC streaks; the LIC renderer already fills the core
        # with a translucent yellow label.
        ax.contour(
            label_axis,
            label_axis,
            label,
            levels=[0.0],
            colors=["#171717"],
            linewidths=[3.5],
            zorder=2,
        )
        ax.contour(
            label_axis,
            label_axis,
            label,
            levels=[0.0],
            colors=["#FFD400"],
            linewidths=[2.1],
            zorder=2.1,
        )
        ax.scatter(
            seeds[:, 0],
            seeds[:, 1],
            s=18,
            c="#00D9FF",
            edgecolors="#071015",
            linewidths=0.45,
            alpha=0.94,
            zorder=3,
        )

        ax.set_title(
            SAMPLING_TITLES[method],
            color="#111827",
            fontsize=11,
            fontweight="semibold",
            pad=8,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("#FFFFFF")
        ax.tick_params(colors="#374151", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#6B7280")
        ax.grid(False)

    for ax in axes:
        ax.set_xlabel("unit-domain x", color="#111827")
    axes[0].set_ylabel("unit-domain y", color="#111827")

    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markerfacecolor="#00D9FF",
            markeredgecolor="#071015",
            markersize=7,
            label="Seeding point",
        ),
        Line2D(
            [],
            [],
            color="#FFD400",
            linewidth=3,
            label="Vatistas vortex label",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
        labelcolor="#111827",
    )
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))

    output_stem = output.with_suffix("")
    png_output = output_stem.with_suffix(".png")
    pdf_output = output_stem.with_suffix(".pdf")
    mesh_pdf_output = output_stem.parent / (
        f"{output_stem.name}_feature_adaptive_mesh_seeding.pdf"
    )
    mesh_svg_output = mesh_pdf_output.with_suffix(".svg")
    lic_label_svg_output = output_stem.parent / (
        f"{output_stem.name}_lic_label.svg"
    )
    blurred_flow_svg_output = output_stem.parent / (
        f"{output_stem.name}_blurred_flow.svg"
    )
    mesh_points_svg_output = output_stem.parent / (
        f"{output_stem.name}_feature_adaptive_mesh_points.svg"
    )
    flow_grid_points_svg_output = output_stem.parent / (
        f"{output_stem.name}_original_flow_grid_points.svg"
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_output, dpi=dpi, facecolor=figure.get_facecolor())
    figure.savefig(pdf_output, facecolor=figure.get_facecolor())

    # Export the mesh-adaptation panel separately. The four-panel comparison
    # uses shared y axes, so restore the y tick labels and label for the crop.
    mesh_ax = axes[SAMPLING_METHODS.index(FEATURE_ADAPTIVE_MESH_SEEDING)]
    for other_ax in axes:
        if other_ax is not mesh_ax:
            other_ax.set_visible(False)
    mesh_ax.tick_params(labelleft=True)
    mesh_ax.set_ylabel("unit-domain y", color="#111827")
    mesh_ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
        labelcolor="#111827",
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    mesh_bbox = mesh_ax.get_tightbbox(renderer).transformed(
        figure.dpi_scale_trans.inverted()
    )
    figure.savefig(
        mesh_pdf_output,
        facecolor=figure.get_facecolor(),
        bbox_inches=mesh_bbox,
        pad_inches=0.08,
    )

    # The standalone SVG is an annotation-free visual asset. Keep the LIC,
    # mesh, samples, vortex boundary, and axes frame only.
    mesh_ax.set_title("")
    mesh_ax.set_xlabel("")
    mesh_ax.set_ylabel("")
    mesh_ax.set_xticks([])
    mesh_ax.set_yticks([])
    mesh_legend = mesh_ax.get_legend()
    if mesh_legend is not None:
        mesh_legend.remove()
    for figure_legend in list(figure.legends):
        figure_legend.remove()
    figure.canvas.draw()
    svg_bbox = mesh_ax.get_tightbbox(figure.canvas.get_renderer()).transformed(
        figure.dpi_scale_trans.inverted()
    )
    figure.savefig(
        mesh_svg_output,
        facecolor=figure.get_facecolor(),
        bbox_inches=svg_bbox,
        pad_inches=0.0,
    )

    plt.close(figure)

    # Export a clean LIC/label asset without samples, mesh, axes, or text.
    lic_figure = plt.figure(figsize=(6.0, 6.0))
    lic_figure.patch.set_facecolor("#FFFFFF")
    lic_ax = lic_figure.add_axes((0.0, 0.0, 1.0, 1.0))
    lic_ax.imshow(
        lic_image,
        extent=(0.0, 1.0, 0.0, 1.0),
        origin="lower",
        interpolation="bilinear",
        zorder=0,
    )
    lic_ax.contour(
        label_axis,
        label_axis,
        label,
        levels=[0.0],
        colors=["#171717"],
        linewidths=[3.5],
        zorder=2,
    )
    lic_ax.contour(
        label_axis,
        label_axis,
        label,
        levels=[0.0],
        colors=["#FFD400"],
        linewidths=[2.1],
        zorder=2.1,
    )
    lic_ax.set_xlim(0.0, 1.0)
    lic_ax.set_ylim(0.0, 1.0)
    lic_ax.set_aspect("equal", adjustable="box")
    lic_ax.set_axis_off()
    lic_figure.savefig(
        lic_label_svg_output,
        facecolor=lic_figure.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.0,
    )
    plt.close(lic_figure)

    # Export a color-only vorticity-magnitude field. This asset deliberately
    # contains neither LIC texture nor an additional blur operation.
    blur_figure = plt.figure(figsize=(6.0, 6.0))
    blur_ax = blur_figure.add_axes((0.0, 0.0, 1.0, 1.0))
    blur_ax.imshow(
        color_flow_image,
        extent=(0.0, 1.0, 0.0, 1.0),
        origin="lower",
        interpolation="bilinear",
        zorder=0,
    )
    blur_ax.set_xlim(0.0, 1.0)
    blur_ax.set_ylim(0.0, 1.0)
    blur_ax.set_aspect("equal", adjustable="box")
    blur_ax.set_axis_off()
    blur_figure.savefig(
        blurred_flow_svg_output,
        bbox_inches=None,
        pad_inches=0.0,
    )
    plt.close(blur_figure)

    # Export only the final feature-adaptive mesh seeding points on a
    # transparent canvas, without the LIC, label, mesh edges, or annotations.
    mesh_points = unit_coordinates(samples[FEATURE_ADAPTIVE_MESH_SEEDING])
    points_figure = plt.figure(figsize=(6.0, 6.0))
    points_figure.patch.set_alpha(0.0)
    points_ax = points_figure.add_axes((0.0, 0.0, 1.0, 1.0))
    points_ax.set_facecolor("none")
    points_ax.scatter(
        mesh_points[:, 0],
        mesh_points[:, 1],
        s=18,
        c="#00D9FF",
        edgecolors="#071015",
        linewidths=0.45,
        alpha=0.94,
    )
    points_ax.set_xlim(0.0, 1.0)
    points_ax.set_ylim(0.0, 1.0)
    points_ax.set_aspect("equal", adjustable="box")
    points_ax.set_axis_off()
    points_figure.savefig(
        mesh_points_svg_output,
        transparent=True,
        bbox_inches=None,
        pad_inches=0.0,
    )
    plt.close(points_figure)

    # Export a compact 16 x 16 representation of the regular flow-field grid.
    output_grid_size = 16
    grid_x = np.linspace(0.0, 1.0, output_grid_size)
    grid_y = np.linspace(0.0, 1.0, output_grid_size)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    grid_figure = plt.figure(figsize=(6.0, 6.0))
    grid_figure.patch.set_alpha(0.0)
    grid_ax = grid_figure.add_axes((0.0, 0.0, 1.0, 1.0))
    grid_ax.set_facecolor("none")
    grid_ax.scatter(
        grid_xx.ravel(),
        grid_yy.ravel(),
        s=18,
        c="#00D9FF",
        edgecolors="none",
        alpha=0.95,
    )
    grid_ax.set_xlim(-0.012, 1.012)
    grid_ax.set_ylim(-0.012, 1.012)
    grid_ax.set_aspect("equal", adjustable="box")
    grid_ax.set_axis_off()
    grid_figure.savefig(
        flow_grid_points_svg_output,
        transparent=True,
        bbox_inches=None,
        pad_inches=0.0,
    )
    plt.close(grid_figure)
    return (
        png_output,
        pdf_output,
        mesh_pdf_output,
        mesh_svg_output,
        lic_label_svg_output,
        blurred_flow_svg_output,
        mesh_points_svg_output,
        flow_grid_points_svg_output,
    )


def main() -> int:
    args = parse_args()
    if not 0.0 < args.high_ivd_percentile < 100.0:
        raise ValueError("--high-ivd-percentile must be in (0, 100).")
    if args.field_grid < 8:
        raise ValueError("--field-grid must be at least 8.")
    if args.lic_size < 64:
        raise ValueError("--lic-size must be at least 64.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")

    cfg = EasyConfig()
    cfg.load(str(args.config), recursive=False)
    # Share the illustration's region size between the sampler and its replay.
    cfg.setdefault("pathline", EasyConfig())["feature_high_ivd_percentile"] = (
        args.high_ivd_percentile
    )
    seed = int(cfg.get("seed", 0) if args.seed is None else args.seed)
    device = torch.device(args.device)
    bounds = build_bounds(cfg)

    velocity_eval, lic_image, color_flow_image, label = build_shared_lic(
        cfg,
        bounds,
        device,
        seed,
        args.field_grid,
        args.lic_size,
    )
    samples, adaptive_iteration_samples = sample_all_methods(
        cfg,
        velocity_eval,
        seed,
        args.count_k,
    )
    initialization = trace_adaptive_initialization(
        cfg, velocity_eval, seed, args.count_k, adaptive_iteration_samples[0]
    )
    (
        png_output,
        pdf_output,
        mesh_pdf_output,
        mesh_svg_output,
        lic_label_svg_output,
        blurred_flow_svg_output,
        mesh_points_svg_output,
        flow_grid_points_svg_output,
    ) = draw_figure(
        lic_image,
        color_flow_image,
        label,
        samples,
        args.output,
        args.dpi,
    )
    initialization_outputs = draw_adaptive_initialization(
        initialization, lic_image, label, args.output, args.dpi,
        adaptive_iteration_samples,
    )

    counts = ", ".join(
        f"{method}={len(samples[method])}" for method in SAMPLING_METHODS
    )
    print(f"Saved PNG: {png_output.resolve()}")
    print(f"Saved PDF: {pdf_output.resolve()}")
    print(f"Saved mesh-only PDF: {mesh_pdf_output.resolve()}")
    print(f"Saved mesh-only SVG: {mesh_svg_output.resolve()}")
    print(f"Saved LIC-and-label SVG: {lic_label_svg_output.resolve()}")
    print(f"Saved color-only flow SVG: {blurred_flow_svg_output.resolve()}")
    print(f"Saved mesh-points-only SVG: {mesh_points_svg_output.resolve()}")
    print(f"Saved original-grid-points SVG: {flow_grid_points_svg_output.resolve()}")
    for initialization_output in initialization_outputs:
        print(f"Saved adaptive-mesh diagram: {initialization_output.resolve()}")
    print(f"Seeding counts: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
