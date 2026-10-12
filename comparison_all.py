#!/usr/bin/env python3
"""Run all five eight-method vortex comparisons and draw one 8 x 5 figure.

The columns are the datasets handled by ``comparison_pipedcylinder.py``,
``comparison_jungtelziemniak2d.py``, ``comparison_dg.py``,
``comparison_rfc.py``, and ``comparison.py``.  The rows are, in order, IVD,
Q, Delta, lambda2, MVUNet, VortexUNet, VortexTransformer, and FGPA-VT.

This module deliberately reuses the readers, criteria, model adapters, and
thresholds from the five dataset-specific scripts.  It only provides the
common orchestration, result adapter, 40-panel visualization, and CSV
summary, so the combined plot cannot drift from the standalone comparisons.
The reference view retains every method's original detection mask. Numerical
results are also saved to NPZ; use ``--load-results`` to redraw the same
contours without rerunning inference.

Example
-------
MPLCONFIGDIR=/tmp/pyflowvis-mpl .venv/bin/python comparison_all.py
MPLCONFIGDIR=/tmp/pyflowvis-mpl .venv/bin/python comparison_all.py \
    --load-results outputs/comparison_all.npz
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle
import numpy as np
from scipy.ndimage import label as connected_components

import comparison as cylinder
import comparison_dg as double_gyre
import comparison_jungtelziemniak2d as jung
import comparison_pipedcylinder as piped_cylinder
import comparison_plot_style as plot_style
import comparison_rfc as rfc
import mvunet_comparison as mvunet
import vortexunet_comparison as vortexboundary


METHODS = (
    "IVD",
    "Q",
    "Delta",
    "lambda2",
    "MVUNet",
    "VortexUNet",
    "VortexTransformer",
    "FGPA-VT",
)

METHOD_LABELS = (
    r"\mathrm{IVD}",
    r"Q",
    r"\Delta",
    r"\lambda_2",
    r"\mathrm{MVUNet}",
    r"\mathrm{VortexUNet}",
    r"\mathrm{VortexTransformer}",
    r"\mathrm{FGPA\!\!-\!VT}",
)

FONT_FAMILY = "Times New Roman"

# Local to this figure: the supplied reference has more saturated green/yellow
# fills than the shared palette, with the same teal on masked/solid regions.
REFERENCE_FIELD_CMAP = LinearSegmentedColormap.from_list(
    "comparison_reference_teal_green_yellow",
    [(0.0, "#00958f"), (0.35, "#00c95c"), (1.0, "#ffff00")],
)
STREAMLINE_ALPHA = 0.9
STREAMLINE_WIDTH = 0.5
DETECTION_LINE_WIDTH = 0.85


def configure_latex(enabled: bool) -> None:
    """Use Times New Roman for regular and mathematical figure text."""

    plt.rcParams.update(
        {
            "text.usetex": enabled,
            # The TeX backend accepts generic family names only; newtx below
            # supplies its Times faces. The internal renderer uses the exact
            # installed Times New Roman font.
            "font.family": "serif" if enabled else FONT_FAMILY,
            "font.serif": ["Times"] if enabled else [FONT_FAMILY],
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT_FAMILY,
            "mathtext.it": f"{FONT_FAMILY}:italic",
            "mathtext.bf": f"{FONT_FAMILY}:bold",
            "axes.unicode_minus": False,
        }
    )
    if enabled:
        # LaTeX cannot load the system TTF directly; newtx provides matching
        # Times text and mathematics faces for TeX-rendered labels.
        plt.rcParams["text.latex.preamble"] = (
            r"\usepackage{amsmath}\usepackage{newtxtext,newtxmath}"
        )
    else:
        plt.rcParams["text.latex.preamble"] = ""


@dataclass(frozen=True)
class PanelResult:
    """One method's grid-aligned detection result."""

    method: str
    field: np.ndarray
    detected: np.ndarray
    rule: str
    regions: int
    centers: np.ndarray
    learned: bool = False


@dataclass(frozen=True)
class DatasetResult:
    """The eight results and plotting context for one input dataset."""

    key: str
    title: str
    path: Path
    snapshot_time: float
    x: np.ndarray
    y: np.ndarray
    snapshot_u: np.ndarray
    snapshot_v: np.ndarray
    panels: tuple[PanelResult, ...]
    obstacles: tuple[tuple[float, float, float], ...] = ()


def _centers(detections: Sequence[object]) -> np.ndarray:
    if not detections:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(
        [(float(item.x), float(item.y)) for item in detections],
        dtype=np.float64,
    )


def _panel(
    method: str,
    field: np.ndarray,
    detected: np.ndarray,
    rule: str,
    detections: Sequence[object] = (),
    *,
    learned: bool = False,
    regions: int | None = None,
) -> PanelResult:
    values = np.asarray(field, dtype=np.float64)
    mask = np.asarray(detected, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError(
            f"{method}: field shape {values.shape} != mask shape {mask.shape}."
        )
    if regions is None:
        regions = len(detections)
    return PanelResult(
        method=method,
        field=values,
        detected=mask,
        rule=rule,
        regions=int(regions),
        centers=_centers(detections),
        learned=learned,
    )


def _validate_dataset(result: DatasetResult) -> DatasetResult:
    methods = tuple(panel.method for panel in result.panels)
    if methods != METHODS:
        raise RuntimeError(
            f"{result.key}: expected methods {METHODS}, received {methods}."
        )
    expected = (len(result.y), len(result.x))
    if result.snapshot_u.shape != expected or result.snapshot_v.shape != expected:
        raise RuntimeError(f"{result.key}: snapshot does not match grid {expected}.")
    for panel in result.panels:
        if panel.field.shape != expected:
            raise RuntimeError(
                f"{result.key}/{panel.method}: field shape {panel.field.shape}, "
                f"expected {expected}."
            )
        if panel.detected.shape != expected or panel.detected.dtype != np.bool_:
            raise RuntimeError(
                f"{result.key}/{panel.method}: detection mask must be a "
                f"boolean array shaped {expected}."
            )
    return result


def _resolve_script(path: Path) -> Path:
    """Resolve path spelling variants used by older standalone scripts."""

    if path.is_file():
        return path
    # Some historical parsers contain more than one typo at once, so apply
    # every correction cumulatively instead of checking only one replacement.
    spellings = {str(path)}
    replacements = (
        ("trainVortexTransformer", "trainedVortexTransformer"),
        ("Backup_", "BackUp_"),
        ("BackUp_", "Backup_"),
        ("chekcpoint", "checkpoint"),
    )
    for old, new in replacements:
        spellings.update(text.replace(old, new) for text in tuple(spellings))
    for spelling in spellings:
        candidate = Path(spelling)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(path)


def _ensure_mvunet_arguments(args: argparse.Namespace) -> None:
    """Supply MVUNet defaults for source parsers that predate its adapter."""

    defaults = {
        "mvunet_checkpoint": mvunet.DEFAULT_CHECKPOINT,
        "mvunet_torchscript": mvunet.DEFAULT_TORCHSCRIPT,
        "mvunet_threshold": 0.2,
        "mvunet_smooth_sigma": 1.0,
        "mvunet_min_component_pixels": 8,
        "mvunet_min_value": mvunet.DEFAULT_MIN_VALUE,
        "mvunet_max_value": mvunet.DEFAULT_MAX_VALUE,
    }
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    mvunet.validate_arguments(args)


def _ensure_vortexboundary_arguments(args: argparse.Namespace) -> None:
    """Supply VortexUNet defaults for source parsers without its adapter."""

    defaults = {
        "vortexunet_checkpoint": vortexboundary.DEFAULT_CHECKPOINT,
        "vortexunet_torchscript": vortexboundary.DEFAULT_TORCHSCRIPT,
        "vortexunet_threshold": 0.5,
        "vortexunet_smooth_sigma": 1.0,
        "vortexunet_min_component_pixels": 8,
        "vortexunet_min_value": vortexboundary.DEFAULT_MIN_VALUE,
        "vortexunet_max_value": vortexboundary.DEFAULT_MAX_VALUE,
    }
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    vortexboundary.validate_arguments(args)


def _base_comparison(
    *,
    key: str,
    title: str,
    args: argparse.Namespace,
    v1_runner: Callable[..., tuple[object, int, int, str]],
) -> DatasetResult:
    """Run the shared cylinder/RFC implementation without a standalone plot."""

    cylinder._validate_args(args)
    reference_indices = (
        None
        if args.v0_ivd_reference_mean is not None
        else (
            args.v0_ivd_reference_start_index,
            args.v0_ivd_reference_end_index,
        )
    )
    duration = max(args.v0_duration, args.v1_duration)
    flow, start, snapshot_u, snapshot_v, constants = (
        cylinder.v0_common.load_flow_window(
            args.input,
            args.time_index,
            duration,
            legacy_reference_indices=reference_indices,
            legacy_reference_mean=args.v0_ivd_reference_mean,
        )
    )
    obstacle_center = (args.obstacle_x, args.obstacle_y)
    obstacle_radius = (
        float(constants.get("radius", 0.0))
        if args.obstacle_radius is None
        else float(args.obstacle_radius)
    )

    print(f"[{title}] loaded {args.input}, t={start:g}; running models...")
    vt = cylinder.run_v0(
        flow, start, args, obstacle_center, obstacle_radius, "cpu"
    )
    fgma, _, _, _ = v1_runner(
        flow, start, args, obstacle_center, obstacle_radius, "cpu"
    )
    ivd, ivd_threshold, _, ivd_detections = (
        cylinder.v0_common.detect_ivd_regions(
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
    criteria = cylinder.build_eulerian_criteria(
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
    mv = mvunet.run(
        snapshot_u,
        snapshot_v,
        flow.x,
        flow.y,
        args,
        valid_mask=mvunet_valid,
        border_cells=args.classic_border_cells,
        device="cpu",
    )
    vb = vortexboundary.run(
        snapshot_u,
        snapshot_v,
        flow.x,
        flow.y,
        args,
        valid_mask=mvunet_valid,
        border_cells=args.classic_border_cells,
        device="cpu",
    )

    panels: list[PanelResult] = [
        _panel(
            "IVD",
            ivd,
            np.isfinite(ivd) & (ivd >= ivd_threshold),
            f"IVD >= {ivd_threshold:.4g}",
            ivd_detections,
        )
    ]
    for method, criterion in zip(METHODS[1:4], criteria):
        score = -criterion.values if criterion.lower_is_vortex else criterion.values
        score_threshold = (
            -criterion.threshold
            if criterion.lower_is_vortex
            else criterion.threshold
        )
        panels.append(
            _panel(
                method,
                score,
                np.isfinite(score) & (score > score_threshold),
                criterion.decision_rule,
                criterion.detections,
            )
        )
    panels.append(
        _panel(
            "MVUNet",
            mv.probability,
            mv.mask,
            f"p >= {args.mvunet_threshold:g}",
            mv.detections,
            learned=True,
        )
    )
    panels.append(
        _panel(
            "VortexUNet",
            vb.probability,
            vb.mask,
            f"p >= {args.vortexunet_threshold:g}",
            vb.detections,
            learned=True,
        )
    )
    for method, model_result in zip(METHODS[6:], (vt, fgma)):
        probability = np.asarray(model_result.probability, dtype=np.float64)
        panels.append(
            _panel(
                method,
                probability,
                np.isfinite(probability)
                & (probability >= model_result.threshold),
                f"p >= {model_result.threshold:g}",
                model_result.detections,
                learned=True,
            )
        )

    obstacles = (
        ((obstacle_center[0], obstacle_center[1], obstacle_radius),)
        if obstacle_radius > 0.0
        else ()
    )
    return _validate_dataset(
        DatasetResult(
            key=key,
            title=title,
            path=args.input,
            snapshot_time=float(start),
            x=np.asarray(flow.x),
            y=np.asarray(flow.y),
            snapshot_u=np.asarray(snapshot_u),
            snapshot_v=np.asarray(snapshot_v),
            panels=tuple(panels),
            obstacles=obstacles,
        )
    )


def run_cylinder(input_path: Path, time_index: int) -> DatasetResult:
    args = cylinder.build_parser().parse_args([])
    args.input = input_path
    args.time_index = time_index
    return _base_comparison(
        key="cylinder",
        title="Cylinder Flow with von Karman Vortex Street",
        args=args,
        v1_runner=cylinder.run_v1,
    )


def run_rfc(input_path: Path, time_index: int) -> DatasetResult:
    args = rfc.build_parser().parse_args([])
    args.input = input_path
    args.time_index = time_index
    return _base_comparison(
        key="rfc",
        title="RFC64",
        args=args,
        v1_runner=rfc.run_rfc_v1,
    )


def run_piped_cylinder(input_path: Path, time_index: int) -> DatasetResult:
    args = piped_cylinder.build_parser().parse_args([])
    args.input = input_path
    args.time_index = time_index
    args.vt_checkpoint = _resolve_script(args.vt_checkpoint)
    args.vt_torchscript = _resolve_script(args.vt_torchscript)
    args.fgma_checkpoint = _resolve_script(args.fgma_checkpoint)
    args.fgma_torchscript = _resolve_script(args.fgma_torchscript)
    piped_cylinder.validate_args(args)

    flow, start = piped_cylinder.load_flow(args.input, args.time_index, args.duration)
    patches = piped_cylinder.make_patches(
        flow,
        args.patch_points,
        args.patch_stride,
        args.minimum_fluid_fraction,
    )
    print(f"[Piped cylinder] loaded {args.input}, t={start:g}; running models...")
    results = piped_cylinder.classical_results(flow, args)
    vt, _, _, _, _ = piped_cylinder.legacy_vt_result(
        flow,
        start,
        piped_cylinder.existing(args.vt_checkpoint),
        args.vt_torchscript,
        args,
    )
    fgma, _, _, _, _ = piped_cylinder.neural_result(
        "FGPA-VT",
        piped_cylinder.FEATURE_ADAPTIVE_MESH_SEEDING,
        flow,
        start,
        patches,
        piped_cylinder.existing(args.fgma_checkpoint),
        args.fgma_torchscript,
        args,
        args.random_seed + 100000,
        args.fgma_threshold,
        args.fgma_smooth_sigma,
        args.fgma_min_component_pixels,
    )
    mv = mvunet.run(
        flow.u[0],
        flow.v[0],
        flow.x,
        flow.y,
        args,
        valid_mask=flow.analysis_fluid,
        device="cpu",
    )
    vb = vortexboundary.run(
        flow.u[0],
        flow.v[0],
        flow.x,
        flow.y,
        args,
        valid_mask=flow.analysis_fluid,
        device="cpu",
    )
    results.extend(
        (
            piped_cylinder.Result(
                "MVUNet",
                mv.probability,
                args.mvunet_threshold,
                mv.mask,
                mv.detections,
                mvunet.TRAINING_SEEDING,
            ),
            piped_cylinder.Result(
                "VortexUNet",
                vb.probability,
                args.vortexunet_threshold,
                vb.mask,
                vb.detections,
                vortexboundary.TRAINING_SEEDING,
            ),
            vt,
            fgma,
        )
    )

    panels = tuple(
        _panel(
            method,
            result.score,
            result.mask,
            f"score >= {result.threshold:.4g}",
            result.detections,
            learned=method in METHODS[4:],
        )
        for method, result in zip(METHODS, results)
    )
    obstacles = tuple(
        (float(cx), float(cy), float(flow.radius)) for cx, cy in flow.cylinders
    )
    return _validate_dataset(
        DatasetResult(
            key="piped_cylinder",
            title="Cynlinder Flow Around Corners",
            path=args.input,
            snapshot_time=float(start),
            x=np.asarray(flow.x),
            y=np.asarray(flow.y),
            snapshot_u=np.asarray(flow.u[0]),
            snapshot_v=np.asarray(flow.v[0]),
            panels=panels,
            obstacles=obstacles,
        )
    )


def run_jung(
    input_path: Path,
    time_index: int,
    device_name: str,
) -> DatasetResult:
    args = jung.parser().parse_args([])
    args.input = input_path
    args.time_index = time_index
    args.device = device_name
    args.vt_checkpoint = _resolve_script(args.vt_checkpoint)
    args.vt_script = _resolve_script(args.vt_script)
    args.fgma_checkpoint = _resolve_script(args.fgma_checkpoint)
    args.fgma_script = _resolve_script(args.fgma_script)
    _ensure_mvunet_arguments(args)
    _ensure_vortexboundary_arguments(args)
    device = jung.choose_device(args.device)
    window = jung.load_time_window(args.input, args.time_index, args.duration)
    tiles = jung.make_tiles(window, args.patch_size, args.patch_overlap)
    print(
        f"[Jung-Tel-Ziemniak] loaded {args.input}, "
        f"t={window.snapshot_time:g}; running models..."
    )
    results = jung.instantaneous_criteria(
        window,
        args.smoothing_sigma,
        args.ivd_percentile,
        args.classical_core_percentile,
    )
    vt, _, _ = jung.neural_result(
        "VortexTransformer",
        jung.GRID_CROSS_SEEDING,
        window,
        tiles,
        args.duration,
        jung.find_checkpoint(args.vt_checkpoint, (), "VortexTransformer checkpoint"),
        args.vt_script,
        device,
        args.vt_threshold,
        args.seed,
    )
    fgma, _, _ = jung.neural_result(
        "FGPA-VT",
        jung.FEATURE_ADAPTIVE_MESH_SEEDING,
        window,
        tiles,
        args.duration,
        jung.find_checkpoint(args.fgma_checkpoint, (), "FGPA-VT checkpoint"),
        args.fgma_script,
        device,
        args.fgma_threshold,
        args.seed + 1000,
    )
    mv = mvunet.run(
        window.snapshot[..., 0],
        window.snapshot[..., 1],
        window.x,
        window.y,
        args,
        device="cpu",
    )
    vb = vortexboundary.run(
        window.snapshot[..., 0],
        window.snapshot[..., 1],
        window.x,
        window.y,
        args,
        device="cpu",
    )
    results.extend(
        (
            jung.Result(
                "MVUNet",
                mv.probability,
                mv.mask,
                f"p >= {args.mvunet_threshold:g}",
            ),
            jung.Result(
                "VortexUNet",
                vb.probability,
                vb.mask,
                f"p >= {args.vortexunet_threshold:g}",
            ),
            vt,
            fgma,
        )
    )

    panels: list[PanelResult] = []
    for method, result in zip(METHODS, results):
        # The common plot uses a vortex-strength orientation: larger is more
        # vortex-like, hence -lambda2 in the fourth row.
        field = -result.values if method == "lambda2" else result.values
        _, region_count = connected_components(result.detected)
        panels.append(
            _panel(
                method,
                field,
                result.detected,
                result.rule,
                learned=method in METHODS[4:],
                regions=int(region_count),
            )
        )
    return _validate_dataset(
        DatasetResult(
            key="jungtelziemniak",
            title="Cynlinder Flow",
            path=args.input,
            snapshot_time=float(window.snapshot_time),
            x=np.asarray(window.x),
            y=np.asarray(window.y),
            snapshot_u=np.asarray(window.snapshot[..., 0]),
            snapshot_v=np.asarray(window.snapshot[..., 1]),
            panels=tuple(panels),
        )
    )


def run_double_gyre(input_path: Path, time_index: int) -> DatasetResult:
    args = double_gyre.parser().parse_args([])
    args.input = input_path
    args.time_index = time_index
    args.vt_checkpoint = _resolve_script(args.vt_checkpoint)
    args.vt_torchscript = _resolve_script(args.vt_torchscript)
    args.fgma_checkpoint = _resolve_script(args.fgma_checkpoint)
    args.fgma_torchscript = _resolve_script(args.fgma_torchscript)
    double_gyre.validate(args)
    flow, start = double_gyre.load_flow(args.input, args.time_index, args.duration)
    patches = double_gyre.make_patches(flow, args.patch_points, args.patch_stride)
    print(f"[Double gyre] loaded {args.input}, t={start:g}; running models...")
    results = double_gyre.classical(flow, args)
    vt, _, _, _ = double_gyre.neural(
        "VortexTransformer",
        double_gyre.GRID_CROSS_SEEDING,
        flow,
        start,
        patches,
        double_gyre.existing(args.vt_checkpoint),
        args.vt_torchscript,
        args,
        args.random_seed,
    )
    fgma, _, _, _ = double_gyre.neural(
        "FGPA-VT",
        double_gyre.FEATURE_ADAPTIVE_MESH_SEEDING,
        flow,
        start,
        patches,
        double_gyre.existing(args.fgma_checkpoint),
        args.fgma_torchscript,
        args,
        args.random_seed + 100000,
    )
    mv = mvunet.run(
        flow.u[0],
        flow.v[0],
        flow.x,
        flow.y,
        args,
        border_cells=args.border_cells,
        device="cpu",
    )
    vb = vortexboundary.run(
        flow.u[0],
        flow.v[0],
        flow.x,
        flow.y,
        args,
        border_cells=args.border_cells,
        device="cpu",
    )
    results.extend(
        (
            double_gyre.Result(
                "MVUNet",
                mv.probability,
                mv.probability,
                args.mvunet_threshold,
                mv.mask,
                mv.detections,
                mvunet.TRAINING_SEEDING,
            ),
            double_gyre.Result(
                "VortexUNet",
                vb.probability,
                vb.probability,
                args.vortexunet_threshold,
                vb.mask,
                vb.detections,
                vortexboundary.TRAINING_SEEDING,
            ),
            vt,
            fgma,
        )
    )
    panels = tuple(
        _panel(
            method,
            result.score,
            result.mask,
            f"score >= {result.threshold:.4g}",
            result.detections,
            learned=method in METHODS[4:],
        )
        for method, result in zip(METHODS, results)
    )
    return _validate_dataset(
        DatasetResult(
            key="double_gyre",
            title="Double Gyre",
            path=args.input,
            snapshot_time=float(start),
            x=np.asarray(flow.x),
            y=np.asarray(flow.y),
            snapshot_u=np.asarray(flow.u[0]),
            snapshot_v=np.asarray(flow.v[0]),
            panels=panels,
        )
    )


def _field_norm(panel: PanelResult) -> Normalize:
    display = np.maximum(panel.field, 0.0)
    finite = display[np.isfinite(display)]
    if panel.learned:
        return Normalize(vmin=0.0, vmax=1.0)
    if not finite.size:
        return Normalize(vmin=0.0, vmax=1.0)
    percentile = 99.5 if panel.method == "IVD" else 99.0
    upper = float(np.percentile(finite, percentile))
    return Normalize(vmin=0.0, vmax=max(upper, np.finfo(float).eps))


def save_results(path: Path, datasets: Sequence[DatasetResult]) -> None:
    """Store the actual fields and masks, not fitted curves from a screenshot."""

    arrays = {}
    metadata = {"version": 1, "datasets": []}
    for column, dataset in enumerate(datasets):
        prefix = f"dataset_{column}"
        for name in ("x", "y", "snapshot_u", "snapshot_v"):
            arrays[f"{prefix}_{name}"] = getattr(dataset, name)
        panels = []
        for row, panel in enumerate(dataset.panels):
            for name in ("field", "detected", "centers"):
                arrays[f"{prefix}_panel_{row}_{name}"] = getattr(panel, name)
            panels.append({
                "method": panel.method, "rule": panel.rule,
                "regions": panel.regions, "learned": panel.learned,
            })
        metadata["datasets"].append({
            "key": dataset.key, "title": dataset.title, "path": str(dataset.path),
            "snapshot_time": dataset.snapshot_time,
            "obstacles": dataset.obstacles, "panels": panels,
        })
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def load_results(path: Path) -> tuple[DatasetResult, ...]:
    """Restore grids, raw scores, and masks for repeatable contour rendering."""

    datasets = []
    with np.load(path.expanduser(), allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["metadata"].item()))
        if metadata.get("version") != 1 or len(metadata.get("datasets", [])) != 5:
            raise ValueError("Expected a version-1, five-dataset comparison archive.")
        for column, data in enumerate(metadata["datasets"]):
            prefix = f"dataset_{column}"
            panels = tuple(
                PanelResult(
                    **panel,
                    **{name: arrays[f"{prefix}_panel_{row}_{name}"]
                       for name in ("field", "detected", "centers")},
                )
                for row, panel in enumerate(data["panels"])
            )
            datasets.append(_validate_dataset(DatasetResult(
                key=data["key"], title=data["title"], path=Path(data["path"]),
                snapshot_time=data["snapshot_time"], panels=panels,
                obstacles=tuple(tuple(item) for item in data["obstacles"]),
                **{name: arrays[f"{prefix}_{name}"]
                   for name in ("x", "y", "snapshot_u", "snapshot_v")},
            )))
    return tuple(datasets)


def plot_all(
    datasets: Sequence[DatasetResult],
    output: Path,
    dpi: int,
    *,
    panel_height: float,
    streamlines: bool,
    show: bool,
) -> tuple[Path, Path]:
    """Draw an aligned 8-by-5 comparison with equal panel heights.

    Every GridSpec column is proportional to its physical-domain aspect
    ratio.  Combined with an equal data aspect, this makes all 40 axes fill
    their cells at one common height instead of leaving letterbox whitespace.
    """

    if len(datasets) != 5:
        raise ValueError("Exactly five datasets are required for the 8 x 5 plot.")
    aspect_ratios = np.asarray(
        [
            abs(float(dataset.x[-1] - dataset.x[0]))
            / abs(float(dataset.y[-1] - dataset.y[0]))
            for dataset in datasets
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(aspect_ratios)) or np.any(aspect_ratios <= 0.0):
        raise ValueError("Every dataset must have a finite, nonzero domain extent.")

    # Derive the canvas from the requested common axes height.  GridSpec's
    # wspace/hspace units are fractions of the average axes width/height.
    column_gap = 0.025
    row_gap = 0.055
    bottom, top = 0.053, 0.962
    axes_width = panel_height * float(np.sum(aspect_ratios))
    inner_width = axes_width * (1.0 + column_gap * 4.0 / 5.0)
    row_count = len(METHODS)
    inner_height = panel_height * (
        float(row_count) + row_gap * float(row_count - 1)
    )
    left_margin_inches = 2.9
    right_margin_inches = 0.15
    figure_width = inner_width + left_margin_inches + right_margin_inches
    figure_height = inner_height / (top - bottom)
    left = left_margin_inches / figure_width
    right = 1.0 - right_margin_inches / figure_width
    figure, axes = plt.subplots(
        row_count,
        5,
        figsize=(figure_width, figure_height),
        gridspec_kw={"width_ratios": aspect_ratios},
        squeeze=False,
    )
    figure.subplots_adjust(
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        wspace=column_gap,
        hspace=row_gap,
    )
    field_cmap = REFERENCE_FIELD_CMAP
    background_color = plot_style.BACKGROUND_COLOR
    threshold_color = plot_style.THRESHOLD_COLOR

    for column, dataset in enumerate(datasets):
        axes[0, column].set_title(
            f"{dataset.title}",
            fontsize=30,
            weight="bold",
            pad=7,
        )
        skip = max(1, min(len(dataset.x), len(dataset.y)) // 72)
        stream_x = np.linspace(dataset.x[0], dataset.x[-1], len(dataset.x))[::skip]
        stream_y = np.linspace(dataset.y[0], dataset.y[-1], len(dataset.y))[::skip]

        for row, panel in enumerate(dataset.panels):
            axis = axes[row, column]
            axis.set_facecolor(background_color)
            display_field = np.maximum(panel.field, 0.0)
            axis.pcolormesh(
                dataset.x,
                dataset.y,
                np.ma.masked_invalid(display_field),
                shading="auto",
                cmap=field_cmap,
                norm=_field_norm(panel),
                rasterized=True,
                zorder=1,
            )
            if streamlines:
                axis.streamplot(
                    stream_x,
                    stream_y,
                    dataset.snapshot_u[::skip, ::skip],
                    dataset.snapshot_v[::skip, ::skip],
                    color=plot_style.streamline_color(STREAMLINE_ALPHA),
                    density=0.55,
                    linewidth=STREAMLINE_WIDTH,
                    arrowsize=0.38,
                    minlength=0.08,
                    zorder=3,
                )
            if np.any(panel.detected) and np.any(~panel.detected):
                axis.contour(
                    dataset.x,
                    dataset.y,
                    panel.detected.astype(float),
                    levels=[0.5],
                    colors=[threshold_color],
                    linewidths=DETECTION_LINE_WIDTH,
                    algorithm="mpl2014",
                    zorder=4,
                )
            for obstacle_x, obstacle_y, obstacle_radius in dataset.obstacles:
                axis.add_patch(
                    Circle(
                        (obstacle_x, obstacle_y),
                        obstacle_radius,
                        facecolor=plot_style.OBSTACLE_FACE_COLOR,
                        edgecolor=plot_style.OBSTACLE_EDGE_COLOR,
                        linewidth=0.65,
                        zorder=6,
                    )
                )
            axis.set_xlim(float(dataset.x[0]), float(dataset.x[-1]))
            axis.set_ylim(float(dataset.y[0]), float(dataset.y[-1]))
            axis.set_aspect("equal", adjustable="box")
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(
                    f"${METHOD_LABELS[row]}$",
                    fontsize=25,
                    weight="bold",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=8,
                )

    output = output.expanduser()
    if output.suffix.lower() == ".pdf":
        raster_output = output.with_suffix(".png")
        pdf_output = output
    elif output.suffix:
        raster_output = output
        pdf_output = output.with_suffix(".pdf")
    else:
        raster_output = output.with_suffix(".png")
        pdf_output = output.with_suffix(".pdf")
    raster_output.parent.mkdir(parents=True, exist_ok=True)
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(raster_output, dpi=dpi)
    # Use the same curve widths in PNG and PDF so neither export loses lines.
    figure.savefig(pdf_output, dpi=dpi, format="pdf")
    if show:
        plt.show()
    plt.close(figure)
    return raster_output, pdf_output


def save_summary(path: Path, datasets: Sequence[DatasetResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "dataset",
                "input",
                "snapshot_time",
                "method",
                "rule",
                "regions",
                "detected_pixels",
                "finite_pixels",
                "detected_fraction",
            )
        )
        for dataset in datasets:
            for panel in dataset.panels:
                finite = np.isfinite(panel.field)
                detected_pixels = int(np.count_nonzero(panel.detected & finite))
                finite_pixels = int(np.count_nonzero(finite))
                writer.writerow(
                    (
                        dataset.key,
                        dataset.path,
                        f"{dataset.snapshot_time:.9g}",
                        panel.method,
                        panel.rule,
                        panel.regions,
                        detected_pixels,
                        finite_pixels,
                        f"{detected_pixels / max(finite_pixels, 1):.9g}",
                    )
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cylinder-input", type=Path, default=Path("dataset/cylinder2d.nc"))
    parser.add_argument(
        "--piped-cylinder-input",
        type=Path,
        default=Path("dataset/pipedcylinder2d.nc"),
    )
    parser.add_argument(
        "--jung-input",
        type=Path,
        default=Path("dataset/jungtelziemniak2d.nc"),
    )
    parser.add_argument(
        "--double-gyre-input",
        type=Path,
        default=Path("dataset/doublegyre2d.nc"),
    )
    parser.add_argument("--rfc-input", type=Path, default=Path("dataset/fourcenters2d.nc"))
    parser.add_argument("--cylinder-time-index", type=int, default=1000)
    parser.add_argument("--piped-cylinder-time-index", type=int, default=1000)
    parser.add_argument("--jung-time-index", type=int, default=0)
    parser.add_argument("--double-gyre-time-index", type=int, default=0)
    parser.add_argument("--rfc-time-index", type=int, default=256)
    parser.add_argument(
        "--jung-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device for the Jung-Tel-Ziemniak model panels.",
    )
    parser.add_argument(
        "--load-results", type=Path,
        help="Redraw a saved comparison NPZ without rerunning model inference.",
    )
    parser.add_argument(
        "--results-output", type=Path,
        help="Save numerical fields and masks here (default: output stem + .npz).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/comparison_all.png"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/comparison_all.csv"),
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--panel-height",
        type=float,
        default=2.0,
        help="Common physical height in inches for every one of the 40 panels.",
    )
    parser.add_argument("--no-streamlines", action="store_true")
    parser.add_argument(
        "--no-latex",
        dest="use_latex",
        action="store_false",
        help="Use Matplotlib's internal text renderer instead of system LaTeX.",
    )
    parser.set_defaults(use_latex=True)
    parser.add_argument("--show", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    inputs = (
        args.cylinder_input,
        args.piped_cylinder_input,
        args.jung_input,
        args.double_gyre_input,
        args.rfc_input,
    )
    required = (args.load_results.expanduser(),) if args.load_results else inputs
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input dataset(s): " + ", ".join(missing))
    if min(
        args.cylinder_time_index,
        args.piped_cylinder_time_index,
        args.jung_time_index,
        args.double_gyre_time_index,
        args.rfc_time_index,
    ) < 0:
        raise ValueError("All time indices must be non-negative.")
    if args.dpi < 50:
        raise ValueError("--dpi must be at least 50.")
    if not np.isfinite(args.panel_height) or args.panel_height <= 0.0:
        raise ValueError("--panel-height must be a finite positive number.")


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    configure_latex(args.use_latex)
    # Column order follows the five source files named in the request.
    datasets = load_results(args.load_results) if args.load_results else (
        run_piped_cylinder(
            args.piped_cylinder_input, args.piped_cylinder_time_index
        ),
        run_jung(args.jung_input, args.jung_time_index, args.jung_device),
        run_double_gyre(args.double_gyre_input, args.double_gyre_time_index),
        run_rfc(args.rfc_input, args.rfc_time_index),
        run_cylinder(args.cylinder_input, args.cylinder_time_index),
    )
    results_output = args.results_output or args.output.with_suffix(".npz")
    save_results(results_output, datasets)
    save_summary(args.summary_output, datasets)
    raster_output, pdf_output = plot_all(
        datasets,
        args.output,
        args.dpi,
        panel_height=args.panel_height,
        streamlines=not args.no_streamlines,
        show=args.show,
    )
    print(f"40-panel comparison: {raster_output}")
    print(f"40-panel comparison PDF: {pdf_output}")
    print(f"Combined summary: {args.summary_output}")
    print(f"Replayable fields and detection masks: {results_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
