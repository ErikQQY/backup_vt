"""Generate Vatistas/Tobias data in the legacy VortexViz training format.

This module reuses the analytical steady-flow and moving-observer machinery in
``VatistasFlowDatasetGenerator.py`` and adds the representation required by:

* ``DeepUtils/dataset/vortexVizDataset.py``;
* ``DeepUtils/models/classification/VortexViz.py``; and
* ``config/classification/vortex_viz.yaml``.

For every pathline the exporter writes:

1. a seed-centred ``65 x 65`` binary trajectory image;
2. a 1024-value cumulative-absolute-curl vector; and
3. the binary label of the pathline seed.

The three arrays are stored as Cereal-compatible vectors using the exact file
names expected by ``VortexVizDataset``.  All observer variants of one steady
field are assigned to the same split to prevent train/validation/test leakage.

Direct, physically faithful generation (default verification subset):

    python VatistasFlowDatasetGeneratorVortexViz.py

Full paper-scale generation is extremely expensive and produces millions of
per-pathline examples:

    python VatistasFlowDatasetGeneratorVortexViz.py --full

Existing VortexTransformer pathline binaries can also be converted.  Because
those binaries contain IVD rather than signed curl, this mode exports cumulative
IVD as an explicitly-labelled proxy and is not a faithful reproduction of the
VortexViz curl feature:

    python VatistasFlowDatasetGeneratorVortexViz.py \
        --convert-existing outputs/trainable_segnet
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import VatistasFlowDatasetGenerator as vatistas


LOGGER = logging.getLogger("vatistas_vortexviz")
DEFAULT_OUTPUT_DIR = "CppProjects/data/VortexViz"
DEFAULT_IMAGE_SIZE = 65
DEFAULT_MAX_PATHLINE_LENGTH = 1024
DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)
SPLIT_NAMES = ("train", "validation", "test")


def _read_cereal_vector(path: Path, dtype: np.dtype) -> np.ndarray:
    """Read a Cereal ``std::vector<T>`` and validate its uint64 header."""
    dtype = np.dtype(dtype)
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype="<u8", count=1)
        if header.size != 1:
            raise ValueError(f"Missing Cereal vector header in {path}.")
        count = int(header[0])
        values = np.fromfile(handle, dtype=dtype)
    if values.size != count:
        raise ValueError(
            f"Invalid Cereal vector in {path}: header says {count} values, "
            f"but the payload contains {values.size}.")
    return values


def _write_cereal_vector(path: Path, values: np.ndarray, dtype: np.dtype) -> None:
    """Write one little-endian Cereal ``std::vector<T>`` for streaming output."""
    dtype = np.dtype(dtype)
    flat = np.ascontiguousarray(values, dtype=dtype).reshape(-1)
    if np.issubdtype(dtype, np.floating) and not np.all(np.isfinite(flat)):
        raise ValueError(f"Cannot write non-finite values to {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(np.asarray([flat.size], dtype="<u8").tobytes())
        handle.write(flat.tobytes(order="C"))


def _vortexviz_paths(folder: Path, sample_tag: str) -> tuple[Path, Path, Path]:
    """Return the exact three paths derived by ``VortexVizDataset``.

    The missing underscore before ``pathlineImages.bin`` is intentional: the
    existing loader replaces ``_CulmulateAbsCurl.bin`` with
    ``pathlineImages.bin`` verbatim.
    """
    information = folder / f"{sample_tag}_CulmulateAbsCurl.bin"
    images = folder / f"{sample_tag}pathlineImages.bin"
    labels = folder / f"{sample_tag}_segmentationLabel.bin"
    return information, images, labels


def _write_vortexviz_triplet(
    folder: Path,
    sample_tag: str,
    cumulative_abs_curl: np.ndarray,
    binary_images: np.ndarray,
    labels: np.ndarray,
) -> None:
    information_path, image_path, label_path = _vortexviz_paths(
        folder, sample_tag)
    cumulative_abs_curl = np.asarray(cumulative_abs_curl, dtype=np.float32)
    binary_images = np.asarray(binary_images, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    if cumulative_abs_curl.ndim != 2:
        raise ValueError(
            "Cumulative-absolute-curl data must be [pathline, step], got "
            f"{cumulative_abs_curl.shape}.")
    if binary_images.ndim != 3:
        raise ValueError(
            f"Binary images must be [pathline, y, x], got {binary_images.shape}.")
    line_count = cumulative_abs_curl.shape[0]
    if binary_images.shape[0] != line_count or labels.size != line_count:
        raise ValueError(
            "VortexViz triplet has inconsistent pathline counts: "
            f"information={line_count}, images={binary_images.shape[0]}, "
            f"labels={labels.size}.")
    _write_cereal_vector(information_path, cumulative_abs_curl, "<f4")
    _write_cereal_vector(image_path, binary_images, np.uint8)
    _write_cereal_vector(label_path, labels, np.uint8)


def _draw_segment(image: np.ndarray, start: np.ndarray, end: np.ndarray) -> None:
    """Rasterize a line segment with an integer Bresenham traversal."""
    x0, y0 = int(start[0]), int(start[1])
    x1, y1 = int(end[0]), int(end[1])
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        if 0 <= y0 < image.shape[0] and 0 <= x0 < image.shape[1]:
            image[y0, x0] = 1
        if x0 == x1 and y0 == y1:
            return
        twice_error = 2 * error
        if twice_error >= dy:
            error += dy
            x0 += sx
        if twice_error <= dx:
            error += dx
            y0 += sy


def pathlines_to_seed_centered_images(
    positions: np.ndarray,
    valid_lengths: np.ndarray,
    image_size: int = DEFAULT_IMAGE_SIZE,
    domain_min: tuple[float, float] = (-1.0, -1.0),
    domain_max: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """Convert global pathline coordinates to VortexViz binary images.

    The seed is placed at the centre.  A trajectory that fits in the native
    window keeps the domain's native pixel scale.  A longer trajectory is
    isotropically shrunk just enough to fit, matching the representation
    described in the VortexViz paper.
    """
    positions = np.asarray(positions, dtype=np.float64)
    valid_lengths = np.asarray(valid_lengths, dtype=np.int64).reshape(-1)
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError(f"Expected positions [pathline, step, 2], got {positions.shape}.")
    if positions.shape[0] != valid_lengths.size:
        raise ValueError("valid_lengths must contain one entry per pathline.")
    if image_size < 3 or image_size % 2 == 0:
        raise ValueError("VortexViz image_size must be an odd integer >= 3.")

    domain_min_array = np.asarray(domain_min, dtype=np.float64)
    domain_max_array = np.asarray(domain_max, dtype=np.float64)
    domain_range = domain_max_array - domain_min_array
    if np.any(domain_range <= 0.0):
        raise ValueError("domain_max must be larger than domain_min on both axes.")

    centre = (image_size - 1) / 2.0
    native_pixels_per_unit = (image_size - 1) / domain_range
    images = np.zeros(
        (positions.shape[0], image_size, image_size), dtype=np.uint8)

    for line_index, raw_length in enumerate(valid_lengths):
        length = min(max(int(raw_length), 0), positions.shape[1])
        if length == 0:
            continue
        line = positions[line_index, :length]
        finite = np.all(np.isfinite(line), axis=1)
        line = line[finite]
        if line.size == 0:
            continue
        offsets = (line - line[0]) * native_pixels_per_unit
        extent = float(np.max(np.abs(offsets))) if offsets.size else 0.0
        shrink = max(1.0, extent / centre) if centre > 0.0 else 1.0
        pixels = np.rint(offsets / shrink + centre).astype(np.int64)
        # Image rows grow downward while the analytical y axis grows upward.
        pixels[:, 1] = (image_size - 1) - pixels[:, 1]
        pixels = np.clip(pixels, 0, image_size - 1)
        image = images[line_index]
        image[pixels[0, 1], pixels[0, 0]] = 1
        for start, end in zip(pixels[:-1], pixels[1:]):
            _draw_segment(image, start, end)
    return images


def _finite_difference_vorticity(
    velocity_eval: Any,
    points: np.ndarray,
    time: float,
    delta: float,
) -> np.ndarray:
    """Evaluate ``dv/dx - du/dy`` at scattered points in one batched call."""
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.empty(points.shape[:-1], dtype=np.float64)
    offsets = np.asarray(
        [[delta, 0.0], [-delta, 0.0], [0.0, delta], [0.0, -delta]],
        dtype=np.float64,
    )
    samples = points[None, ...] + offsets[:, None, :]
    velocities = np.asarray(velocity_eval(samples, float(time)), dtype=np.float64)
    dv_dx = (velocities[0, :, 1] - velocities[1, :, 1]) / (2.0 * delta)
    du_dy = (velocities[2, :, 0] - velocities[3, :, 0]) / (2.0 * delta)
    return dv_dx - du_dy


def cumulative_absolute_curl(
    velocity_eval: Any,
    positions: np.ndarray,
    valid_lengths: np.ndarray,
    times: np.ndarray,
    maximum_length: int,
    *,
    method: str = "finite_difference",
    finite_difference_delta: float = 1.0e-4,
) -> np.ndarray:
    """Return the VortexViz information vector from equations (4) and (6).

    Short trajectories are zero padded after their last valid cumulative value;
    this preserves the early-stop/trajectory-length signal used by VortexViz.
    """
    positions = np.asarray(positions, dtype=np.float64)
    valid_lengths = np.asarray(valid_lengths, dtype=np.int64).reshape(-1)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if maximum_length != positions.shape[1] or times.size != maximum_length:
        raise ValueError(
            "maximum_length, positions.shape[1], and len(times) must match.")
    if finite_difference_delta <= 0.0:
        raise ValueError("finite_difference_delta must be positive.")
    if method not in {"finite_difference", "autograd"}:
        raise ValueError("curl method must be 'finite_difference' or 'autograd'.")

    curls = np.zeros((positions.shape[0], maximum_length), dtype=np.float64)
    for step, time in enumerate(times):
        active = valid_lengths > step
        if not np.any(active):
            break
        points = positions[active, step]
        if method == "autograd":
            differentiable_eval = getattr(velocity_eval, "torch_eval", None)
            if differentiable_eval is None:
                raise ValueError("Autograd curl requires velocity_eval.torch_eval.")
            values = vatistas._autograd_vorticity_at_points(
                differentiable_eval, points, float(time))
        else:
            values = _finite_difference_vorticity(
                velocity_eval, points, float(time), finite_difference_delta)
        curls[active, step] = np.abs(values)

    cumulative = np.cumsum(curls, axis=1, dtype=np.float64)
    valid_mask = np.arange(maximum_length)[None, :] < valid_lengths[:, None]
    cumulative[~valid_mask] = 0.0
    return cumulative.astype(np.float32)


def _integrate_vortexviz_pathlines(
    velocity_eval: Any,
    label_eval: Any,
    pcfg: dict[str, Any],
    ocfg: dict[str, Any],
    rng: np.random.Generator,
    maximum_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Seed, integrate, and label one observer's VortexViz pathlines."""
    seeds = vatistas._pathline_seedings(
        pcfg, rng, velocity_eval=velocity_eval, time=ocfg["tmin"])
    positions, valid_lengths, times = vatistas._integrate_pathline_positions(
        velocity_eval, seeds, ocfg["tmin"], ocfg["tmax"], maximum_length)

    # Retain the base generator's policy for very short out-of-domain paths.
    for _ in range(int(pcfg["max_reseed_attempts"])):
        failed = valid_lengths <= 2
        if not np.any(failed):
            break
        shifts = rng.uniform(1.0e-5, 0.5, int(failed.sum()))
        seeds[failed] *= 1.0 - shifts[:, None]
        positions, valid_lengths, times = vatistas._integrate_pathline_positions(
            velocity_eval, seeds, ocfg["tmin"], ocfg["tmax"], maximum_length)

    seed_signed_distance = label_eval(seeds.reshape(-1, 1, 2)).reshape(-1)
    labels = (seed_signed_distance > 0.0).astype(np.uint8)
    return positions, valid_lengths, times, labels


def _validate_split_ratios(values: Iterable[float]) -> tuple[float, float, float]:
    ratios = tuple(float(value) for value in values)
    if len(ratios) != 3 or any(value < 0.0 for value in ratios):
        raise ValueError("split ratios must be three non-negative values.")
    total = sum(ratios)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError(f"split ratios must sum to 1, got {total}.")
    return ratios


def _group_split_map(
    group_ids: Iterable[int],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[int, str]:
    """Assign whole steady-field groups to splits with deterministic shuffling."""
    unique_ids = np.asarray(sorted(set(int(value) for value in group_ids)), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    shuffled = unique_ids[rng.permutation(unique_ids.size)]
    train_end = int(math.floor(unique_ids.size * ratios[0]))
    validation_end = train_end + int(math.floor(unique_ids.size * ratios[1]))
    result: dict[int, str] = {}
    for split, values in zip(
        SPLIT_NAMES,
        (shuffled[:train_end], shuffled[train_end:validation_end], shuffled[validation_end:]),
    ):
        result.update({int(value): split for value in values})
    return result


def _split_interleaved_order(
    order: Iterable[int],
    split_map: dict[int, str],
) -> np.ndarray:
    """Interleave split groups so a small verification subset covers all splits."""
    buckets = {
        split: [int(value) for value in order if split_map[int(value)] == split]
        for split in SPLIT_NAMES
    }
    result: list[int] = []
    offsets = {split: 0 for split in SPLIT_NAMES}
    while len(result) < sum(len(values) for values in buckets.values()):
        for split in SPLIT_NAMES:
            offset = offsets[split]
            if offset < len(buckets[split]):
                result.append(buckets[split][offset])
                offsets[split] += 1
    return np.asarray(result, dtype=np.int64)


def _prepare_output_directories(output_dir: Path, subfolder: str) -> None:
    for split in SPLIT_NAMES:
        (output_dir / split / subfolder).mkdir(parents=True, exist_ok=True)


def _write_meta(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _base_meta(
    image_size: int,
    maximum_length: int,
    split_ratios: tuple[float, float, float],
    split_seed: int,
    subfolder: str,
    information_source: str,
) -> dict[str, Any]:
    return {
        "format": "pyflowvis_vortexviz_cereal_v1",
        "Xdim": int(image_size),
        "Ydim": int(image_size),
        "maximumPathlineLengthVortexViz": int(maximum_length),
        "informationVector": "cumulative_absolute_curl",
        "informationVectorSource": information_source,
        "informationVectorPadding": "zero_after_trajectory_exit",
        "imageRepresentation": "seed_centered_isotropic_fit_binary_pathline",
        "binaryLayout": "cereal uint64 count + contiguous payload",
        "splitNames": list(SPLIT_NAMES),
        "splitRatios": list(split_ratios),
        "splitSeed": int(split_seed),
        "sampleSubfolder": subfolder,
        "splitGrouping": "steady_field_all_observers_together",
    }


def _vortexviz_config(cfg: vatistas.EasyConfig, args: argparse.Namespace) -> dict[str, Any]:
    section = cfg.get("vortex_viz", {}) or {}
    ratios = _validate_split_ratios(
        args.split_ratios
        if args.split_ratios is not None
        else section.get("split_ratios", DEFAULT_SPLIT_RATIOS)
    )
    return {
        "output_dir": Path(
            args.out_dir or section.get("out_dir", DEFAULT_OUTPUT_DIR)),
        "image_size": int(
            args.image_size
            if args.image_size is not None
            else section.get("image_size", DEFAULT_IMAGE_SIZE)),
        "maximum_length": int(
            args.maximum_pathline_length
            if args.maximum_pathline_length is not None
            else section.get(
                "maximum_pathline_length", DEFAULT_MAX_PATHLINE_LENGTH)),
        "split_ratios": ratios,
        "split_seed": int(
            args.split_seed
            if args.split_seed is not None
            else section.get("split_seed", int(cfg.get("seed", 0)) + 101)),
        "subfolder": str(section.get("subfolder", "velocity_mix")),
        "curl_method": str(section.get("curl_method", "finite_difference")),
        "curl_delta": float(section.get("curl_delta", 1.0e-4)),
    }


def generate_vortexviz_dataset(
    cfg: vatistas.EasyConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Generate a physically faithful VortexViz dataset from Vatistas fields."""
    settings = _vortexviz_config(cfg, args)
    output_dir: Path = settings["output_dir"]
    image_size = settings["image_size"]
    maximum_length = settings["maximum_length"]
    if maximum_length < 2:
        raise ValueError("maximum_pathline_length must be at least 2.")
    if image_size != DEFAULT_IMAGE_SIZE:
        LOGGER.warning(
            "The current vortex_viz.yaml uses DataSizeX=%d; set it to %d before training.",
            DEFAULT_IMAGE_SIZE,
            image_size,
        )

    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    bounds = vatistas.build_bounds(cfg)
    if args.refit:
        distribution, fitted = vatistas.refit(
            cfg.get("fit_config", "config/FittingVatistas.yaml"), device)
    else:
        distribution, fitted = vatistas.load_fit_outputs(
            cfg.get("fit_dir", "outputs/vatistas_fit"))

    specs, counts = vatistas.build_steady_specs(
        distribution, fitted, cfg, bounds, device)
    ocfg = vatistas.read_observer_cfg(cfg)
    if ocfg["transformation"] != "tobias":
        raise ValueError("VortexViz generation requires observer.transformation: tobias.")
    pcfg = vatistas.read_pathline_cfg(cfg)
    if not pcfg["enabled"]:
        raise ValueError("VortexViz generation requires pathline.enabled: true.")

    evaluators: list[Any] = []
    label_evaluators: list[Any] = []
    for spec in specs:
        velocity, label = vatistas.make_steady_eval(spec, bounds, device)
        evaluators.append(velocity)
        label_evaluators.append(label)

    target_total = len(specs) * int(ocfg["per_field"])
    configured_subset = int(cfg.output.get("unsteady_subset", 50))
    if args.max_unsteady is not None:
        to_write = min(target_total, int(args.max_unsteady))
    elif args.full:
        to_write = target_total
    else:
        to_write = min(target_total, configured_subset)
    if to_write <= 0:
        raise ValueError("No VortexViz samples requested.")

    _prepare_output_directories(output_dir, settings["subfolder"])
    split_map = _group_split_map(
        range(len(specs)), settings["split_ratios"], settings["split_seed"])
    rng = np.random.default_rng(seed + 17)
    order = rng.permutation(len(specs))
    if to_write < target_total:
        order = _split_interleaved_order(order, split_map)
    written = 0
    line_counts = {split: 0 for split in SPLIT_NAMES}
    file_counts = {split: 0 for split in SPLIT_NAMES}
    pathlines_per_observer = int(vatistas._pathline_seed_count(pcfg))
    estimated_payload_bytes = int(
        to_write
        * pathlines_per_observer
        * (maximum_length * np.dtype("<f4").itemsize + image_size * image_size + 1)
    )
    LOGGER.info(
        "[VortexViz] estimated binary payload: %.2f GiB in %d files",
        estimated_payload_bytes / float(1024 ** 3),
        to_write * 3,
    )
    if estimated_payload_bytes >= 32 * 1024 ** 3:
        LOGGER.warning(
            "The current VortexVizDataset eagerly converts every binary image to "
            "float32, so full-dataset training needs substantially more RAM than "
            "the on-disk payload."
        )

    initial_meta = _base_meta(
        image_size,
        maximum_length,
        settings["split_ratios"],
        settings["split_seed"],
        settings["subfolder"],
        "signed_vorticity_from_analytical_tobias_velocity",
    )
    initial_meta.update({
        "targetUnsteady": int(target_total),
        "writtenUnsteady": 0,
        "pathlineSamplingMethod": pcfg["sampling"],
        "PathlineCount": pathlines_per_observer,
        "observerTransformation": ocfg["transformation"],
        "curlMethod": settings["curl_method"],
        "curlFiniteDifferenceDelta": settings["curl_delta"],
        "estimatedBinaryPayloadBytes": estimated_payload_bytes,
    })
    _write_meta(output_dir, initial_meta)

    for steady_index in order:
        velocity = evaluators[int(steady_index)]
        label = label_evaluators[int(steady_index)]
        split = split_map[int(steady_index)]
        folder = output_dir / split / settings["subfolder"]
        for _observer_index in range(int(ocfg["per_field"])):
            _abc_function, observer = vatistas.sample_observer(rng, ocfg)
            unsteady_velocity = vatistas._make_tobias_velocity_evaluator(
                velocity,
                observer[:3],
                observer[3:],
                ocfg["tmin"],
                ocfg["tmax"],
                ocfg["timesteps"],
            )
            positions, valid_lengths, times, labels = _integrate_vortexviz_pathlines(
                unsteady_velocity, label, pcfg, ocfg, rng, maximum_length)
            information = cumulative_absolute_curl(
                unsteady_velocity,
                positions,
                valid_lengths,
                times,
                maximum_length,
                method=settings["curl_method"],
                finite_difference_delta=settings["curl_delta"],
            )
            images = pathlines_to_seed_centered_images(
                positions, valid_lengths, image_size=image_size)
            type_tag = vatistas._pathline_sample_type_tag(specs[int(steady_index)])
            sample_tag = f"sample_{written:06d}_{type_tag}"
            _write_vortexviz_triplet(
                folder, sample_tag, information, images, labels)
            line_counts[split] += int(labels.size)
            file_counts[split] += 1
            written += 1
            if written == 1 or written % max(1, to_write // 20) == 0:
                LOGGER.info(
                    "[VortexViz] wrote %d/%d observer samples (%d pathlines each)",
                    written,
                    to_write,
                    labels.size,
                )
            if written >= to_write:
                break
        if written >= to_write:
            break

    meta = dict(initial_meta)
    meta.update({
        "writtenUnsteady": int(written),
        "observerFilesPerSplit": file_counts,
        "pathlineExamplesPerSplit": line_counts,
        "sourceCounts": {
            "random": int(counts[vatistas.SRC_RANDOM]),
            "fitted": int(counts[vatistas.SRC_FITTED]),
            "killing": int(counts[vatistas.SRC_KILLING]),
        },
    })
    _write_meta(output_dir, meta)
    LOGGER.info("[VortexViz] dataset ready at %s", output_dir)
    return meta


def _sample_number(path: Path, fallback: int) -> int:
    match = re.search(r"sample_(\d+)", path.name)
    return int(match.group(1)) if match else int(fallback)


def convert_existing_pathlines(
    source_dir: Path,
    cfg: vatistas.EasyConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Convert existing 7-feature pathlines using cumulative IVD as a proxy.

    Signed curl cannot be reconstructed from IVD alone.  The resulting files
    are structurally compatible with VortexVizDataset, while metadata records
    the semantic difference so they cannot be mistaken for faithful features.
    """
    settings = _vortexviz_config(cfg, args)
    output_dir: Path = settings["output_dir"]
    meta_path = source_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"No source meta.json found at {meta_path}.")
    with meta_path.open("r", encoding="utf-8") as handle:
        source_meta = json.load(handle)
    source_files = sorted(
        path for path in source_dir.rglob("*_pathline.bin") if path.is_file())
    if args.max_unsteady is not None:
        source_files = source_files[: int(args.max_unsteady)]
    if not source_files:
        raise FileNotFoundError(f"No *_pathline.bin files found below {source_dir}.")

    pathline_count = int(source_meta["PathlineCount"])
    source_length = int(source_meta["outputPathlineLength"])
    feature_count = int(source_meta["PathlineFeature"])
    if feature_count < 5:
        raise ValueError("Existing pathlines need at least five features including label.")
    observers_per_field = int(args.observers_per_field)
    sample_numbers = [
        _sample_number(path, index) for index, path in enumerate(source_files)]
    group_ids = [number // observers_per_field for number in sample_numbers]
    split_map = _group_split_map(
        group_ids, settings["split_ratios"], settings["split_seed"])
    _prepare_output_directories(output_dir, settings["subfolder"])
    LOGGER.warning(
        "Converting existing pathlines with cumulative absolute IVD as a curl proxy; "
        "use direct generation for the paper-faithful signed-curl feature.")

    file_counts = {split: 0 for split in SPLIT_NAMES}
    line_counts = {split: 0 for split in SPLIT_NAMES}
    maximum_length = settings["maximum_length"]
    for index, (source_path, _sample_number_value, group_id) in enumerate(
        zip(source_files, sample_numbers, group_ids)):
        values = _read_cereal_vector(source_path, "<f4")
        expected = pathline_count * source_length * feature_count
        if values.size != expected:
            raise ValueError(
                f"Unexpected pathline payload in {source_path}: got {values.size}, "
                f"expected {expected}.")
        pathlines = values.reshape(pathline_count, source_length, feature_count)
        positions = pathlines[..., 0:2].astype(np.float64, copy=False)
        finite = np.all(np.isfinite(positions), axis=-1)
        not_padding = np.all(positions > -999.0, axis=-1)
        valid = finite & not_padding
        valid_lengths = np.sum(valid, axis=1, dtype=np.int64)
        labels = (pathlines[:, 0, 4] > 0.5).astype(np.uint8)
        proxy = np.zeros((pathline_count, maximum_length), dtype=np.float32)
        copy_length = min(source_length, maximum_length)
        ivd = np.abs(pathlines[:, :copy_length, 3].astype(np.float64))
        step_mask = np.arange(copy_length)[None, :] < valid_lengths[:, None]
        ivd[~step_mask] = 0.0
        proxy[:, :copy_length] = np.cumsum(ivd, axis=1, dtype=np.float64)
        proxy[:, :copy_length][~step_mask] = 0.0
        images = pathlines_to_seed_centered_images(
            positions, valid_lengths, image_size=settings["image_size"])
        split = split_map[group_id]
        folder = output_dir / split / settings["subfolder"]
        sample_tag = source_path.name.removesuffix("_pathline.bin")
        _write_vortexviz_triplet(folder, sample_tag, proxy, images, labels)
        file_counts[split] += 1
        line_counts[split] += pathline_count
        if index == 0 or (index + 1) % max(1, len(source_files) // 20) == 0:
            LOGGER.info(
                "[VortexViz conversion] wrote %d/%d source files",
                index + 1,
                len(source_files),
            )

    meta = _base_meta(
        settings["image_size"],
        maximum_length,
        settings["split_ratios"],
        settings["split_seed"],
        settings["subfolder"],
        "cumulative_absolute_ivd_proxy_from_existing_pathline_feature_3",
    )
    meta.update({
        "informationVector": "cumulative_absolute_ivd_proxy",
        "paperFaithfulCurl": False,
        "sourceDataset": str(source_dir),
        "sourcePathlineLength": source_length,
        "sourcePathlineFeatureCount": feature_count,
        "observerFilesPerSplit": file_counts,
        "pathlineExamplesPerSplit": line_counts,
        "writtenUnsteady": len(source_files),
        "observersPerSteadyField": observers_per_field,
    })
    _write_meta(output_dir, meta)
    LOGGER.info("[VortexViz conversion] dataset ready at %s", output_dir)
    return meta


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Vatistas/Tobias data for the DeSilva VortexViz model.")
    parser.add_argument("--config", default="config/VatistasDataset.yaml")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--refit", action="store_true")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Generate all configured steady-field observer variants.",
    )
    parser.add_argument(
        "--max-unsteady",
        type=int,
        default=None,
        help="Hard cap for smoke tests or partial generation; overrides --full.",
    )
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--maximum-pathline-length", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VALIDATION", "TEST"),
        default=None,
    )
    parser.add_argument(
        "--convert-existing",
        type=Path,
        default=None,
        metavar="DATASET_ROOT",
        help="Convert existing 7-feature pathline binaries using IVD as a proxy.",
    )
    parser.add_argument(
        "--observers-per-field",
        type=int,
        default=20,
        help="Grouping used to prevent split leakage in --convert-existing mode.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    args = build_argument_parser().parse_args()
    if args.max_unsteady is not None and args.max_unsteady <= 0:
        raise ValueError("--max-unsteady must be positive.")
    if args.observers_per_field <= 0:
        raise ValueError("--observers-per-field must be positive.")
    cfg = vatistas.EasyConfig()
    cfg.load(args.config, recursive=False)
    if args.convert_existing is not None:
        convert_existing_pathlines(args.convert_existing, cfg, args)
    else:
        generate_vortexviz_dataset(cfg, args)


if __name__ == "__main__":
    main()
