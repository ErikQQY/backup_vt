#!/usr/bin/env python3
"""Infer a cylinder-wake time series with trainedVortexTransformerBetter.

Reuses the repository's cylinder FEATURE_ADAPTIVE_MESH_SEEDING adapter in
detect_test.py, including its finite-difference curl monitor, patch-local
coordinates, normalized time/velocity/IVD, RK4, and short-trajectory reseeding.
This is model inference on NetCDF data; no ground-truth accuracy is implied.

Example (the existing vortex conda environment contains the dependencies)::

    python animate_cylinder_vortices.py --stride 10

Every displayed time is an independently inferred initial condition. Forward
pathlines need one additional time unit, so default output spans t=0..14 while
reading the full t=0..15 input. The probability projection is spatially smoothed
only; no interpolated, extrapolated, or temporally smoothed predictions are used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import netCDF4 as nc
import numpy as np
from scipy.ndimage import label
import torch

import detect as common
import detect_test as adaptive


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, default=ROOT / "dataset/cylinder2d.nc")
    result.add_argument("--checkpoint", type=Path,
                        default=ROOT / "trainedVortexTransformerBetter/best_checkpoint.pth.tar")
    result.add_argument("--torchscript", type=Path,
                        default=ROOT / "trainedVortexTransformerBetter/BackUp_best_checkpoint.pthtraced.pt")
    result.add_argument("--output-dir", type=Path,
                        default=ROOT / "output/cylinder2d_vortex_animation")
    result.add_argument("--start-index", type=int, default=0)
    result.add_argument("--end-index", type=int, default=None,
                        help="Inclusive; default is the last complete forward window.")
    result.add_argument("--stride", type=int, default=10)
    result.add_argument("--duration", type=float, default=1.0)
    result.add_argument("--tile-size", type=float, default=1.0)
    result.add_argument("--tile-stride", type=float, default=0.5)
    result.add_argument("--threshold", type=float, default=0.5)
    result.add_argument("--smooth-sigma", type=float, default=3.0)
    result.add_argument("--min-component-pixels", type=int, default=8)
    result.add_argument("--seed", type=int, default=7068)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--fps", type=int, default=12)
    result.add_argument("--dpi", type=int, default=120)
    result.add_argument("--skip-render", action="store_true")
    return result


def threshold_regions(probability, support, threshold, min_pixels, x, y):
    components, count = label((probability >= threshold) & (support > 1e-5))
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    keep = sizes >= min_pixels
    keep[0] = False
    mask = keep[components]
    regions = []
    for component in np.flatnonzero(keep):
        yy, xx = np.nonzero(components == component)
        values = probability[yy, xx]
        weights = np.maximum(values - threshold, 1e-6)
        regions.append((float(np.average(x[xx], weights=weights)),
                        float(np.average(y[yy], weights=weights)),
                        float(values.max()), len(xx)))
    regions.sort()
    return mask, np.asarray(regions, dtype=np.float64).reshape(-1, 4)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.stride < 1 or args.duration <= 0 or args.threads < 1 or args.batch_size < 1:
        raise ValueError("Stride, duration, threads and batch size must be positive.")
    if not 0 < args.threshold < 1 or args.smooth_sigma <= 0 or args.min_component_pixels < 1:
        raise ValueError("Require 0<threshold<1, positive sigma and minimum region size.")
    if args.fps < 1 or args.dpi < 1:
        raise ValueError("FPS and DPI must be positive.")
    with nc.Dataset(args.input) as dataset:
        all_times = np.asarray(dataset["tdim"][:], dtype=np.float64)
        x = np.asarray(dataset["xdim"][:], dtype=np.float64)
        y = np.asarray(dataset["ydim"][:], dtype=np.float64)
        radius = float(np.asarray(dataset["radius"][:]).flat[0])
        units = {name: getattr(dataset[name], "units", "unspecified")
                 for name in ("xdim", "ydim", "tdim", "u", "v")}
    last = int(np.searchsorted(all_times, all_times[-1] - args.duration, side="right") - 1)
    end = last if args.end_index is None else args.end_index
    if not 0 <= args.start_index <= end <= last:
        raise ValueError(f"Require 0 <= start-index <= end-index <= {last} for this duration.")
    indices = np.arange(args.start_index, end + 1, args.stride, dtype=int)
    if indices[-1] != end:
        indices = np.append(indices, end)

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    adapter_args = adaptive.build_parser().parse_args([])
    adapter_args.duration, adapter_args.seed = args.duration, args.seed
    config = ROOT / "config/segmentation/pathline_transformer_v1.yaml"
    model, epoch, backend = adaptive.load_v1_model(
        args.checkpoint, args.torchscript, config, "cpu", "torchscript")
    patches = common.make_cross_patches(x, y, args.tile_size, args.tile_stride)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "frame_cache"
    cache_dir.mkdir(exist_ok=True)
    settings = {key: str(value.resolve()) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key not in ("output_dir", "skip_render", "fps", "dpi")}
    settings["input_sha256"] = sha256(args.input)
    settings["checkpoint_sha256"] = sha256(args.checkpoint)
    settings["torchscript_sha256"] = sha256(args.torchscript)
    settings["adapter_sha256"] = {name: sha256(ROOT / name) for name in
                                 ("detect.py", "detect_test.py", Path(__file__).name)}
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    print(f"Loaded Better checkpoint epoch={epoch}, backend={backend}, device=cpu, "
          f"patches={len(patches)}, input=[B,16,256,7]", flush=True)
    print(f"Inferring {len(indices)} frames, t={all_times[indices[0]]:g}.."
          f"{all_times[indices[-1]]:g}, forward duration={args.duration:g}", flush=True)

    stack_keys = ("u", "v", "vorticity", "probability", "mask", "support",
                  "seed_points", "seed_probability", "seed_valid")
    frames = {key: [] for key in stack_keys}
    frame_summary, locations = [], []
    wall_start = time.perf_counter()
    determinism_checked = False
    for number, index in enumerate(indices):
        frame_start = time.perf_counter()
        cache_path = cache_dir / f"frame_{index:04d}.npz"
        cached = None
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as stored:
                if str(stored["signature"]) == signature:
                    cached = {key: stored[key] for key in stored.files}
        if cached is None:
            flow, start, u, v, _ = common.load_flow_window(args.input, int(index), args.duration)
            prepared, reseed_passes, invalid_count = adaptive.prepare_adaptive_inputs(
                flow, patches, adapter_args, start, (0.0, 0.0), radius)
            # Reset the seed for every frame so stochastic graph operations, if
            # present in a supplied trace, are reproducible without extra noise.
            torch.manual_seed(args.seed)
            raw_scores = common.predict_batches(model, prepared.features, "cpu", args.batch_size)
            if not determinism_checked:
                torch.manual_seed(args.seed)
                repeated = common.predict_batches(model, prepared.features, "cpu", args.batch_size)
                np.testing.assert_allclose(raw_scores, repeated, rtol=0, atol=1e-7)
                determinism_checked = True
            if not np.isfinite(raw_scores).all() or np.any((raw_scores < 0) | (raw_scores > 1)):
                raise RuntimeError("Model returned nonfinite or non-probability outputs.")
            points, scores, _ = common.merge_duplicate_seeds(
                prepared.seeds, raw_scores, prepared.display_valid)
            probability, support = common.build_probability_map(x, y, points, scores, args.smooth_sigma)
            common.mask_obstacle(probability, support, x, y, (0.0, 0.0), radius)
            mask, regions = threshold_regions(probability, support, args.threshold,
                                              args.min_component_pixels, x, y)
            cached = dict(signature=np.asarray(signature), u=u, v=v,
                          vorticity=flow.vorticity[0], probability=probability.astype(np.float32),
                          support=support.astype(np.float32), mask=mask, regions=regions,
                          seed_points=np.concatenate(prepared.seeds).astype(np.float32),
                          seed_probability=raw_scores.ravel(),
                          seed_valid=np.concatenate(prepared.display_valid),
                          reseed_passes=np.asarray(reseed_passes),
                          invalid_seeds=np.asarray(invalid_count))
            np.savez_compressed(cache_path, **cached)
        for key in stack_keys:
            frames[key].append(cached[key])
        for region_id, (cx, cy, peak, pixels) in enumerate(cached["regions"], 1):
            locations.append((number, int(index), float(all_times[index]), region_id,
                              cx, cy, peak, int(pixels)))
        valid_scores = cached["seed_probability"][cached["seed_valid"]]
        frame_summary.append(dict(frame=number, time_index=int(index), time=float(all_times[index]),
                                  regions=len(cached["regions"]),
                                  positive_seeds=int(np.count_nonzero(valid_scores >= args.threshold)),
                                  valid_seeds=len(valid_scores),
                                  detected_pixels=int(cached["mask"].sum()),
                                  coverage_fraction=float(np.isfinite(cached["probability"]).mean()),
                                  probability_max=float(valid_scores.max())))
        if number % 10 == 0 or number == len(indices) - 1:
            elapsed = time.perf_counter() - wall_start
            print(f"[{number + 1:3d}/{len(indices)}] t={all_times[index]:5.2f} "
                  f"regions={len(cached['regions']):2d} "
                  f"positive seeds={frame_summary[-1]['positive_seeds']}/{len(valid_scores)} "
                  f"frame={time.perf_counter() - frame_start:.2f}s elapsed={elapsed:.1f}s", flush=True)

    output = args.output_dir / "predictions.npz"
    np.savez_compressed(output, x=x, y=y, times=all_times[indices], time_indices=indices,
                        radius=radius, threshold=args.threshold,
                        **{key: np.stack(value) for key, value in frames.items()})
    with (args.output_dir / "detections.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("frame", "time_index", "time", "region_id_within_frame",
                         "x", "y", "peak_probability", "component_pixels"))
        writer.writerows(locations)
    with (args.output_dir / "frame_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_summary[0]))
        writer.writeheader()
        writer.writerows(frame_summary)
    metadata = dict(settings=settings, checkpoint_epoch=epoch, backend=backend, device="cpu",
                    frame_count=len(indices), time_range=all_times[indices[[0, -1]]].tolist(),
                    grid_shape=[len(y), len(x)], patch_count=len(patches), units=units,
                    frame_indices=indices.tolist(), inference_seconds=time.perf_counter() - wall_start,
                    input_features=["local_x", "local_y", "normalized_time", "normalized_IVD",
                                    "distance_from_seed", "normalized_u", "normalized_v"],
                    preprocessing="detect_test.py FEATURE_ADAPTIVE_MESH_SEEDING cylinder adapter",
                    notes=["Uses abs(finite-difference curl) as the existing cylinder adapter's seeding monitor.",
                           "IVD feature is absolute deviation from each patch's mean curl.",
                           "Coordinates are patch-local [-1,1]; time is [0,1]; velocity/IVD scale with duration.",
                           "15 overlapping square patches, 256 seeds and 16 RK4 samples per patch with default settings.",
                           "Same random seed per frame; no temporal smoothing or interpolated inference frames.",
                           "Probability map uses normalized Gaussian averaging of seed scores; NaN means unsupported/solid.",
                           "Connected region IDs reset each frame; centroids are not tracked vortex cores.",
                           "Threshold is fixed; no ground-truth labels or accuracy calibration used.",
                           "Final forward integration window reaches the input endpoint; no time extrapolation."])
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved predictions: {output}", flush=True)
    if not args.skip_render:
        from render_vortex_animation import main as render_main
        render_main(["--input", str(output), "--output-dir", str(args.output_dir),
                     "--fps", str(args.fps), "--dpi", str(args.dpi)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())