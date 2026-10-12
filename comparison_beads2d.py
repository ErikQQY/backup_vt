#!/usr/bin/env python3
"""Compare eight vortex detectors on the analytic ``beads2d.nc`` flow.

The four classical criteria and two transformer adapters reuse the tested
implementation in :mod:`comparison_jungtelziemniak2d`.  MVUNet uses the
VatistasDatasetGenerator t0 ``[u,v,curl,IVD]`` representation and the supplied
``trainedMVUNet`` checkpoint; ``GRID_CROSS_SEEDING`` is retained as the
training-data sampling provenance.
VortexUNet uses the generator's t0 ``[u,v]`` representation and the supplied
``trainedVortexUNet`` checkpoint with the same sampling provenance.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import comparison_jungtelziemniak2d as shared
import mvunet_comparison as mvunet
import vortexunet_comparison as vortexunet
from VatistasFlowDatasetGenerator import (
    FEATURE_ADAPTIVE_MESH_SEEDING,
    GRID_CROSS_SEEDING,
)


def parser() -> argparse.ArgumentParser:
    arguments = shared.parser()
    arguments.description = __doc__
    arguments.set_defaults(
        input=Path("dataset/beads2d.nc"),
        output=Path("outputs/beads2d_comparison.png"),
        time_index=0,
        duration=0.05,
        patch_size=4.0,
        patch_overlap=0.5,
    )
    mvunet.add_arguments(arguments)
    vortexunet.add_arguments(arguments)
    return arguments


def main() -> None:
    args = parser().parse_args()
    mvunet.validate_arguments(args)
    vortexunet.validate_arguments(args)
    for option, value in (
        ("--vt-threshold", args.vt_threshold),
        ("--fgma-threshold", args.fgma_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{option} must lie in [0,1].")
    if args.threshold is not None:
        if not 0.0 <= args.threshold <= 1.0:
            raise ValueError("--threshold must lie in [0,1].")
        warnings.warn(
            "--threshold overrides both transformer thresholds; prefer "
            "--vt-threshold and --fgma-threshold.",
            stacklevel=2,
        )
        vt_threshold = fgma_threshold = args.threshold
    else:
        vt_threshold = args.vt_threshold
        fgma_threshold = args.fgma_threshold
    if not 0.0 < args.ivd_percentile < 100.0:
        raise ValueError("--ivd-percentile must lie in (0,100).")
    if not 0.0 < args.classical_core_percentile < 100.0:
        raise ValueError("--classical-core-percentile must lie in (0,100).")

    device = shared.choose_device(args.device)
    window = shared.load_time_window(args.input, args.time_index, args.duration)
    tiles = shared.make_tiles(window, args.patch_size, args.patch_overlap)
    vt_checkpoint = shared.find_checkpoint(
        args.vt_checkpoint, (), "VortexTransformer checkpoint"
    )
    vt_script = shared.find_checkpoint(
        args.vt_script,
        (Path("trainedVortexTransformer/BackUp_best_checkpoint.pthtraced.pt"),),
        "VortexTransformer TorchScript",
    )
    fgma_checkpoint = shared.find_checkpoint(
        args.fgma_checkpoint,
        (Path("trainedVortexTransformerBetter/best_checkpoint.pth.tar"),),
        "FGMA-VT checkpoint",
    )
    fgma_script = shared.find_checkpoint(
        args.fgma_script,
        (
            Path(
                "trainedVortexTransformerBetter/"
                "BackUp_best_checkpoint.pthtraced.pt"
            ),
        ),
        "FGMA-VT TorchScript",
    )

    results = shared.instantaneous_criteria(
        window,
        args.smoothing_sigma,
        args.ivd_percentile,
        args.classical_core_percentile,
    )
    print(f"Using {len(tiles)} overlapping square inference tiles.")
    print(f"Running VortexTransformer with {GRID_CROSS_SEEDING}...")
    vt, vt_epoch, vt_valid = shared.neural_result(
        "VortexTransformer",
        GRID_CROSS_SEEDING,
        window,
        tiles,
        args.duration,
        vt_checkpoint,
        vt_script,
        device,
        vt_threshold,
        args.seed,
    )
    print(f"Running FGMA-VT with {FEATURE_ADAPTIVE_MESH_SEEDING}...")
    fgma, fgma_epoch, fgma_valid = shared.neural_result(
        "FGMA-VT",
        FEATURE_ADAPTIVE_MESH_SEEDING,
        window,
        tiles,
        args.duration,
        fgma_checkpoint,
        fgma_script,
        device,
        fgma_threshold,
        args.seed + 1000,
    )
    print(f"Running MVUNet with {mvunet.TRAINING_SEEDING} training data...")
    mv = mvunet.run(
        window.snapshot[..., 0],
        window.snapshot[..., 1],
        window.x,
        window.y,
        args,
        device="cpu",
    )
    mv_result = shared.Result(
        "MVUNet",
        mv.probability,
        mv.mask,
        f">= {args.mvunet_threshold:g}; {mvunet.TRAINING_SEEDING}",
    )
    print(
        f"Running VortexUNet with {vortexunet.TRAINING_SEEDING} "
        "training data..."
    )
    vu = vortexunet.run(
        window.snapshot[..., 0],
        window.snapshot[..., 1],
        window.x,
        window.y,
        args,
        device="cpu",
    )
    vu_result = shared.Result(
        "VortexUNet",
        vu.probability,
        vu.mask,
        f">= {args.vortexunet_threshold:g}; {vortexunet.TRAINING_SEEDING}",
    )
    results.extend((mv_result, vu_result, vt, fgma))
    print(
        f"VortexTransformer epoch={vt_epoch}, valid pathline samples={vt_valid:.1%}; "
        f"FGMA-VT epoch={fgma_epoch}, valid pathline samples={fgma_valid:.1%}; "
        f"MVUNet epoch={mv.epoch}, regions={len(mv.detections)}; "
        f"VortexUNet epoch={vu.epoch}, regions={len(vu.detections)}."
    )
    shared.draw_comparison(
        window, results, tiles, args.output, args.show, dataset_label="beads2d"
    )


if __name__ == "__main__":
    main()
