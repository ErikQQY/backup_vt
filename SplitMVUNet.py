"""Split the generated MVUnet binaries into train/validation/test folders.

The source dataset is expected to have the layout produced by
``VatistasFlowDatasetGenerator.py``::

    outputs/vatistas_unsteady_mvunet/
        meta.json
        dataset_manifest.npz
        mvunet/
            sample_0....bin
            sample_0..._segmentation.bin
            ...

The output layout is intentionally matched to
``SteadyVelocityGridSegmentationMVUnet``::

    outputs/vatistas_mvunet_trainable/
        meta.json
        split_manifest.json
        train/mvunet/*.bin
        validation/mvunet/*.bin
        test/mvunet/*.bin

Each steady field has several consecutive observer variants.  Splitting at
observer-group boundaries keeps all variants of one steady field in the same
split and prevents train/validation/test leakage.

By default files are hard-linked, so the split dataset consumes almost no
additional disk space.  Use ``--mode copy`` when source and destination are on
different filesystems.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import re
import shutil
import struct
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


DEFAULT_SOURCE = Path("outputs/vatistas_unsteady_mvunet")
DEFAULT_TARGET = Path("outputs/vatistas_mvunet_trainable")
SAMPLE_PATTERN = re.compile(r"^sample_(\d+)")
SEGMENTATION_SUFFIX = "_segmentation.bin"
SPLIT_NAMES = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split MVUnet-ready [u,v,curl,IVD] binaries while keeping all "
            "observer variants of a steady field in one split."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"generated dataset root (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"split dataset root (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="fraction of steady-field observer groups used for training",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="fraction of steady-field observer groups used for validation",
    )
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="materialization mode (default: hardlink)",
    )
    parser.add_argument(
        "--skip-header-check",
        action="store_true",
        help="check file sizes and pairings but skip Cereal uint64 count checks",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="print progress after this many samples; 0 disables progress",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest_scalars(path: Path) -> dict:
    """Read only small scalar values needed for group-aware splitting."""
    if not path.is_file():
        return {}
    if np is None:
        raise ModuleNotFoundError(
            "NumPy is required to read dataset_manifest.npz. "
            "Install NumPy or provide a source without that manifest."
        )
    payload = {}
    with np.load(path, allow_pickle=False) as manifest:
        for key in (
            "target_steady",
            "target_unsteady",
            "written_unsteady",
            "observers_per_field",
            "velocity_bin_count",
            "velocity_bin_dir",
        ):
            if key not in manifest:
                continue
            value = manifest[key]
            payload[key] = value.item() if value.ndim == 0 else value.tolist()
    return payload


def validate_source_metadata(meta: dict, manifest: dict) -> tuple[int, int, int, int, str]:
    xdim = int(meta["Xdim"])
    ydim = int(meta["Ydim"])
    if xdim <= 0 or ydim <= 0:
        raise ValueError(f"Invalid grid size: Xdim={xdim}, Ydim={ydim}.")

    channels = list(meta.get("velocityBinaryChannels") or [])
    expected_channels = ["velocity_u", "velocity_v", "curl", "ivd"]
    if channels != expected_channels:
        raise ValueError(
            "MVUnet requires velocityBinaryChannels "
            f"{expected_channels}, got {channels}."
        )

    field_count = int(
        meta.get(
            "velocityBinaryCount",
            manifest.get("velocity_bin_count", 0),
        )
    )
    if field_count <= 0:
        raise ValueError("No MVUnet velocity binaries are declared in metadata.")

    observers_per_field = int(manifest.get("observers_per_field", 1))
    if observers_per_field <= 0:
        raise ValueError(
            f"Invalid observers_per_field={observers_per_field}."
        )
    if field_count % observers_per_field:
        raise ValueError(
            f"{field_count} samples cannot be divided into complete groups of "
            f"{observers_per_field} observer variants."
        )

    binary_dir = str(
        meta.get(
            "velocityBinaryDirectory",
            manifest.get("velocity_bin_dir", "mvunet"),
        )
    )
    if not binary_dir:
        raise ValueError("velocityBinaryDirectory is empty.")

    return xdim, ydim, field_count, observers_per_field, binary_dir


def calculate_boundaries(
    sample_count: int,
    observers_per_field: int,
    train_ratio: float,
    validation_ratio: float,
) -> tuple[int, int]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be in [0, 1).")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError(
            "--train-ratio + --validation-ratio must be less than 1."
        )

    group_count = sample_count // observers_per_field
    train_groups = int(group_count * train_ratio)
    validation_groups = int(group_count * validation_ratio)
    test_groups = group_count - train_groups - validation_groups
    if min(train_groups, validation_groups, test_groups) <= 0:
        raise ValueError(
            "The requested ratios leave an empty train, validation, or test split."
        )

    train_end = train_groups * observers_per_field
    validation_end = (
        train_groups + validation_groups
    ) * observers_per_field
    return train_end, validation_end


def choose_split(
    sample_id: int,
    train_end: int,
    validation_end: int,
) -> str:
    if sample_id < train_end:
        return "train"
    if sample_id < validation_end:
        return "validation"
    return "test"


def read_cereal_count(path: Path) -> int:
    with path.open("rb") as handle:
        header = handle.read(8)
    if len(header) != 8:
        raise ValueError(f"{path} is too short to contain a Cereal vector header.")
    return int(struct.unpack("<Q", header)[0])


def validate_pair(
    field_path: Path,
    segmentation_path: Path,
    xdim: int,
    ydim: int,
    check_header: bool,
) -> None:
    expected_field_count = xdim * ydim * 4
    expected_segmentation_count = xdim * ydim
    expected_field_size = 8 + expected_field_count * 4
    expected_segmentation_size = 8 + expected_segmentation_count

    field_size = field_path.stat().st_size
    segmentation_size = segmentation_path.stat().st_size
    if field_size != expected_field_size:
        raise ValueError(
            f"{field_path} has {field_size} bytes; MVUnet expects "
            f"{expected_field_size} bytes for float32[{ydim},{xdim},4]."
        )
    if segmentation_size != expected_segmentation_size:
        raise ValueError(
            f"{segmentation_path} has {segmentation_size} bytes; the loader "
            f"expects {expected_segmentation_size} bytes for "
            f"uint8[{ydim},{xdim}]."
        )

    if check_header:
        field_count = read_cereal_count(field_path)
        segmentation_count = read_cereal_count(segmentation_path)
        if field_count != expected_field_count:
            raise ValueError(
                f"{field_path} declares {field_count} values; expected "
                f"{expected_field_count}."
            )
        if segmentation_count != expected_segmentation_count:
            raise ValueError(
                f"{segmentation_path} declares {segmentation_count} values; "
                f"expected {expected_segmentation_count}."
            )


def materialize_file(source: Path, destination: Path, mode: str) -> str:
    """Create one output file without overwriting unrelated existing data."""
    if destination.exists():
        try:
            if os.path.samefile(source, destination):
                return "existing"
        except OSError:
            pass
        if (
            source.stat().st_size == destination.stat().st_size
            and filecmp.cmp(source, destination, shallow=False)
        ):
            return "existing"
        raise FileExistsError(
            f"Refusing to overwrite a different existing file: {destination}"
        )

    if mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)
    return "created"


def install_metadata(source_meta: Path, target_meta: Path) -> None:
    if target_meta.exists():
        if filecmp.cmp(source_meta, target_meta, shallow=False):
            return
        raise FileExistsError(
            f"Refusing to overwrite different metadata: {target_meta}"
        )
    shutil.copy2(source_meta, target_meta)


def main() -> None:
    args = parse_args()
    source_root = args.source.resolve()
    target_root = args.target.resolve()
    if source_root == target_root:
        raise ValueError("--source and --target must be different directories.")

    source_meta = source_root / "meta.json"
    meta = load_json(source_meta)
    manifest = load_manifest_scalars(source_root / "dataset_manifest.npz")
    (
        xdim,
        ydim,
        sample_count,
        observers_per_field,
        binary_dir_name,
    ) = validate_source_metadata(meta, manifest)

    source_binary_dir = source_root / binary_dir_name
    if not source_binary_dir.is_dir():
        raise NotADirectoryError(source_binary_dir)

    train_end, validation_end = calculate_boundaries(
        sample_count,
        observers_per_field,
        args.train_ratio,
        args.validation_ratio,
    )

    output_dirs = {
        split: target_root / split / binary_dir_name
        for split in SPLIT_NAMES
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    seen = bytearray(sample_count)
    split_counts = {split: 0 for split in SPLIT_NAMES}
    created_files = 0
    existing_files = 0

    with os.scandir(source_binary_dir) as entries:
        for entry in entries:
            if not entry.is_file(follow_symlinks=False):
                continue
            name = entry.name
            if not name.endswith(".bin") or name.endswith(SEGMENTATION_SUFFIX):
                continue

            match = SAMPLE_PATTERN.match(name)
            if match is None:
                raise ValueError(f"Cannot parse sample id from {entry.path}.")
            sample_id = int(match.group(1))
            if not 0 <= sample_id < sample_count:
                raise ValueError(
                    f"Sample id {sample_id} is outside [0, {sample_count})."
                )
            if seen[sample_id]:
                raise ValueError(f"Duplicate MVUnet sample id: {sample_id}.")

            field_path = Path(entry.path)
            segmentation_path = field_path.with_name(
                field_path.stem + SEGMENTATION_SUFFIX
            )
            if not segmentation_path.is_file():
                raise FileNotFoundError(
                    f"Missing paired segmentation for {field_path}: "
                    f"{segmentation_path}"
                )

            validate_pair(
                field_path,
                segmentation_path,
                xdim,
                ydim,
                check_header=not args.skip_header_check,
            )

            split = choose_split(sample_id, train_end, validation_end)
            destination_dir = output_dirs[split]
            for source_file in (field_path, segmentation_path):
                status = materialize_file(
                    source_file,
                    destination_dir / source_file.name,
                    args.mode,
                )
                if status == "created":
                    created_files += 1
                else:
                    existing_files += 1

            seen[sample_id] = 1
            split_counts[split] += 1
            completed = sum(split_counts.values())
            if (
                args.progress_every > 0
                and completed % args.progress_every == 0
            ):
                print(f"Processed {completed}/{sample_count} samples...")

    missing_ids = [index for index, present in enumerate(seen) if not present]
    if missing_ids:
        preview = ", ".join(str(value) for value in missing_ids[:20])
        suffix = "..." if len(missing_ids) > 20 else ""
        raise ValueError(
            f"Missing {len(missing_ids)} sample ids: {preview}{suffix}"
        )

    expected_counts = {
        "train": train_end,
        "validation": validation_end - train_end,
        "test": sample_count - validation_end,
    }
    if split_counts != expected_counts:
        raise RuntimeError(
            f"Unexpected split counts {split_counts}; expected {expected_counts}."
        )

    # Install root metadata only after every sample has passed validation, so an
    # interrupted/incomplete target is not mistaken for a finished dataset.
    target_root.mkdir(parents=True, exist_ok=True)
    install_metadata(source_meta, target_root / "meta.json")

    split_manifest = {
        "source": str(source_root),
        "binary_directory": binary_dir_name,
        "grid": {"Xdim": xdim, "Ydim": ydim},
        "channels": ["velocity_u", "velocity_v", "curl", "ivd"],
        "observers_per_field": observers_per_field,
        "split_strategy": "contiguous steady-field observer groups",
        "train_ratio": args.train_ratio,
        "validation_ratio": args.validation_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.validation_ratio,
        "sample_counts": split_counts,
        "file_mode": args.mode,
    }
    split_manifest_path = target_root / "split_manifest.json"
    if split_manifest_path.exists():
        existing_manifest = load_json(split_manifest_path)
        if existing_manifest != split_manifest:
            raise FileExistsError(
                "Refusing to overwrite a different split manifest: "
                f"{split_manifest_path}"
            )
    else:
        with split_manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(split_manifest, handle, indent=2)
            handle.write("\n")

    print("MVUnet dataset split completed.")
    print(f"Source: {source_root}")
    print(f"Target: {target_root}")
    print(f"Samples: {split_counts}")
    print(
        f"Files: {created_files} created, {existing_files} already present "
        f"({args.mode})"
    )


if __name__ == "__main__":
    main()
