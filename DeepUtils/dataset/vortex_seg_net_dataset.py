"""Lazy binary dataset for Vortex-Seg-Net.

This adapter validates the Cereal vector header used by the repository and
supports either of the two actual feature modalities:

- ``*_pathline.bin`` float32 trajectories with embedded seed labels;
- dense velocity ``<sample>.bin`` or ``<sample>_velocity.bin`` paired with
  ``<sample>_segmentation.bin``.

Segmentation binaries are labels, never model inputs.  Both the uint8 masks
stored beside MVUnet velocity grids and the standalone float32 masks emitted
by the Python/C++ generators are accepted.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch

from .build import DATASETS


@DATASETS.register_module()
class VortexSegNetBinaryDataset(torch.utils.data.Dataset):
    """Read one homogeneous VortexTransformer modality per training run."""

    VALID_SOURCES = {"auto", "pathline", "velocity"}

    def __init__(
        self,
        data_dir,
        split,
        transform=None,
        input_source="auto",
        velocity_frame=0,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.root = Path(data_dir)
        self.split = str(split)
        self.transform = transform
        self.input_source = str(input_source).lower()
        if self.input_source not in self.VALID_SOURCES:
            raise ValueError(
                f"input_source must be one of {sorted(self.VALID_SOURCES)}."
            )
        self.velocity_frame = int(velocity_frame)
        if self.velocity_frame != 0:
            raise ValueError(
                "VortexTransformer segmentation binaries are aligned with "
                "velocity frame 0; velocity_frame must therefore be 0."
            )
        self.dastasetMetaInfo = self._load_root_metadata()
        self.sample_root = self._resolve_split_root()

        files = sorted(
            path
            for path in self.sample_root.rglob("*.bin")
            if path.is_file()
        )
        pathline_files = [
            path for path in files if path.name.endswith("_pathline.bin")
        ]
        segmentation_files = [
            path for path in files if path.name.endswith("_segmentation.bin")
        ]
        velocity_files = [
            path
            for path in files
            if not path.name.endswith("_pathline.bin")
            and not path.name.endswith("_segmentation.bin")
        ]
        segmentation_by_key = {}
        for path in segmentation_files:
            segmentation_by_key.setdefault(self._sample_key(path), []).append(path)
        velocity_pairs = []
        for path in velocity_files:
            candidates = segmentation_by_key.get(self._sample_key(path), [])
            if candidates:
                velocity_pairs.append(
                    (path, self._choose_segmentation(path, candidates))
                )

        if self.input_source == "auto":
            # Resolve once from the dataset root, not independently per split.
            # A mixed train=velocity/val=pathline run would otherwise build two
            # different model contracts without warning.
            self.input_source = self._preferred_source_for_root()

        if self.input_source == "velocity":
            if not velocity_pairs:
                raise FileNotFoundError(
                    "No velocity/segmentation pairs were found. Supported "
                    "names are <sample>.bin + <sample>_segmentation.bin or "
                    "<sample>_velocity.bin + <sample>_segmentation.bin."
                )
            self.samples = velocity_pairs
        else:
            if not pathline_files:
                raise FileNotFoundError(
                    f"No *_pathline.bin files found below {self.sample_root}."
                )
            self.samples = pathline_files
        self.dataName = [self._display_name(sample) for sample in self.samples]

    def _load_root_metadata(self):
        candidates = [
            self.root / "meta.json",
            self.root / "train" / "meta.json",
            self.root / "validation" / "meta.json",
        ]
        for path in candidates:
            if path.is_file():
                with path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
        raise FileNotFoundError(
            f"No root meta.json found for VortexSegNet dataset {self.root}."
        )

    def _resolve_split_root(self):
        split_name = "validation" if self.split == "val" else self.split
        candidates = [self.root / split_name]
        if self.split == "val":
            candidates.append(self.root / "test")
        elif self.split == "test":
            candidates.append(self.root / "validation")
        for path in candidates:
            if path.is_dir():
                return path
        raise FileNotFoundError(
            f"No directory for split={self.split!r} below {self.root}. "
            "Expected train/validation/test split folders."
        )

    def _preferred_source_for_root(self):
        files = sorted(
            path for path in self.root.rglob("*.bin") if path.is_file()
        )
        pathlines = [
            path for path in files if path.name.endswith("_pathline.bin")
        ]
        segmentations = [
            path for path in files if path.name.endswith("_segmentation.bin")
        ]
        segmentation_keys = {
            self._sample_key(path) for path in segmentations
        }
        velocities = [
            path
            for path in files
            if not path.name.endswith("_pathline.bin")
            and not path.name.endswith("_segmentation.bin")
            and self._sample_key(path) in segmentation_keys
        ]
        if velocities:
            return "velocity"
        if pathlines:
            return "pathline"
        if segmentations:
            raise ValueError(
                "Only *_segmentation.bin labels were found. Vortex-Seg-Net "
                "also needs paired velocity or pathline features."
            )
        raise FileNotFoundError(
            f"No supported .bin samples found below {self.root}."
        )

    @staticmethod
    def _sample_key(path):
        name = path.name
        for suffix in (
            "_segmentation.bin",
            "_pathline.bin",
            "_velocity.bin",
            ".bin",
        ):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return path.stem

    @staticmethod
    def _choose_segmentation(velocity_path, candidates):
        # Prefer the same-directory uint8 MVUnet pairing.  A single sibling
        # float32 mask is also unambiguous; duplicate basenames across task
        # folders must be disambiguated by reorganizing the dataset instead of
        # silently pairing the wrong label.
        colocated = [
            path for path in candidates if path.parent == velocity_path.parent
        ]
        if len(colocated) == 1:
            return colocated[0]
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError(
            f"Ambiguous segmentation pairing for {velocity_path}: "
            f"{[str(path) for path in candidates]}."
        )

    @staticmethod
    def _display_name(sample):
        path = sample[0] if isinstance(sample, tuple) else sample
        return str(path)

    @staticmethod
    def _read_cereal(path, dtype):
        dtype = np.dtype(dtype)
        with path.open("rb") as handle:
            header = handle.read(8)
            if len(header) != 8:
                raise ValueError(f"{path} has no complete Cereal uint64 header.")
            count = struct.unpack("<Q", header)[0]
            payload = np.fromfile(handle, dtype=dtype)
        if payload.size != count:
            raise ValueError(
                f"{path} header declares {count} values but contains "
                f"{payload.size} {dtype.name} values."
            )
        return payload

    @staticmethod
    def _read_segmentation(path, expected_count):
        file_size = path.stat().st_size
        if file_size == 8 + expected_count:
            payload = VortexSegNetBinaryDataset._read_cereal(path, np.uint8)
        elif file_size == 8 + 4 * expected_count:
            payload = VortexSegNetBinaryDataset._read_cereal(path, np.float32)
        else:
            raise ValueError(
                f"Cannot infer segmentation dtype for {path}: file size "
                f"{file_size}, expected {8 + expected_count} (uint8) or "
                f"{8 + 4 * expected_count} (float32) bytes."
            )
        payload = payload.astype(np.float32, copy=False)
        if not np.all(np.isfinite(payload)) or np.any(
            (payload < 0.0) | (payload > 1.0)
        ):
            raise ValueError(
                f"Segmentation labels in {path} must be finite values in [0,1]."
            )
        return payload

    def _load_velocity(self, pair):
        velocity_path, segmentation_path = pair
        height = int(self.dastasetMetaInfo["Ydim"])
        width = int(self.dastasetMetaInfo["Xdim"])
        pixel_count = height * width
        payload = self._read_cereal(velocity_path, np.float32)

        declared_channels = self.dastasetMetaInfo.get("velocityBinaryChannels")
        channel_count = len(declared_channels or [])
        layout = str(
            self.dastasetMetaInfo.get("velocityBinaryLayout", "")
        ).lower().replace(" ", "")
        current_layout = "[y,x,channel]" in layout
        temporal_layout = "[t,y,x,2]" in layout
        current_shape = (
            channel_count > 0
            and payload.size == pixel_count * channel_count
        )
        time_steps = int(
            self.dastasetMetaInfo.get("unsteadyFieldTimeStep", 0) or 0
        )
        temporal_shape = (
            time_steps > 0
            and payload.size == time_steps * pixel_count * 2
        )

        if current_layout:
            if not current_shape:
                raise ValueError(
                    f"{velocity_path} contradicts velocityBinaryLayout={layout!r}: "
                    f"expected Y*X*C={pixel_count}*{channel_count} values, got "
                    f"{payload.size}."
                )
            field = payload.reshape(height, width, channel_count)
        elif temporal_layout:
            if not temporal_shape:
                raise ValueError(
                    f"{velocity_path} contradicts velocityBinaryLayout={layout!r}."
                )
            field = payload.reshape(time_steps, height, width, 2)[0]
        elif temporal_shape and current_shape:
            raise ValueError(
                f"{velocity_path} is ambiguous: its payload fits both "
                "[Y,X,C] and [T,Y,X,2]. Set velocityBinaryLayout in meta.json."
            )
        elif temporal_shape:
            field = payload.reshape(time_steps, height, width, 2)[0]
        elif current_shape:
            field = payload.reshape(height, width, channel_count)
        else:
            inferred_channels, remainder = divmod(payload.size, pixel_count)
            if remainder == 0 and inferred_channels in {2, 3, 4}:
                channel_count = inferred_channels
                field = payload.reshape(height, width, channel_count)
            else:
                raise ValueError(
                    f"Velocity payload {velocity_path} with {payload.size} "
                    f"values has no unambiguous supported layout for "
                    f"Y={height}, X={width}; provide velocityBinaryLayout and "
                    "the matching metadata."
                )

        label = self._read_segmentation(
            segmentation_path, pixel_count
        ).reshape(height, width)
        data = torch.from_numpy(field.copy()).permute(2, 0, 1).contiguous()
        if self.transform is not None:
            data = self.transform(data)
        return data.to(torch.float32), torch.from_numpy(label.copy())

    def _load_pathline(self, path):
        length = int(self.dastasetMetaInfo.get("outputPathlineLength", 16))
        feature_count = int(self.dastasetMetaInfo.get("PathlineFeature", 7))
        line_count = int(self.dastasetMetaInfo.get("PathlineCount", 0) or 0)
        if line_count <= 0:
            count_k = int(
                self.dastasetMetaInfo.get("outputPathlinesCountK", 16)
            )
            line_count = (count_k // 2) ** 2 * 4
        payload = self._read_cereal(path, np.float32)
        expected = line_count * length * feature_count
        if payload.size != expected:
            raise ValueError(
                f"Pathline payload {path} has {payload.size} values; expected "
                f"K*L*C={line_count}*{length}*{feature_count}={expected}."
            )
        pathlines = payload.reshape(line_count, length, feature_count).copy()
        if feature_count <= 4:
            raise ValueError(
                "Pathline feature 4 is required for the embedded seed label."
            )
        if np.any(pathlines[:, 0, 0] <= -999.0):
            raise ValueError(
                f"Pathline payload {path} contains a padded seed at t=0; "
                "a label cannot be assigned to that line."
            )
        label = pathlines[:, 0, 4].copy()
        filename = path.name.lower()
        if (not self.dastasetMetaInfo.get("pathlineSeedLabel")
                and ("saddle" in filename or "zero" in filename)):
            label.fill(0.0)
        if not np.all(np.isfinite(label)) or np.any(
            (label < 0.0) | (label > 1.0)
        ):
            raise ValueError(
                f"Pathline seed labels in {path} must be finite values in [0,1]."
            )
        pathlines[:, 0, 4] = 0.0
        pathlines = torch.from_numpy(pathlines.transpose(1, 0, 2).copy())
        dummy_field = torch.zeros(1, 1, 1, 1, dtype=torch.float32)
        return (
            (dummy_field, pathlines.to(torch.float32)),
            torch.from_numpy(label.astype(np.float32, copy=False)),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if self.input_source == "velocity":
            return self._load_velocity(sample)
        return self._load_pathline(sample)

    def getSampleName(self, index):
        return self.dataName[index]


__all__ = ["VortexSegNetBinaryDataset"]
