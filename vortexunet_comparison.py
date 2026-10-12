"""Shared VortexUNet inference adapter for the comparison scripts.

``TobiasVortexBoundaryUnet`` was trained with the generator's dense t0
velocity grids.  Its input is ``[u, v]``, unlike MVUNet's four-channel
``[u, v, curl, IVD]`` representation.  The paired pathline data used
``GRID_CROSS_SEEDING``; that value is comparison provenance rather than a
seeding operation required by this convolutional network.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, label

from VatistasFlowDatasetGenerator import (
    GRID_CROSS_SEEDING,
    _build_mvunet_velocity_features,
)
from mvunet_comparison import DEFAULT_MAX_VALUE, DEFAULT_MIN_VALUE


DEFAULT_CHECKPOINT = Path("trainedVortexUNet/best_checkpoint.pth.tar")
DEFAULT_TORCHSCRIPT = Path(
    "trainedVortexUNet/BackUp_best_checkpoint.pthtraced.pt"
)
TRAINING_SEEDING = GRID_CROSS_SEEDING
PREPROCESSING = "Vatistas-VortexUNet t0 [u,v] + global MinMax"


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    peak: float
    pixels: int
    rotation_sign: int

    @property
    def sign(self) -> int:
        return self.rotation_sign


@dataclass(frozen=True)
class Output:
    probability: np.ndarray
    mask: np.ndarray
    detections: list[Detection]
    epoch: int
    features: np.ndarray
    checkpoint: Path
    torchscript: Path


def add_arguments(
    parser: argparse.ArgumentParser,
    *,
    threshold: float = 0.5,
    smooth_sigma: float = 1.0,
    min_component_pixels: int = 8,
) -> None:
    group = parser.add_argument_group("VortexUNet")
    group.add_argument(
        "--vortexunet-checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    group.add_argument(
        "--vortexunet-torchscript", type=Path, default=DEFAULT_TORCHSCRIPT
    )
    group.add_argument(
        "--vortexunet-threshold", type=float, default=threshold
    )
    group.add_argument(
        "--vortexunet-smooth-sigma", type=float, default=smooth_sigma
    )
    group.add_argument(
        "--vortexunet-min-component-pixels",
        type=int,
        default=min_component_pixels,
    )
    group.add_argument(
        "--vortexunet-min-value",
        type=float,
        default=DEFAULT_MIN_VALUE,
        help="Fixed generator training-set minimum used by MinMaxNormalization.",
    )
    group.add_argument(
        "--vortexunet-max-value",
        type=float,
        default=DEFAULT_MAX_VALUE,
        help="Fixed generator training-set maximum used by MinMaxNormalization.",
    )


def validate_arguments(args: argparse.Namespace) -> None:
    if not 0.0 < args.vortexunet_threshold < 1.0:
        raise ValueError("--vortexunet-threshold must be in (0,1).")
    if args.vortexunet_smooth_sigma < 0.0:
        raise ValueError("--vortexunet-smooth-sigma cannot be negative.")
    if args.vortexunet_min_component_pixels < 1:
        raise ValueError("--vortexunet-min-component-pixels must be positive.")
    if not (
        np.isfinite(args.vortexunet_min_value)
        and np.isfinite(args.vortexunet_max_value)
        and args.vortexunet_max_value > args.vortexunet_min_value
    ):
        raise ValueError(
            "VortexUNet normalization bounds must be finite and ordered."
        )


def _spelling_variants(path: Path) -> tuple[Path, ...]:
    spellings = [str(path)]
    seen = set(spellings)
    replacements = (
        ("Backup_", "BackUp_"),
        ("BackUp_", "Backup_"),
        ("chekcpoint", "checkpoint"),
    )
    for old, new in replacements:
        for text in tuple(spellings):
            candidate = text.replace(old, new)
            if candidate not in seen:
                spellings.append(candidate)
                seen.add(candidate)
    return tuple(Path(text) for text in spellings)


def resolve_artifact(path: Path, default: Path) -> Path:
    for candidate in (*_spelling_variants(path), *_spelling_variants(default)):
        if candidate.is_file():
            if candidate != path:
                print(f"Path {path} not found; using repository file {candidate}.")
            return candidate
    raise FileNotFoundError(f"No VortexUNet model file found for {path}.")


def _load_model(
    checkpoint_path: Path,
    script_path: Path,
    device: torch.device,
) -> tuple[torch.jit.ScriptModule, int]:
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid VortexUNet checkpoint: {checkpoint_path}")
    state = checkpoint.get("state_dict", checkpoint.get("model"))
    if not isinstance(state, Mapping):
        raise ValueError(
            "VortexUNet checkpoint has no state_dict/model mapping: "
            f"{checkpoint_path}"
        )
    model = torch.jit.load(str(script_path), map_location=device)
    model_state = model.state_dict()
    if set(model_state) == set(state):
        compatible_state = state
    elif set(model_state) == {"model." + key for key in state}:
        compatible_state = {"model." + key: value for key, value in state.items()}
    else:
        raise RuntimeError(
            "VortexUNet TorchScript/checkpoint parameter names do not match."
        )
    incompatible = model.load_state_dict(compatible_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(str(incompatible))
    model.eval()
    return model, int(checkpoint.get("epoch", -1))


def _filtered_components(
    probability: np.ndarray,
    threshold: float,
    valid: np.ndarray,
    curl: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    min_pixels: int,
) -> tuple[np.ndarray, list[Detection]]:
    raw = valid & np.isfinite(probability) & (probability >= threshold)
    final = np.zeros_like(raw)
    detections: list[Detection] = []
    finite_curl = curl[valid & np.isfinite(curl)]
    mean_curl = float(np.mean(finite_curl)) if finite_curl.size else 0.0
    for rotation_sign, signed in (
        (1, curl >= mean_curl),
        (-1, curl < mean_curl),
    ):
        regions, count = label(raw & signed)
        for region_id in range(1, count + 1):
            yy, xx = np.nonzero(regions == region_id)
            if len(xx) < min_pixels:
                continue
            final[yy, xx] = True
            values = probability[yy, xx]
            weights = np.maximum(values - threshold, np.finfo(float).eps)
            detections.append(
                Detection(
                    x=float(np.average(x[xx], weights=weights)),
                    y=float(np.average(y[yy], weights=weights)),
                    peak=float(np.max(values)),
                    pixels=len(xx),
                    rotation_sign=rotation_sign,
                )
            )
    detections.sort(key=lambda item: (item.x, item.y))
    return final, detections


def run(
    snapshot_u: np.ndarray,
    snapshot_v: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    *,
    valid_mask: np.ndarray | None = None,
    border_cells: int = 0,
    device: str | torch.device = "cpu",
) -> Output:
    """Run VortexUNet on one t0 snapshot using its training contract."""

    validate_arguments(args)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    u = np.asarray(snapshot_u, dtype=np.float32)
    v = np.asarray(snapshot_v, dtype=np.float32)
    expected = (len(y), len(x))
    if u.shape != expected or v.shape != expected:
        raise ValueError(
            "VortexUNet snapshots must be shaped "
            f"{expected}; got {u.shape}/{v.shape}."
        )
    if np.any(np.diff(x) <= 0.0) or np.any(np.diff(y) <= 0.0):
        raise ValueError("VortexUNet requires ascending x/y coordinates.")

    features = np.ascontiguousarray(np.stack((u, v), axis=-1))
    # Preserve SteadyVelocityGridSegmentation's historical transpose exactly:
    # [Y,X,2] -> [2,X,Y].  Transpose the prediction back after inference.
    model_input = np.transpose(features, (2, 1, 0)).copy()
    model_input = (
        model_input - float(args.vortexunet_min_value)
    ) / float(args.vortexunet_max_value - args.vortexunet_min_value)
    tensor = torch.from_numpy(model_input[None]).to(torch.device(device))
    pad_x = (-tensor.shape[-2]) % 8
    pad_y = (-tensor.shape[-1]) % 8
    if pad_x or pad_y:
        tensor = F.pad(tensor, (0, pad_y, 0, pad_x), mode="replicate")

    checkpoint = resolve_artifact(
        args.vortexunet_checkpoint, DEFAULT_CHECKPOINT
    )
    torchscript = resolve_artifact(
        args.vortexunet_torchscript, DEFAULT_TORCHSCRIPT
    )
    model, epoch = _load_model(checkpoint, torchscript, tensor.device)
    with torch.inference_mode():
        prediction = model(tensor)
    if prediction.ndim != 3 or prediction.shape[0] != 1:
        raise RuntimeError(
            f"Unexpected VortexUNet output shape {tuple(prediction.shape)}."
        )
    probability = prediction[0, : len(x), : len(y)].cpu().numpy().T
    if probability.shape != expected:
        raise RuntimeError(
            f"VortexUNet output shape {probability.shape}, expected {expected}."
        )
    if args.vortexunet_smooth_sigma > 0.0:
        probability = gaussian_filter(
            probability,
            sigma=args.vortexunet_smooth_sigma,
            mode="nearest",
        )

    valid = np.isfinite(u) & np.isfinite(v)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != expected:
            raise ValueError(
                f"VortexUNet valid mask must be {expected}, got {supplied.shape}."
            )
        valid &= supplied
    if border_cells:
        if border_cells < 0 or 2 * border_cells >= min(expected):
            raise ValueError("Invalid VortexUNet border-cell count.")
        valid[:border_cells, :] = False
        valid[-border_cells:, :] = False
        valid[:, :border_cells] = False
        valid[:, -border_cells:] = False
    probability = np.where(valid, probability, np.nan)

    curl = _build_mvunet_velocity_features(
        features,
        domain=(float(x[0]), float(x[-1]), float(y[0]), float(y[-1])),
    )[..., 2]
    mask, detections = _filtered_components(
        probability,
        args.vortexunet_threshold,
        valid,
        curl,
        x,
        y,
        args.vortexunet_min_component_pixels,
    )
    return Output(
        probability=probability,
        mask=mask,
        detections=detections,
        epoch=epoch,
        features=features,
        checkpoint=checkpoint,
        torchscript=torchscript,
    )
