"""Evaluate a trained MVUnet checkpoint on a complete test split.

Unlike the legacy ``test.py`` evaluator, this script computes the confusion
matrix over every pixel in the ``[B, H, W]`` prediction and target tensors.
By default it also applies the same scalar Min-Max normalization used during
training, but it deliberately does not add random test-time noise.

Example:
    python test_mvunet.py \
        --config config/segmentation/mvu_net.yaml \
        --data_dir ~/data/trainable_mvunet_01noise_seeding/ \
        --model_path outputModels/DengMVUnet/run/best_checkpoint.pth.tar \
        --save_samples 10
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from DeepUtils.dataset import build_dataloader_from_cfg
from DeepUtils.dataset.SteadyVastisDataset import getDatasetRootaMeta
from DeepUtils.models import build_model_from_cfg
from DeepUtils.utils.config import EasyConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DengMVUnet over every pixel in the test split."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data_dir", "--data-dir", type=Path, required=True)
    parser.add_argument(
        "--model_path", "--model-path", type=Path, required=True
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device such as cuda, cuda:0, or cpu (default: config value)",
    )
    parser.add_argument("--batch_size", "--batch-size", type=int, default=None)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=None)
    parser.add_argument(
        "--legacy_xy_layout", "--legacy-xy-layout", action="store_true",
        help="Reproduce the transposed input layout of checkpoints trained before the alignment fix.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to create the predicted mask",
    )
    parser.add_argument(
        "--no_normalize",
        "--no-normalize",
        action="store_true",
        help="Disable Min-Max normalization (only for a checkpoint trained raw)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate at most this many samples (useful for a quick check)",
    )
    parser.add_argument(
        "--save_samples",
        "--save-samples",
        type=int,
        default=0,
        help="Save this many probability/prediction/ground-truth image triplets",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=Path,
        default=Path("testOutput/mvunet"),
        help="Directory for optional images and JSON report",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path",
    )
    return parser.parse_args()


def select_device(requested: str | None, configured: str | None) -> torch.device:
    name = requested or configured or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        print("MPS is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return device


def configure_test(args: argparse.Namespace) -> tuple[EasyConfig, dict, torch.device]:
    config_path = args.config.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {data_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {model_path}")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be strictly between 0 and 1.")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch_size must be positive.")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num_workers cannot be negative.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    if args.save_samples < 0:
        raise ValueError("--save_samples cannot be negative.")

    cfg = EasyConfig()
    cfg.load(str(config_path), recursive=True)
    cfg.dataset["data_dir"] = str(data_dir)
    if args.legacy_xy_layout:
        cfg.dataset["legacy_xy_layout"] = True
    cfg["model_path"] = str(model_path)
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg.dataloader["num_workers"] = args.num_workers

    dataset_name = cfg.dataset.get("NAME")
    if dataset_name != "SteadyVelocityGridSegmentationMVUnet":
        raise ValueError(
            "test_mvunet.py requires dataset.NAME="
            f"SteadyVelocityGridSegmentationMVUnet, got {dataset_name!r}."
        )

    metadata = getDatasetRootaMeta(str(data_dir))
    if "datatransforms" not in cfg:
        cfg["datatransforms"] = EasyConfig()
    if "kwargs" not in cfg.datatransforms:
        cfg.datatransforms["kwargs"] = EasyConfig()

    if args.no_normalize:
        cfg.datatransforms["test"] = []
    else:
        missing = [key for key in ("minV", "maxV") if key not in metadata]
        if missing:
            raise KeyError(
                f"Dataset metadata is missing {missing}; cannot reproduce the "
                "training MinMaxNormalization. Use --no_normalize only if the "
                "checkpoint was trained without normalization."
            )
        min_value = float(metadata["minV"])
        max_value = float(metadata["maxV"])
        if not math.isfinite(min_value) or not math.isfinite(max_value):
            raise ValueError("Dataset minV/maxV must be finite.")
        if max_value <= min_value:
            raise ValueError(
                f"Dataset requires maxV > minV, got {min_value} and {max_value}."
            )
        cfg.datatransforms["test"] = ["MinMaxNormalization"]
        cfg.datatransforms.kwargs["minV"] = min_value
        cfg.datatransforms.kwargs["maxV"] = max_value

    device = select_device(args.device, cfg.get("device"))
    cfg["device"] = str(device)
    return cfg, metadata, device


def checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint must be a mapping, got {type(checkpoint).__name__}."
        )
    for key in ("state_dict", "model", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise KeyError(
        "Checkpoint contains no state_dict, model, or model_state_dict mapping."
    )


def load_model(cfg: EasyConfig, device: torch.device) -> tuple[torch.nn.Module, dict]:
    model = build_model_from_cfg(cfg.model)
    checkpoint = torch.load(cfg.model_path, map_location=device)
    state_dict = checkpoint_state_dict(checkpoint)

    model_keys = set(model.state_dict())
    if state_dict and all(key.startswith("module.") for key in state_dict):
        stripped = {key[len("module.") :]: value for key, value in state_dict.items()}
        if model_keys.intersection(stripped):
            state_dict = stripped

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def as_spatial_mask(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Convert a one-channel segmentation tensor to strict ``[B,H,W]``."""
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    elif tensor.ndim == 4 and tensor.shape[-1] == 1:
        tensor = tensor[..., 0]
    if tensor.ndim != 3:
        raise ValueError(
            f"{name} must have shape [B,H,W], [B,1,H,W], or [B,H,W,1]; "
            f"got {tuple(tensor.shape)}."
        )
    return tensor


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else float("nan")


def sample_stem(dataset: Any, sample_index: int) -> str:
    if hasattr(dataset, "getSampleName"):
        raw_name = str(dataset.getSampleName(sample_index))
    else:
        raw_name = f"sample_{sample_index}"
    stem = Path(raw_name).stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return stem or f"sample_{sample_index}"


def save_mask_images(
    output_dir: Path,
    dataset: Any,
    first_index: int,
    probabilities: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    remaining: int,
) -> int:
    if remaining <= 0:
        return 0
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Pillow is required when --save_samples is greater than zero."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    count = min(remaining, probabilities.shape[0])
    for local_index in range(count):
        stem = sample_stem(dataset, first_index + local_index)
        arrays = {
            "probability": np.clip(
                probabilities[local_index].numpy() * 255.0, 0, 255
            ).astype(np.uint8),
            "prediction": predictions[local_index].numpy().astype(np.uint8) * 255,
            "ground_truth": targets[local_index].numpy().astype(np.uint8) * 255,
        }
        for suffix, array in arrays.items():
            image = Image.fromarray(array, mode="L")
            image = image.resize(
                (image.width * 4, image.height * 4),
                Image.NEAREST if suffix != "probability" else Image.BILINEAR,
            )
            image.save(output_dir / f"{stem}_{suffix}.png")
    return count


def evaluate(
    model: torch.nn.Module,
    data_loader: Any,
    device: torch.device,
    threshold: float,
    limit: int | None,
    save_samples: int,
    output_dir: Path,
) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    sample_count = positive_target_samples = positive_prediction_samples = 0
    pixel_count = 0
    probability_sum = loss_sum = 0.0
    probability_min = float("inf")
    probability_max = float("-inf")
    target_min = float("inf")
    target_max = float("-inf")
    channel_min = channel_max = None
    saved = 0

    started = time.perf_counter()
    with torch.inference_mode():
        for data, target in tqdm(data_loader, desc="Evaluating MVUnet"):
            if not torch.is_tensor(data):
                raise TypeError(
                    "MVUnet expects one dense [B,4,H,W] tensor per batch, got "
                    f"{type(data).__name__}."
                )
            if limit is not None:
                remaining = limit - sample_count
                if remaining <= 0:
                    break
                if data.shape[0] > remaining:
                    data = data[:remaining]
                    target = target[:remaining]

            data = data.to(device, non_blocking=device.type == "cuda").float()
            target = target.to(device, non_blocking=device.type == "cuda").float()
            if data.ndim != 4 or data.shape[1] != 4:
                raise ValueError(
                    f"MVUnet input must have shape [B,4,H,W], got {tuple(data.shape)}."
                )

            batch_channel_min = data.amin(dim=(0, 2, 3)).detach().cpu()
            batch_channel_max = data.amax(dim=(0, 2, 3)).detach().cpu()
            channel_min = (
                batch_channel_min
                if channel_min is None
                else torch.minimum(channel_min, batch_channel_min)
            )
            channel_max = (
                batch_channel_max
                if channel_max is None
                else torch.maximum(channel_max, batch_channel_max)
            )

            probability = as_spatial_mask(model(data), "prediction")
            target = as_spatial_mask(target, "target")
            if probability.shape != target.shape:
                raise ValueError(
                    f"Prediction shape {tuple(probability.shape)} does not match "
                    f"target shape {tuple(target.shape)}."
                )

            batch_probability_min = float(probability.min().item())
            batch_probability_max = float(probability.max().item())
            if batch_probability_min < -1e-6 or batch_probability_max > 1.0 + 1e-6:
                raise ValueError(
                    "DengMVUnet must return probabilities in [0,1], got range "
                    f"[{batch_probability_min}, {batch_probability_max}]."
                )
            probability = probability.clamp(0.0, 1.0)
            target_binary = target >= 0.5
            prediction_binary = probability >= threshold

            tp += int((prediction_binary & target_binary).sum().item())
            fp += int((prediction_binary & ~target_binary).sum().item())
            fn += int((~prediction_binary & target_binary).sum().item())
            tn += int((~prediction_binary & ~target_binary).sum().item())

            batch_size = target.shape[0]
            batch_pixels = target.numel()
            sample_count += batch_size
            pixel_count += batch_pixels
            positive_target_samples += int(
                target_binary.reshape(batch_size, -1).any(dim=1).sum().item()
            )
            positive_prediction_samples += int(
                prediction_binary.reshape(batch_size, -1).any(dim=1).sum().item()
            )
            loss_sum += float(
                F.binary_cross_entropy(probability, target, reduction="sum").item()
            )
            probability_sum += float(probability.sum().item())
            probability_min = min(probability_min, batch_probability_min)
            probability_max = max(probability_max, batch_probability_max)
            target_min = min(target_min, float(target.min().item()))
            target_max = max(target_max, float(target.max().item()))

            if saved < save_samples:
                batch_start = sample_count - batch_size
                saved += save_mask_images(
                    output_dir,
                    data_loader.dataset,
                    batch_start,
                    probability.detach().cpu(),
                    prediction_binary.detach().cpu(),
                    target_binary.detach().cpu(),
                    save_samples - saved,
                )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if sample_count == 0 or pixel_count == 0:
        raise RuntimeError("The test data loader produced no samples.")
    if tp + fp + fn + tn != pixel_count:
        raise RuntimeError("Internal confusion-matrix count does not match pixel count.")

    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    accuracy = safe_ratio(tp + tn, pixel_count)
    balanced_accuracy = (
        (recall + specificity) / 2.0
        if math.isfinite(recall) and math.isfinite(specificity)
        else float("nan")
    )
    f1 = safe_ratio(2 * tp, 2 * tp + fp + fn)
    iou = safe_ratio(tp, tp + fp + fn)

    return {
        "samples": sample_count,
        "pixels": pixel_count,
        "threshold": threshold,
        "normalization_input_channel_min": channel_min.tolist(),
        "normalization_input_channel_max": channel_max.tolist(),
        "target_value_min": target_min,
        "target_value_max": target_max,
        "positive_target_samples": positive_target_samples,
        "positive_prediction_samples": positive_prediction_samples,
        "positive_target_pixels": tp + fn,
        "positive_prediction_pixels": tp + fp,
        "probability_min": probability_min,
        "probability_max": probability_max,
        "probability_mean": probability_sum / pixel_count,
        "bce_loss": loss_sum / pixel_count,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "F1": f1,
        "IoU": iou,
        "elapsed_seconds": elapsed,
        "samples_per_second": sample_count / elapsed if elapsed else float("inf"),
        "saved_visualizations": saved,
    }


def format_metric(value: float) -> str:
    return "undefined" if not math.isfinite(value) else f"{value:.6f}"


def json_compatible(value: Any) -> Any:
    """Replace non-finite floats with null-compatible ``None`` recursively."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_compatible(item) for item in value]
    return value


def evaluate_trained_model(model: torch.nn.Module, cfg: EasyConfig) -> dict[str, Any]:
    """Finish MVUNet training without the legacy test.py task-name registry."""
    loader = build_dataloader_from_cfg(
        cfg.batch_size, cfg.dataset, cfg.dataloader,
        datatransforms_cfg=cfg.datatransforms, split='test',
    )
    output_dir = Path(cfg.save_model_path) / cfg.run_name / 'test'
    output_dir.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    try:
        report = evaluate(
            model, loader, torch.device(cfg.device), threshold=0.5,
            limit=None, save_samples=int(cfg.get('test_save_samples', 10)),
            output_dir=output_dir,
        )
    finally:
        model.train(was_training)
    report['dataset'] = str(cfg.dataset.data_dir)
    report['legacy_xy_layout'] = bool(cfg.dataset.get('legacy_xy_layout', False))
    report['normalization'] = {
        name: float(cfg.datatransforms.kwargs[name]) for name in ('minV', 'maxV')
    }
    report_path = output_dir / 'metrics.json'
    report_path.write_text(json.dumps(json_compatible(report), indent=2, allow_nan=False) + '\n')
    print('MVUNet test: samples=%d, pixels=%d, BCE=%.6f, IoU=%.6f, Dice=%.6f, recall=%.6f' % (
        report['samples'], report['pixels'], report['bce_loss'],
        report['IoU'], report['F1'], report['recall']))
    print(f'MVUNet test report: {report_path}')
    return report


def print_report(
    report: dict[str, Any],
    cfg: EasyConfig,
    metadata: dict,
    device: torch.device,
    checkpoint: dict,
    normalized: bool,
) -> None:
    print("\nMVUnet test report")
    print(f"  checkpoint: {cfg.model_path}")
    if "epoch" in checkpoint:
        print(f"  checkpoint epoch: {checkpoint['epoch']}")
    print(f"  dataset: {cfg.dataset.data_dir}")
    print(f"  grid: {metadata.get('Xdim')} x {metadata.get('Ydim')}")
    print(f"  device: {device}")
    print(f"  normalized: {normalized}")
    print(f"  samples/pixels: {report['samples']} / {report['pixels']}")
    print(
        "  input channel ranges: "
        f"min={report['normalization_input_channel_min']}, "
        f"max={report['normalization_input_channel_max']}"
    )
    print(
        "  target range: "
        f"[{report['target_value_min']:.6g}, {report['target_value_max']:.6g}]"
    )
    print(
        "  positive samples (target/prediction): "
        f"{report['positive_target_samples']} / "
        f"{report['positive_prediction_samples']}"
    )
    print(
        "  positive pixels (target/prediction): "
        f"{report['positive_target_pixels']} / "
        f"{report['positive_prediction_pixels']}"
    )
    print(
        "  prediction probability (min/mean/max): "
        f"{report['probability_min']:.6f} / "
        f"{report['probability_mean']:.6f} / "
        f"{report['probability_max']:.6f}"
    )
    print(f"  BCE loss: {report['bce_loss']:.8f}")
    print(
        "  confusion matrix: "
        f"TP={report['TP']}, FP={report['FP']}, "
        f"FN={report['FN']}, TN={report['TN']}"
    )
    for key in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "F1",
        "IoU",
    ):
        print(f"  {key}: {format_metric(report[key])}")
    print(
        f"  runtime: {report['elapsed_seconds']:.3f}s "
        f"({report['samples_per_second']:.2f} samples/s)"
    )
    if report["positive_target_pixels"] == 0:
        print("  WARNING: the evaluated targets contain no positive pixels.")


def main() -> None:
    args = parse_args()
    cfg, metadata, device = configure_test(args)
    normalized = not args.no_normalize

    if normalized:
        print(
            "Test normalization: MinMaxNormalization "
            f"(minV={cfg.datatransforms.kwargs.minV}, "
            f"maxV={cfg.datatransforms.kwargs.maxV})"
        )
    else:
        print("Test normalization: disabled")

    data_loader = build_dataloader_from_cfg(
        cfg.batch_size,
        cfg.dataset,
        cfg.dataloader,
        datatransforms_cfg=cfg.datatransforms,
        split="test",
    )
    print(f"Loaded test samples: {len(data_loader.dataset)}")

    model, checkpoint = load_model(cfg, device)
    print("Checkpoint loaded with strict=True")
    report = evaluate(
        model=model,
        data_loader=data_loader,
        device=device,
        threshold=args.threshold,
        limit=args.limit,
        save_samples=args.save_samples,
        output_dir=args.output_dir.expanduser().resolve(),
    )
    print_report(report, cfg, metadata, device, checkpoint, normalized)

    report_path = args.report
    if report_path is not None:
        report_path = report_path.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(json_compatible(report), handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(f"JSON report saved to: {report_path}")
    if report["saved_visualizations"]:
        print(
            f"Saved {report['saved_visualizations']} visualization triplets to: "
            f"{args.output_dir.expanduser().resolve()}"
        )


if __name__ == "__main__":
    main()
