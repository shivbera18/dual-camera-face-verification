from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from src.models.efficientnet import load_deepfake_checkpoint
from src.models.lora import count_trainable_params
from src.training.augmentation import get_val_transform
from src.training.dataset import DeepfakeDataset
from src.training.evaluate import evaluate_classifier
from src.training.train_baseline import get_device
from src.training.train_lora import load_lora_model
from src.utils.config import get_dataset_config, get_model_config, resolve_project_path


def metric(report: dict, name: str) -> float | None:
    return report.get("metrics_at_threshold", {}).get(name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline and LoRA checkpoints on test split."
    )
    parser.add_argument("--baseline", default=None)
    parser.add_argument(
        "--lora", default="artifacts/models/efficientnet_b0_lora_best.pth"
    )
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = get_dataset_config()
    model_cfg = get_model_config()["efficientnet"]
    manifest = resolve_project_path(args.manifest or cfg["splits"]["manifest"])
    device = get_device(args.device)
    ds = DeepfakeDataset(
        manifest, "test", get_val_transform(), max_samples=args.max_samples
    )
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size or model_cfg["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    baseline, _ = load_deepfake_checkpoint(
        resolve_project_path(args.baseline or model_cfg["checkpoint"]),
        device=device,
        pretrained=False,
    )
    lora, _ = load_lora_model(args.lora, device)
    baseline_report = evaluate_classifier(
        baseline, loader, device, artifact_prefix="artifacts/metrics/compare_baseline"
    )
    lora_report = evaluate_classifier(
        lora, loader, device, artifact_prefix="artifacts/metrics/compare_lora"
    )

    rows = [
        (
            "Trainable params",
            baseline.count_parameters(True),
            count_trainable_params(lora),
        ),
        ("ROC-AUC", metric(baseline_report, "roc_auc"), metric(lora_report, "roc_auc")),
        (
            "Accuracy",
            metric(baseline_report, "accuracy"),
            metric(lora_report, "accuracy"),
        ),
        (
            "Precision",
            metric(baseline_report, "precision"),
            metric(lora_report, "precision"),
        ),
        ("Recall", metric(baseline_report, "recall"), metric(lora_report, "recall")),
        ("F1", metric(baseline_report, "f1"), metric(lora_report, "f1")),
        (
            "False positive rate",
            metric(baseline_report, "false_positive_rate"),
            metric(lora_report, "false_positive_rate"),
        ),
        (
            "False negative rate",
            metric(baseline_report, "false_negative_rate"),
            metric(lora_report, "false_negative_rate"),
        ),
        (
            "Inference ms mean",
            baseline_report["latency"]["latency_ms_mean_per_image"],
            lora_report["latency"]["latency_ms_mean_per_image"],
        ),
    ]
    lines = [
        "# Baseline vs LoRA",
        "",
        "| Metric | Baseline | LoRA |",
        "|---|---:|---:|",
    ]
    for name, base_value, lora_value in rows:
        lines.append(f"| {name} | {base_value} | {lora_value} |")
    out = resolve_project_path("artifacts/metrics/baseline_vs_lora.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
