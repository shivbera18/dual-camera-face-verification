"""Full EfficientNet evaluation report (baseline or LoRA)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.efficientnet import build_classifier
from src.models.lora import build_lora_classifier
from src.training.augmentation import get_val_transform
from src.training.dataset import DeepfakeDataset
from src.training.evaluate import (
    evaluate_classifier,
    plot_confusion_matrix,
    plot_roc,
    save_report,
)
from src.utils.config import get_dataset_config, resolve
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _make_loader(manifest: str, batch_size: int, num_workers: int) -> DataLoader:
    ds = DeepfakeDataset(manifest, "test", transform=get_val_transform())
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


@torch.no_grad()
def _collect_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[float] = []
    labels: list[int] = []
    for x, y in loader:
        x = x.to(device)
        p = model.predict_proba(x).detach().cpu().numpy()
        probs.extend(p.tolist())
        labels.extend(y.numpy().tolist())
    return np.array(probs), np.array(labels)


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = args.manifest or get_dataset_config()["splits"]["manifest"]
    loader = _make_loader(manifest, args.batch_size, args.num_workers)
    if args.model == "lora":
        model = build_lora_classifier(
            checkpoint=args.checkpoint,
            rank=args.rank,
            alpha=args.alpha,
            device=device,
        )
    else:
        model = build_classifier(
            checkpoint=args.checkpoint,
            pretrained=False,
            freeze_backbone=True,
            device=device,
        )
    metrics = evaluate_classifier(model, loader, threshold=args.threshold, device=device)
    probs, y = _collect_probs(model, loader, device)
    plot_confusion_matrix(metrics["confusion_matrix"], f"artifacts/metrics/{args.tag}_confusion_matrix.png")
    plot_roc(y, probs, f"artifacts/metrics/{args.tag}_roc.png", title=f"{args.tag} ROC")
    report = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "metrics": metrics,
    }
    save_report(report, f"artifacts/metrics/{args.tag}_report.json")
    logger.info("eval done tag=%s auc=%.4f f1=%.4f", args.tag, metrics["roc_auc"], metrics["f1"])
    print(f"{args.tag}: AUC={metrics['roc_auc']:.4f} F1={metrics['f1']:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="baseline", choices=["baseline", "lora"])
    p.add_argument("--checkpoint", type=str, default="artifacts/models/efficientnet_b0_baseline_best.pth")
    p.add_argument("--tag", type=str, default="eval")
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
