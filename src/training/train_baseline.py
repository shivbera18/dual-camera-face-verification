"""EfficientNet-B0 baseline trainer with 2-stage fine-tuning."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.efficientnet import build_classifier
from src.training.augmentation import get_train_transform, get_val_transform
from src.training.dataset import DeepfakeDataset
from src.training.evaluate import (
    evaluate_classifier,
    plot_confusion_matrix,
    plot_roc,
    save_report,
)
from src.utils.config import get_dataset_config, get_model_config, resolve
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _make_loaders(manifest: str, batch_size: int, num_workers: int) -> dict[str, DataLoader]:
    train_ds = DeepfakeDataset(manifest, "train", transform=get_train_transform())
    val_ds = DeepfakeDataset(manifest, "val", transform=get_val_transform())
    test_ds = DeepfakeDataset(manifest, "test", transform=get_val_transform())
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return {
        "train": DataLoader(train_ds, shuffle=True, drop_last=True, **common),
        "val": DataLoader(val_ds, shuffle=False, **common),
        "test": DataLoader(test_ds, shuffle=False, **common),
    }


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.float().to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def train_baseline(args: argparse.Namespace) -> None:
    cfg_model = get_model_config()["efficientnet"]
    cfg_ds = get_dataset_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = cfg_ds["splits"]["manifest"]
    loaders = _make_loaders(manifest, args.batch_size, args.num_workers)
    model = build_classifier(
        pretrained=True,
        freeze_backbone=True,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr_head,
    )
    best_auc = -1.0
    best_state: Optional[dict] = None
    patience = cfg_model["early_stopping_patience"]
    bad_epochs = 0
    ckpt_path = resolve(cfg_model["checkpoint"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("device=%s train_batches=%d val_batches=%d", device, len(loaders["train"]), len(loaders["val"]))
    logger.info("Stage 1: head only, %d epochs lr=%.1e", args.epochs_head, args.lr_head)
    for epoch in range(args.epochs_head):
        t0 = time.time()
        train_loss = _train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_metrics = evaluate_classifier(model, loaders["val"], threshold=0.5, device=device)
        elapsed = time.time() - t0
        logger.info(
            "stage1 epoch=%d loss=%.4f val_auc=%.4f val_f1=%.4f t=%.1fs",
            epoch + 1, train_loss, val_metrics["roc_auc"], val_metrics["f1"], elapsed,
        )
        if val_metrics["roc_auc"] > best_auc:
            best_auc = val_metrics["roc_auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    logger.info("Stage 2: unfreeze last blocks, %d epochs lr=%.1e", args.epochs_ft, args.lr_ft)
    model.unfreeze_last_blocks(n=args.unfreeze_blocks)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr_ft,
    )
    bad_epochs = 0
    for epoch in range(args.epochs_ft):
        t0 = time.time()
        train_loss = _train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_metrics = evaluate_classifier(model, loaders["val"], threshold=0.5, device=device)
        elapsed = time.time() - t0
        logger.info(
            "stage2 epoch=%d loss=%.4f val_auc=%.4f val_f1=%.4f t=%.1fs",
            epoch + 1, train_loss, val_metrics["roc_auc"], val_metrics["f1"], elapsed,
        )
        if val_metrics["roc_auc"] > best_auc:
            best_auc = val_metrics["roc_auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    payload = {
        "epoch": args.epochs_head + args.epochs_ft,
        "val_auc": best_auc,
        "model_state": model.state_dict(),
        "config": {"input_size": cfg_model["input_size"]},
    }
    torch.save(payload, ckpt_path)
    logger.info("checkpoint saved: %s best_auc=%.4f", ckpt_path, best_auc)
    test_metrics = evaluate_classifier(model, loaders["test"], threshold=0.5, device=device)
    plot_confusion_matrix(test_metrics["confusion_matrix"], "artifacts/metrics/baseline_confusion_matrix.png")
    y_true_list = []
    probs_list = []
    model.eval()
    with torch.no_grad():
        for x, y in loaders["test"]:
            x = x.to(device)
            p = model.predict_proba(x).detach().cpu().numpy()
            probs_list.extend(p.tolist())
            y_true_list.extend(y.numpy().tolist())
    import numpy as np
    plot_roc(
        np.array(y_true_list),
        np.array(probs_list),
        "artifacts/metrics/baseline_roc.png",
        title="Baseline ROC",
    )
    report = {
        "best_val_auc": best_auc,
        "test": test_metrics,
        "checkpoint": str(ckpt_path),
    }
    save_report(report, "artifacts/metrics/baseline_report.json")
    logger.info("DONE test_auc=%.4f test_f1=%.4f", test_metrics["roc_auc"], test_metrics["f1"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--epochs_head", type=int, default=5)
    p.add_argument("--epochs_ft", type=int, default=15)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_ft", type=float, default=1e-4)
    p.add_argument("--unfreeze_blocks", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    train_baseline(parse_args())
