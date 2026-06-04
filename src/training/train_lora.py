"""LoRA / adapter fine-tuning loop."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.efficientnet import build_classifier
from src.models.lora import count_trainable_params, inject_lora
from src.training.augmentation import get_train_transform, get_val_transform
from src.training.dataset import DeepfakeDataset
from src.training.evaluate import evaluate_classifier, save_report
from src.utils.config import get_dataset_config, resolve
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
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def train_lora(args: argparse.Namespace) -> None:
    cfg_ds = get_dataset_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = _make_loaders(cfg_ds["splits"]["manifest"], args.batch_size, args.num_workers)
    model = build_classifier(
        checkpoint=args.baseline_checkpoint,
        pretrained=False,
        freeze_backbone=True,
        device=device,
    )
    model = inject_lora(model, rank=args.rank, alpha=args.alpha)
    trainable = count_trainable_params(model)
    logger.info("lora trainable params: %d", trainable)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    criterion = nn.BCEWithLogitsLoss()
    best_auc = -1.0
    best_state = None
    bad_epochs = 0
    ckpt_path = resolve("artifacts/models/efficientnet_b0_lora_best.pth")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = _train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_metrics = evaluate_classifier(model, loaders["val"], threshold=0.5, device=device)
        elapsed = time.time() - t0
        logger.info(
            "lora epoch=%d loss=%.4f val_auc=%.4f val_f1=%.4f t=%.1fs",
            epoch + 1, train_loss, val_metrics["roc_auc"], val_metrics["f1"], elapsed,
        )
        if val_metrics["roc_auc"] > best_auc:
            best_auc = val_metrics["roc_auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "epoch": args.epochs,
            "val_auc": best_auc,
            "model_state": model.state_dict(),
            "config": {"rank": args.rank, "alpha": args.alpha},
            "trainable_params": trainable,
        },
        ckpt_path,
    )
    test_metrics = evaluate_classifier(model, loaders["test"], threshold=0.5, device=device)
    report = {
        "best_val_auc": best_auc,
        "test": test_metrics,
        "trainable_params": trainable,
        "checkpoint": str(ckpt_path),
    }
    save_report(report, "artifacts/metrics/lora_report.json")
    logger.info("DONE lora test_auc=%.4f test_f1=%.4f", test_metrics["roc_auc"], test_metrics["f1"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_checkpoint", type=str, default="artifacts/models/efficientnet_b0_baseline_best.pth")
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    train_lora(parse_args())
