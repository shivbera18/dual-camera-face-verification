from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.efficientnet import DeepfakeClassifier
from src.training.augmentation import get_train_transform, get_val_transform
from src.training.dataset import DeepfakeDataset, make_weighted_sampler
from src.training.evaluate import (
    compute_binary_metrics,
    evaluate_classifier,
    save_report,
    sigmoid_np,
)
from src.training.preprocess import prepare_deepfake_data
from src.utils.config import (
    get_dataset_config,
    get_model_config,
    load_all_configs,
    resolve_project_path,
)
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = -float("inf")

    def __call__(self, metric: float) -> bool:
        if metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(
    dataset: DeepfakeDataset, batch_size: int, train: bool, num_workers: int
) -> DataLoader:
    sampler = make_weighted_sampler(dataset) if train else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        persistent_workers=num_workers > 0,
    )


def _autocast_enabled(device: torch.device) -> bool:
    return device.type == "cuda"


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=_autocast_enabled(device),
        ):
            logits = model(images)
            loss = criterion(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(loss.detach().cpu()))
        logits_all.append(logits.detach().float().cpu().numpy())
        labels_all.append(labels.detach().float().cpu().numpy())
    probs = sigmoid_np(np.concatenate(logits_all))
    y = np.concatenate(labels_all)
    metrics = compute_binary_metrics(y, probs, threshold=0.5)
    return {
        "loss": float(np.mean(losses)),
        "auc": float(metrics.get("roc_auc") or 0.0),
        "f1": float(metrics["f1"]),
    }


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    for images, labels in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        losses.append(float(loss.detach().cpu()))
        logits_all.append(logits.detach().float().cpu().numpy())
        labels_all.append(labels.detach().float().cpu().numpy())
    probs = sigmoid_np(np.concatenate(logits_all))
    y = np.concatenate(labels_all)
    metrics = compute_binary_metrics(y, probs, threshold=0.5)
    return {
        "loss": float(np.mean(losses)),
        "auc": float(metrics.get("roc_auc") or 0.0),
        "f1": float(metrics["f1"]),
        "accuracy": float(metrics["accuracy"]),
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    val_metrics: dict[str, float],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict()
            if scheduler is not None
            else None,
            "val_auc": val_metrics["auc"],
            "val_loss": val_metrics["loss"],
            "val_metrics": val_metrics,
            "config": config,
        },
        path,
    )


def plot_history(history: list[dict[str, Any]], save_path: str | Path) -> None:
    path = resolve_project_path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(df["epoch"], df["train_auc"], label="train_auc")
    axes[1].plot(df["epoch"], df["val_auc"], label="val_auc")
    axes[1].plot(df["epoch"], df["val_f1"], label="val_f1")
    axes[1].set_title("Metrics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fit(args: argparse.Namespace) -> dict[str, Any]:
    dataset_cfg = get_dataset_config()
    model_cfg = get_model_config()["efficientnet"]
    manifest = resolve_project_path(args.manifest or dataset_cfg["splits"]["manifest"])
    if not manifest.exists() and args.prepare_if_missing:
        LOGGER.info("Manifest missing, preparing dataset first: %s", manifest)
        prepare_deepfake_data(
            link_mode=args.link_mode,
            verify_images=False,
            include_antispoof=args.include_antispoof,
        )
    if not manifest.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest}. Run python -m src.training.preprocess first."
        )

    batch_size = int(args.batch_size or model_cfg["batch_size"])
    epochs_stage1 = int(args.epochs_stage1)
    epochs_stage2 = int(
        args.epochs_stage2
        if args.epochs_stage2 is not None
        else max(0, int(model_cfg["epochs"]) - epochs_stage1)
    )
    num_workers = int(args.num_workers)
    device = get_device(args.device)
    LOGGER.info("Training on %s", device)

    train_ds = DeepfakeDataset(
        manifest,
        "train",
        transform=get_train_transform(),
        max_samples=args.max_train_samples,
    )
    val_ds = DeepfakeDataset(
        manifest, "val", transform=get_val_transform(), max_samples=args.max_val_samples
    )
    test_ds = DeepfakeDataset(
        manifest,
        "test",
        transform=get_val_transform(),
        max_samples=args.max_test_samples,
    )
    train_loader = make_loader(
        train_ds, batch_size, train=True, num_workers=num_workers
    )
    val_loader = make_loader(val_ds, batch_size, train=False, num_workers=num_workers)
    test_loader = make_loader(test_ds, batch_size, train=False, num_workers=num_workers)

    pretrained = args.pretrained
    if pretrained is None:
        pretrained = str(model_cfg.get("pretrained", "imagenet")).lower() not in {
            "false",
            "none",
            "0",
        }
    model = DeepfakeClassifier(
        pretrained=pretrained, dropout=float(args.dropout), freeze_backbone=True
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    early = EarlyStopping(
        patience=int(args.patience or model_cfg["early_stopping_patience"])
    )
    best_auc = -float("inf")
    history: list[dict[str, Any]] = []
    checkpoint_path = resolve_project_path(args.checkpoint or model_cfg["checkpoint"])
    last_path = checkpoint_path.with_name(f"efficientnet_b0_{args.run_name}_last.pth")
    config = load_all_configs() | {"cli_args": vars(args)}

    def run_stage(
        stage_name: str,
        epochs: int,
        lr: float,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
    ) -> bool:
        nonlocal best_auc
        for local_epoch in range(1, epochs + 1):
            epoch_id = len(history) + 1
            start = time.perf_counter()
            tr = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler, scheduler
            )
            va = validate(model, val_loader, criterion, device)
            elapsed = time.perf_counter() - start
            row = {
                "epoch": epoch_id,
                "stage": stage_name,
                "lr": lr,
                "train_loss": tr["loss"],
                "train_auc": tr["auc"],
                "train_f1": tr["f1"],
                "val_loss": va["loss"],
                "val_auc": va["auc"],
                "val_f1": va["f1"],
                "val_accuracy": va["accuracy"],
                "time_s": elapsed,
            }
            history.append(row)
            LOGGER.info(
                "epoch=%d stage=%s train_loss=%.4f train_auc=%.4f val_loss=%.4f val_auc=%.4f val_f1=%.4f time=%.1fs",
                epoch_id,
                stage_name,
                tr["loss"],
                tr["auc"],
                va["loss"],
                va["auc"],
                va["f1"],
                elapsed,
            )
            save_checkpoint(
                last_path, model, optimizer, scheduler, epoch_id, va, config
            )
            if va["auc"] > best_auc:
                best_auc = va["auc"]
                save_checkpoint(
                    checkpoint_path, model, optimizer, scheduler, epoch_id, va, config
                )
            if early(va["auc"]):
                LOGGER.info("Early stopping triggered after epoch %d", epoch_id)
                return True
        return False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr_stage1),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs_stage1 * len(train_loader))
    )
    stopped = run_stage(
        "head", epochs_stage1, float(args.lr_stage1), optimizer, scheduler
    )

    if not stopped and epochs_stage2 > 0:
        model.unfreeze_last_blocks(n=int(args.unfreeze_blocks))
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=float(args.lr_stage2), weight_decay=float(args.weight_decay)
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(args.lr_stage2),
            total_steps=max(1, epochs_stage2 * len(train_loader)),
            pct_start=0.1,
        )
        run_stage(
            "finetune", epochs_stage2, float(args.lr_stage2), optimizer, scheduler
        )

    history_path = resolve_project_path(
        f"artifacts/metrics/{args.run_name}_training_history.csv"
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    plot_history(history, f"artifacts/metrics/{args.run_name}_training_curves.png")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_report = evaluate_classifier(
        model,
        test_loader,
        device,
        threshold=float(args.threshold),
        artifact_prefix=f"artifacts/metrics/{args.run_name}_test",
    )
    final_report = {
        "best_val_auc": float(best_auc),
        "checkpoint": str(checkpoint_path),
        "history_csv": str(history_path),
        "test": test_report,
    }
    save_report(final_report, f"artifacts/metrics/{args.run_name}_report.json")
    return final_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B0 deepfake classifier."
    )
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--run-name",
        default="baseline",
        help="Prefix for metric artifacts, e.g. baseline or smoke.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs-stage1", type=int, default=5)
    parser.add_argument("--epochs-stage2", type=int, default=None)
    parser.add_argument("--lr-stage1", type=float, default=1e-3)
    parser.add_argument("--lr-stage2", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--unfreeze-blocks", type=int, default=3)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=None,
        help="Force ImageNet-pretrained EfficientNet weights.",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Disable pretrained weights for offline smoke tests only.",
    )
    parser.add_argument("--prepare-if-missing", action="store_true", default=True)
    parser.add_argument(
        "--no-prepare-if-missing", dest="prepare_if_missing", action="store_false"
    )
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--include-antispoof", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    report = fit(parse_args())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
