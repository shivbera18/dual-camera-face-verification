from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.efficientnet import DeepfakeClassifier, load_deepfake_checkpoint
from src.models.lora import count_trainable_params, inject_lora
from src.training.augmentation import get_train_transform, get_val_transform
from src.training.dataset import DeepfakeDataset, make_weighted_sampler
from src.training.evaluate import evaluate_classifier, save_report
from src.training.train_baseline import (
    EarlyStopping,
    get_device,
    train_one_epoch,
    validate,
)
from src.utils.config import get_dataset_config, get_model_config, resolve_project_path
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def load_lora_model(
    checkpoint_path: str, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    ckpt = torch.load(resolve_project_path(checkpoint_path), map_location=device)
    base = DeepfakeClassifier(pretrained=False)
    base.load_state_dict(ckpt["base_model_state"])
    cfg = ckpt.get("lora_config", {})
    model = inject_lora(
        base,
        rank=int(cfg.get("rank", 4)),
        alpha=float(cfg.get("alpha", 1.0)),
        target_keywords=tuple(cfg.get("target_keywords", ["classifier.1"])),
        include_conv=bool(cfg.get("include_conv", False)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune EfficientNet-B0 baseline."
    )
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument(
        "--output", default="artifacts/models/efficientnet_b0_lora_best.pth"
    )
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--include-conv", action="store_true")
    parser.add_argument("--target-keywords", nargs="+", default=["classifier.1"])
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    args = parser.parse_args()

    dataset_cfg = get_dataset_config()
    model_cfg = get_model_config()["efficientnet"]
    device = get_device(args.device)
    base_checkpoint = resolve_project_path(
        args.base_checkpoint or model_cfg["checkpoint"]
    )
    base_model, base_meta = load_deepfake_checkpoint(
        base_checkpoint, device=device, pretrained=False
    )
    lora_cfg = {
        "rank": args.rank,
        "alpha": args.alpha,
        "target_keywords": args.target_keywords,
        "include_conv": args.include_conv,
    }
    model = inject_lora(
        base_model,
        rank=args.rank,
        alpha=args.alpha,
        target_keywords=tuple(args.target_keywords),
        include_conv=args.include_conv,
    ).to(device)
    LOGGER.info("LoRA trainable params: %d", count_trainable_params(model))

    manifest = resolve_project_path(args.manifest or dataset_cfg["splits"]["manifest"])
    batch_size = int(args.batch_size or model_cfg["batch_size"])
    train_ds = DeepfakeDataset(
        manifest, "train", get_train_transform(), max_samples=args.max_train_samples
    )
    val_ds = DeepfakeDataset(
        manifest, "val", get_val_transform(), max_samples=args.max_val_samples
    )
    test_ds = DeepfakeDataset(
        manifest, "test", get_val_transform(), max_samples=args.max_test_samples
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=make_weighted_sampler(train_ds),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=max(1, args.epochs * len(train_loader)),
        pct_start=0.1,
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    early = EarlyStopping(patience=3)
    best_auc = -float("inf")
    history: list[dict[str, Any]] = []
    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        tr = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, scheduler
        )
        va = validate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_auc": tr["auc"],
            "val_loss": va["loss"],
            "val_auc": va["auc"],
            "val_f1": va["f1"],
            "time_s": time.perf_counter() - start,
        }
        history.append(row)
        LOGGER.info(
            "lora epoch=%d train_auc=%.4f val_auc=%.4f val_f1=%.4f",
            epoch,
            tr["auc"],
            va["auc"],
            va["f1"],
        )
        if va["auc"] > best_auc:
            best_auc = va["auc"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "base_model_state": base_model.state_dict(),
                    "base_checkpoint": str(base_checkpoint),
                    "val_auc": va["auc"],
                    "val_metrics": va,
                    "lora_config": lora_cfg,
                },
                output,
            )
        if early(va["auc"]):
            break

    pd.DataFrame(history).to_csv(
        resolve_project_path("artifacts/metrics/lora_training_history.csv"), index=False
    )
    best_model, _ = load_lora_model(str(output), device)
    test_report = evaluate_classifier(
        best_model,
        test_loader,
        device,
        threshold=0.5,
        artifact_prefix="artifacts/metrics/lora_test",
    )
    report = {
        "best_val_auc": float(best_auc),
        "checkpoint": str(output),
        "trainable_params": count_trainable_params(best_model),
        "test": test_report,
        "history": history,
    }
    save_report(report, "artifacts/metrics/lora_report.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
