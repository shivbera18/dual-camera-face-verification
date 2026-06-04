"""Evaluation helpers for the deepfake classifier."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.utils.config import resolve
from src.utils.logger import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module,
    dataloader: DataLoader,
    threshold: float = 0.5,
    device: str | torch.device = "cpu",
) -> dict:
    model.eval()
    probs: list[float] = []
    labels: list[int] = []
    for x, y in dataloader:
        x = x.to(device)
        p = model(x).detach().cpu().numpy()
        probs.extend(p.tolist())
        labels.extend(y.numpy().tolist())
    if not probs:
        raise RuntimeError("evaluation dataloader is empty")
    preds = (np.array(probs) >= threshold).astype(int)
    y_true = np.array(labels)
    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
        "threshold": float(threshold),
        "num_samples": int(len(y_true)),
    }
    return metrics


def plot_confusion_matrix(cm: list[list[int]], save_path: str | Path) -> None:
    save_path = resolve(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["real", "fake"])
    ax.set_yticklabels(["real", "fake"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_roc(
    y_true: np.ndarray,
    probs: np.ndarray,
    save_path: str | Path,
    title: str = "ROC",
) -> float:
    save_path = resolve(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = float(roc_auc_score(y_true, probs))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return auc


def save_report(metrics: dict, save_path: str | Path) -> None:
    import json
    save_path = resolve(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("report saved: %s", save_path)
