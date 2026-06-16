from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config import resolve_project_path


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def compute_binary_metrics(
    labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    labels = labels.astype(int)
    probs = probs.astype(float)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    fnr = fn / (fn + tp + 1e-12)
    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "support_real": int((labels == 0).sum()),
        "support_fake": int((labels == 1).sum()),
    }
    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
        metrics["pr_auc"] = float(average_precision_score(labels, probs))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


def sweep_thresholds(
    labels: np.ndarray, probs: np.ndarray, thresholds: np.ndarray | None = None
) -> list[dict[str, Any]]:
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    return [compute_binary_metrics(labels, probs, float(t)) for t in thresholds]


def find_best_threshold(
    labels: np.ndarray, probs: np.ndarray, metric: str = "f1"
) -> dict[str, Any]:
    rows = sweep_thresholds(labels, probs)
    return max(rows, key=lambda row: float(row.get(metric) or 0.0))


def plot_confusion_matrix(
    labels: np.ndarray, probs: np.ndarray, threshold: float, save_path: str | Path
) -> None:
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels.astype(int), preds, labels=[0, 1])
    path = resolve_project_path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks([0, 1], labels=["pred_real", "pred_fake"])
    ax.set_yticks([0, 1], labels=["true_real", "true_fake"])
    ax.set_title(f"Confusion matrix @ {threshold:.2f}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_roc_curve(
    labels: np.ndarray, probs: np.ndarray, save_path: str | Path
) -> None:
    path = resolve_project_path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        ax.plot(fpr, tpr, label=f"ROC-AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pr_curve(labels: np.ndarray, probs: np.ndarray, save_path: str | Path) -> None:
    path = resolve_project_path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    if len(np.unique(labels)) > 1:
        precision, recall, _ = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)
        ax.plot(recall, precision, label=f"PR-AUC/AP={ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_report(report: dict[str, Any], save_path: str | Path) -> None:
    path = resolve_project_path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module, dataloader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    model.eval()
    labels: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    batch_latencies: list[float] = []
    total_images = 0
    for images, y in tqdm(dataloader, desc="predict", leave=False):
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        batch_latencies.append(elapsed_ms / max(1, images.shape[0]))
        total_images += int(images.shape[0])
        logits.append(out.detach().float().cpu().numpy())
        labels.append(y.detach().float().cpu().numpy())
    all_logits = np.concatenate(logits)
    all_labels = np.concatenate(labels)
    probs = sigmoid_np(all_logits)
    latency = {
        "samples": float(total_images),
        "latency_ms_mean_per_image": float(np.mean(batch_latencies))
        if batch_latencies
        else 0.0,
        "latency_ms_p50_per_image": float(np.percentile(batch_latencies, 50))
        if batch_latencies
        else 0.0,
        "latency_ms_p95_per_image": float(np.percentile(batch_latencies, 95))
        if batch_latencies
        else 0.0,
    }
    return all_labels, probs, latency


def evaluate_classifier(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    artifact_prefix: str = "artifacts/metrics/deepfake",
) -> dict[str, Any]:
    labels, probs, latency = predict_loader(model, dataloader, device)
    metrics = compute_binary_metrics(labels, probs, threshold)
    best_f1 = find_best_threshold(labels, probs, metric="f1")
    report = {
        "metrics_at_threshold": metrics,
        "best_f1_threshold": best_f1,
        "latency": latency,
    }
    plot_confusion_matrix(
        labels, probs, threshold, f"{artifact_prefix}_confusion_matrix.png"
    )
    plot_roc_curve(labels, probs, f"{artifact_prefix}_roc.png")
    plot_pr_curve(labels, probs, f"{artifact_prefix}_pr.png")
    save_report(report, f"{artifact_prefix}_report.json")
    return report
