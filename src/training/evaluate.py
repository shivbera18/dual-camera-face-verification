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
    det_curve,
)
from sklearn.calibration import calibration_curve
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


def plot_normalized_confusion_matrix(
    labels: np.ndarray, probs: np.ndarray, threshold: float, save_path: str | Path
) -> None:
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels.astype(int), preds, labels=[0, 1], normalize="true")
    path = resolve_project_path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks([0, 1], labels=["pred_real", "pred_fake"])
    ax.set_yticks([0, 1], labels=["true_real", "true_fake"])
    ax.set_title(f"Normalized CM @ {threshold:.2f}")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]*100:.1f}%", ha="center", va="center", color=color)
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


def plot_score_distribution(labels: np.ndarray, probs: np.ndarray, save_path: str | Path) -> None:
    path = resolve_project_path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    real_probs = probs[labels == 0]
    fake_probs = probs[labels == 1]
    ax.hist(real_probs, bins=50, alpha=0.6, label="Real", color="blue", density=True)
    ax.hist(fake_probs, bins=50, alpha=0.6, label="Fake", color="red", density=True)
    ax.set_xlabel("Predicted Probability (Fake)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper center")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_det_curve(labels: np.ndarray, probs: np.ndarray, save_path: str | Path) -> None:
    path = resolve_project_path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    if len(np.unique(labels)) > 1:
        fpr, fnr, _ = det_curve(labels, probs)
        ax.plot(fpr, fnr, label="DET Curve")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("False Positive Rate (FAR)")
    ax.set_ylabel("False Negative Rate (FRR)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_calibration_curve_diag(labels: np.ndarray, probs: np.ndarray, save_path: str | Path) -> None:
    path = resolve_project_path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    if len(np.unique(labels)) > 1:
        prob_true, prob_pred = calibration_curve(labels, probs, n_bins=10)
        ax.plot(prob_pred, prob_true, marker='o', label="Model Calibration")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_threshold_sweep(labels: np.ndarray, probs: np.ndarray, save_path: str | Path) -> None:
    path = resolve_project_path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sweep = sweep_thresholds(labels, probs)
    thresholds = [r["threshold"] for r in sweep]
    f1s = [r["f1"] for r in sweep]
    accs = [r["accuracy"] for r in sweep]
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, f1s, label="F1 Score", color="purple")
    ax.plot(thresholds, accs, label="Accuracy", color="green")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Metrics vs Threshold")
    ax.legend(loc="lower center")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_latency_distribution(latencies: list[float], save_path: str | Path) -> None:
    path = resolve_project_path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    if latencies:
        ax.hist(latencies, bins=30, color="orange", alpha=0.7, edgecolor="black")
        mean_lat = float(np.mean(latencies))
        p95_lat = float(np.percentile(latencies, 95))
        ax.axvline(mean_lat, color='red', linestyle='dashed', linewidth=1.5, label=f"Mean: {mean_lat:.2f}ms")
        ax.axvline(p95_lat, color='blue', linestyle='dashed', linewidth=1.5, label=f"P95: {p95_lat:.2f}ms")
    ax.set_xlabel("Inference Latency per Image (ms)")
    ax.set_ylabel("Frequency (Batches)")
    ax.set_title("Inference Latency Distribution")
    ax.legend()
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
        "raw_latencies": batch_latencies,
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
    plot_normalized_confusion_matrix(
        labels, probs, threshold, f"{artifact_prefix}_confusion_matrix_pct.png"
    )
    plot_roc_curve(labels, probs, f"{artifact_prefix}_roc.png")
    plot_pr_curve(labels, probs, f"{artifact_prefix}_pr.png")
    plot_score_distribution(labels, probs, f"{artifact_prefix}_score_dist.png")
    plot_det_curve(labels, probs, f"{artifact_prefix}_det.png")
    plot_calibration_curve_diag(labels, probs, f"{artifact_prefix}_calibration.png")
    plot_threshold_sweep(labels, probs, f"{artifact_prefix}_threshold_sweep.png")
    
    raw_lat = latency.pop("raw_latencies", [])
    plot_latency_distribution(raw_lat, f"{artifact_prefix}_latency_dist.png")
    
    save_report(report, f"{artifact_prefix}_report.json")
    return report
