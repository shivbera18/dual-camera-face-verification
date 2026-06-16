from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from tqdm import tqdm

from src.models.arcface import ArcFaceExtractor
from src.utils.config import get_dataset_config, resolve_project_path


def lfw_image_path(lfw_root: Path, person: str, index: str) -> Path:
    return lfw_root / person / f"{person}_{int(index):04d}.jpg"


def read_lfw_pairs(
    pairs_file: str | Path, lfw_root: str | Path
) -> list[tuple[Path, Path, int]]:
    root = resolve_project_path(lfw_root)
    lines = [
        line.strip().split()
        for line in resolve_project_path(pairs_file)
        .read_text(encoding="utf-8", errors="ignore")
        .splitlines()
        if line.strip()
    ]
    pairs: list[tuple[Path, Path, int]] = []
    for parts in lines[1:]:
        if len(parts) == 3:
            person, idx_a, idx_b = parts
            pairs.append(
                (
                    lfw_image_path(root, person, idx_a),
                    lfw_image_path(root, person, idx_b),
                    1,
                )
            )
        elif len(parts) == 4:
            person_a, idx_a, person_b, idx_b = parts
            pairs.append(
                (
                    lfw_image_path(root, person_a, idx_a),
                    lfw_image_path(root, person_b, idx_b),
                    0,
                )
            )
    return pairs


def load_or_compute_embeddings(
    image_paths: list[Path],
    extractor: ArcFaceExtractor,
    cache_path: Path,
) -> dict[str, np.ndarray]:
    if cache_path.exists():
        with cache_path.open("rb") as fh:
            return pickle.load(fh)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings: dict[str, np.ndarray] = {}
    for path in tqdm(image_paths, desc="lfw embeddings"):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        emb = extractor.get_embedding(img)
        if emb is not None:
            embeddings[str(path)] = emb
    with cache_path.open("wb") as fh:
        pickle.dump(embeddings, fh)
    return embeddings


def threshold_sweep(
    labels: np.ndarray, sims: np.ndarray, thresholds: np.ndarray | None = None
) -> list[dict[str, float]]:
    if thresholds is None:
        thresholds = np.arange(0.20, 0.90, 0.01)
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        preds = (sims >= threshold).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        far = fp / (fp + tn + 1e-12)
        frr = fn / (fn + tp + 1e-12)
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float((tp + tn) / max(1, len(labels))),
                "far_false_accept_rate": float(far),
                "frr_false_reject_rate": float(frr),
                "false_accepts": float(fp),
                "false_rejects": float(fn),
                "true_accepts": float(tp),
                "true_rejects": float(tn),
            }
        )
    return rows


def evaluate_lfw(
    pairs_file: str | Path,
    lfw_root: str | Path,
    cache_path: str | Path,
    ctx_id: int = -1,
) -> dict[str, Any]:
    pairs = read_lfw_pairs(pairs_file, lfw_root)
    unique_paths = sorted({p for pair in pairs for p in pair[:2]})
    extractor = ArcFaceExtractor(ctx_id=ctx_id)
    embeddings = load_or_compute_embeddings(
        unique_paths, extractor, resolve_project_path(cache_path)
    )
    labels: list[int] = []
    sims: list[float] = []
    missing = 0
    for a, b, label in tqdm(pairs, desc="lfw pairs"):
        emb_a = embeddings.get(str(a))
        emb_b = embeddings.get(str(b))
        if emb_a is None or emb_b is None:
            missing += 1
            continue
        labels.append(label)
        sims.append(ArcFaceExtractor.similarity(emb_a, emb_b))
    y = np.asarray(labels, dtype=int)
    s = np.asarray(sims, dtype=float)
    sweep = threshold_sweep(y, s)
    fpr, tpr, thresholds = roc_curve(y, s)
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])
    best_acc = max(sweep, key=lambda row: row["accuracy"])
    default_060 = min(sweep, key=lambda row: abs(row["threshold"] - 0.60))
    report: dict[str, Any] = {
        "pairs_total": len(pairs),
        "pairs_evaluated": int(len(y)),
        "pairs_missing_embedding": int(missing),
        "roc_auc": float(roc_auc_score(y, s)),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "best_accuracy_threshold": best_acc,
        "default_threshold_0_60": default_060,
        "sweep": sweep,
    }
    return report


def plot_sweep(report: dict[str, Any], save_path: str | Path) -> None:
    path = resolve_project_path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sweep = report["sweep"]
    thresholds = [row["threshold"] for row in sweep]
    far = [row["far_false_accept_rate"] for row in sweep]
    frr = [row["frr_false_reject_rate"] for row in sweep]
    acc = [row["accuracy"] for row in sweep]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, far, label="FAR")
    ax.plot(thresholds, frr, label="FRR")
    ax.plot(thresholds, acc, label="Accuracy")
    ax.axvline(
        report["eer_threshold"],
        linestyle="--",
        color="gray",
        label=f"EER {report['eer']:.4f}",
    )
    ax.set_xlabel("Cosine threshold")
    ax.set_ylabel("Rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune ArcFace verification threshold on LFW pairs."
    )
    parser.add_argument("--lfw-root", default=None)
    parser.add_argument("--pairs-file", default=None)
    parser.add_argument(
        "--cache", default="artifacts/metrics/lfw_arcface_embeddings.pkl"
    )
    parser.add_argument("--ctx-id", type=int, default=-1)
    parser.add_argument(
        "--save", default="artifacts/metrics/arcface_threshold_report.json"
    )
    parser.add_argument(
        "--plot", default="artifacts/metrics/arcface_threshold_sweep.png"
    )
    args = parser.parse_args()
    cfg = get_dataset_config()
    report = evaluate_lfw(
        args.pairs_file or cfg["raw"]["lfw_pairs"],
        args.lfw_root or cfg["raw"]["lfw"],
        args.cache,
        ctx_id=args.ctx_id,
    )
    out = resolve_project_path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_sweep(report, args.plot)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
