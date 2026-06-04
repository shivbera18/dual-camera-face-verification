"""LFW-based threshold sweep for the ArcFace verifier."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from src.face.aligner import align_face
from src.face.detector import FaceDetector
from src.models.arcface import ArcFaceExtractor
from src.utils.config import get_dataset_config, resolve
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _parse_pairs(pairs_file: Path) -> list[tuple[tuple[str, str], tuple[str, str], int]]:
    """Returns (left, right, is_same)."""
    out = []
    with pairs_file.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) == 3:
                name, idx1, idx2 = parts
                out.append(((f"{name}/{idx1}.jpg", name), (f"{name}/{idx2}.jpg", name), 1))
            elif len(parts) == 4:
                n1, i1, n2, i2 = parts
                out.append(((f"{n1}/{i1}.jpg", n1), (f"{n2}/{i2}.jpg", n2), 0))
    return out


def _img_path(lfw_root: Path, identity_name: str, file_name: str) -> Path:
    return lfw_root / identity_name / file_name


def _extract_embedding(
    detector: FaceDetector,
    extractor: ArcFaceExtractor,
    img_path: Path,
) -> Optional[np.ndarray]:
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    best = detector.detect_best(img)
    if best is None:
        return None
    try:
        crop = align_face(img, best.landmarks, output_size=(112, 112))
    except Exception:
        return None
    return extractor.get_embedding(crop)


def _sweep_metrics(sims: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> list[dict]:
    rows = []
    for t in thresholds:
        preds = (sims >= t).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        far = fp / max(fp + tn, 1)
        frr = fn / max(fn + tp, 1)
        acc = (tp + tn) / max(tp + tn + fp + fn, 1)
        rows.append({"threshold": float(t), "tp": tp, "tn": tn, "fp": fp, "fn": fn, "far": far, "frr": frr, "accuracy": acc})
    return rows


def _eer(rows: list[dict]) -> tuple[float, float]:
    best = None
    for r in rows:
        diff = abs(r["far"] - r["frr"])
        if best is None or diff < best[1]:
            best = (r["threshold"], diff)
    return best if best is not None else (0.0, 1.0)


def run(args: argparse.Namespace) -> None:
    ds = get_dataset_config()
    lfw_root = resolve(ds["raw"]["lfw"])
    pairs_file = resolve(ds["raw"]["lfw_pairs"])
    if not lfw_root.exists():
        raise FileNotFoundError(f"lfw dir missing: {lfw_root}")
    if not pairs_file.exists():
        raise FileNotFoundError(f"pairs file missing: {pairs_file}")
    pairs = _parse_pairs(pairs_file)
    logger.info("loaded %d pairs", len(pairs))
    detector = FaceDetector()
    extractor = ArcFaceExtractor()
    cache: dict[str, Optional[np.ndarray]] = {}
    sims: list[float] = []
    labels: list[int] = []
    t0 = time.time()
    for left, right, is_same in tqdm(pairs, desc="lfw"):
        l_id, l_file = left[1], left[0]
        r_id, r_file = right[1], right[0]
        l_key = f"{l_id}/{l_file}"
        r_key = f"{r_id}/{r_file}"
        if l_key not in cache:
            cache[l_key] = _extract_embedding(detector, extractor, _img_path(lfw_root, l_id, l_file))
        if r_key not in cache:
            cache[r_key] = _extract_embedding(detector, extractor, _img_path(lfw_root, r_id, r_file))
        le, re_ = cache[l_key], cache[r_key]
        if le is None or re_ is None:
            continue
        sims.append(ArcFaceExtractor.similarity(le, re_))
        labels.append(is_same)
    logger.info("computed %d/%d pair similarities in %.1fs", len(sims), len(pairs), time.time() - t0)
    sims_arr = np.array(sims)
    labels_arr = np.array(labels)
    thresholds = np.arange(0.30, 0.90, 0.01)
    rows = _sweep_metrics(sims_arr, labels_arr, thresholds)
    eer_t, _ = _eer(rows)
    out = {
        "num_pairs_total": len(pairs),
        "num_pairs_used": int(len(sims)),
        "eer_threshold": float(eer_t),
        "rows": rows,
    }
    save_path = resolve("artifacts/metrics/arcface_threshold_report.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w") as f:
        json.dump(out, f, indent=2)
    logger.info("threshold report saved: %s eer=%.3f", save_path, eer_t)
    print(f"EER threshold: {eer_t:.3f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
