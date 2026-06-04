"""Convert raw datasets into the canonical processed/deepfake_faces/ layout."""
from __future__ import annotations

import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
from tqdm import tqdm

from src.face.aligner import align_face
from src.face.detector import FaceDetector
from src.utils.config import get_dataset_config, get_model_config, resolve
from src.utils.logger import get_logger

logger = get_logger(__name__)

LABEL_REAL = 0
LABEL_FAKE = 1


@dataclass
class CopyStats:
    copied: int = 0
    skipped_missing: int = 0
    skipped_small: int = 0
    skipped_unreadable: int = 0

    def total_in(self) -> int:
        return self.copied + self.skipped_missing + self.skipped_small + self.skipped_unreadable


def _ensure_split_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {}
    for split in ("train", "val", "test"):
        for label_name in ("real", "fake"):
            d = output_dir / split / label_name
            d.mkdir(parents=True, exist_ok=True)
            dirs[f"{split}_{label_name}"] = d
    return dirs


def _assign_split(
    rng: random.Random,
    ratios: tuple[float, float, float],
) -> str:
    r = rng.random()
    if r < ratios[0]:
        return "train"
    if r < ratios[0] + ratios[1]:
        return "val"
    return "test"


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _copy_valid(src: Path, dst: Path, min_side: int) -> str:
    img = cv2.imread(str(src))
    if img is None:
        return "unreadable"
    h, w = img.shape[:2]
    if min(h, w) < min_side:
        return "small"
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img)
    return "ok"


def process_ffpp(
    metadata_csv: str | Path,
    faces_dir: str | Path,
    output_dir: str | Path,
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    min_side: int = 64,
) -> CopyStats:
    metadata_csv = resolve(metadata_csv)
    faces_dir = resolve(faces_dir)
    output_dir = resolve(output_dir)
    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_csv}")
    if not faces_dir.exists():
        raise FileNotFoundError(f"faces_dir not found: {faces_dir}")
    _ensure_split_dirs(output_dir)
    rng = random.Random(seed)
    stats = CopyStats()
    with metadata_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="ffpp"):
            label_raw = row.get("label") or row.get("LABEL") or ""
            label_norm = label_raw.strip().upper()
            if label_norm in {"REAL", "0"}:
                label_name = "real"
            elif label_norm in {"FAKE", "1"}:
                label_name = "fake"
            else:
                stats.skipped_missing += 1
                continue
            rel_path = (
                row.get("path")
                or row.get("image_path")
                or row.get("filename")
                or ""
            )
            if not rel_path:
                stats.skipped_missing += 1
                continue
            src = faces_dir / rel_path
            if not src.exists():
                src = faces_dir / Path(rel_path).name
            if not src.exists():
                stats.skipped_missing += 1
                continue
            split = _assign_split(rng, split_ratios)
            dst = output_dir / split / label_name / src.name
            outcome = _copy_valid(src, dst, min_side=min_side)
            if outcome == "ok":
                stats.copied += 1
            elif outcome == "small":
                stats.skipped_small += 1
            else:
                stats.skipped_unreadable += 1
    logger.info("ffpp done: %s", stats)
    return stats


def process_140k(
    base_dir: str | Path,
    output_dir: str | Path,
    min_side: int = 64,
) -> CopyStats:
    base_dir = resolve(base_dir)
    output_dir = resolve(output_dir)
    _ensure_split_dirs(output_dir)
    stats = CopyStats()
    mapping = [
        ("train/real", "train/real"),
        ("train/fake", "train/fake"),
        ("test/real", "test/real"),
        ("test/fake", "test/fake"),
        ("val/real", "val/real"),
        ("val/fake", "val/fake"),
    ]
    for src_rel, dst_rel in mapping:
        src_root = base_dir / src_rel
        dst_root = output_dir / dst_rel
        if not src_root.exists():
            logger.warning("missing source: %s", src_root)
            continue
        for img_path in tqdm(list(src_root.rglob("*")), desc=f"140k:{src_rel}"):
            if not img_path.is_file() or not _is_image(img_path):
                continue
            dst = dst_root / img_path.name
            outcome = _copy_valid(img_path, dst, min_side=min_side)
            if outcome == "ok":
                stats.copied += 1
            elif outcome == "small":
                stats.skipped_small += 1
            else:
                stats.skipped_unreadable += 1
    logger.info("140k done: %s", stats)
    return stats


def process_ciplab(
    base_dir: str | Path,
    output_dir: str | Path,
    split: str = "val",
    min_side: int = 64,
) -> CopyStats:
    base_dir = resolve(base_dir)
    output_dir = resolve(output_dir)
    _ensure_split_dirs(output_dir)
    stats = CopyStats()
    mapping = [
        ("training_real", "real"),
        ("training_fake", "fake"),
    ]
    for src_rel, label_name in mapping:
        src_root = base_dir / src_rel
        dst_root = output_dir / split / label_name
        if not src_root.exists():
            logger.warning("missing source: %s", src_root)
            continue
        for img_path in tqdm(list(src_root.rglob("*")), desc=f"ciplab:{src_rel}"):
            if not img_path.is_file() or not _is_image(img_path):
                continue
            dst = dst_root / img_path.name
            outcome = _copy_valid(img_path, dst, min_side=min_side)
            if outcome == "ok":
                stats.copied += 1
            elif outcome == "small":
                stats.skipped_small += 1
            else:
                stats.skipped_unreadable += 1
    logger.info("ciplab done: %s", stats)
    return stats


def run_retinaface_on_dir(
    img_dir: str | Path,
    output_dir: str | Path,
    detector: Optional[FaceDetector] = None,
    output_size: tuple[int, int] = (224, 224),
    min_side: int = 80,
) -> CopyStats:
    img_dir = resolve(img_dir)
    output_dir = resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if detector is None:
        detector = FaceDetector()
    stats = CopyStats()
    images = [p for p in img_dir.rglob("*") if p.is_file() and _is_image(p)]
    for img_path in tqdm(images, desc="retinaface"):
        img = cv2.imread(str(img_path))
        if img is None:
            stats.skipped_unreadable += 1
            continue
        best = detector.detect_best(img)
        if best is None:
            stats.skipped_missing += 1
            continue
        try:
            crop = align_face(img, best.landmarks, output_size=output_size)
        except Exception:
            stats.skipped_unreadable += 1
            continue
        if min(crop.shape[:2]) < min_side:
            stats.skipped_small += 1
            continue
        dst = output_dir / img_path.name
        cv2.imwrite(str(dst), crop)
        stats.copied += 1
    logger.info("retinaface done: %s", stats)
    return stats


def main() -> None:
    ds = get_dataset_config()
    processed = resolve(ds["processed"]["deepfake_faces"])
    raw = ds["raw"]
    process_ffpp(
        metadata_csv=resolve(raw["faceforensicspp"]) / "metadata.csv",
        faces_dir=resolve(raw["faceforensicspp"]) / "faces_224",
        output_dir=processed,
        split_ratios=(
            ds["splits"]["train_ratio"],
            ds["splits"]["val_ratio"],
            ds["splits"]["test_ratio"],
        ),
    )
    process_140k(
        base_dir=resolve(raw["celebdf"]) / "real_vs_fake/real-vs-fake",
        output_dir=processed,
    )
    process_ciplab(
        base_dir=resolve(raw["celebdf"]) / "real_and_fake_face",
        output_dir=processed,
    )


if __name__ == "__main__":
    main()
