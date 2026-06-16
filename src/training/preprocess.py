from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
from tqdm import tqdm

from src.face.aligner import align_face
from src.face.detector import FaceDetector
from src.training.build_manifest import MANIFEST_COLUMNS
from src.utils.config import get_dataset_config, resolve_project_path
from src.utils.logger import get_logger

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    image_path: str
    label: int
    source_dataset: str
    original_video: str = ""
    subject_id: str = ""
    split: str = "train"
    face_confidence: float = 1.0
    crop_quality_status: str = "ok"

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "label": self.label,
            "source_dataset": self.source_dataset,
            "original_video": self.original_video,
            "subject_id": self.subject_id,
            "split": self.split,
            "face_confidence": self.face_confidence,
            "crop_quality_status": self.crop_quality_status,
        }


def stable_split(
    key: str, ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
) -> str:
    """Deterministic group split using a stable hash."""
    value = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) / float(16**32)
    if value < ratios[0]:
        return "train"
    if value < ratios[0] + ratios[1]:
        return "val"
    return "test"


def _iter_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(
        f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )


def _safe_link_or_copy(src: Path, dst: Path, mode: str = "symlink") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        rel_src = os.path.relpath(src, dst.parent)
        os.symlink(rel_src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _read_ffpp_metadata(metadata_csv: Path) -> dict[str, dict[str, str]]:
    with metadata_csv.open("r", newline="", encoding="utf-8", errors="ignore") as fh:
        reader = csv.DictReader(fh)
        return {
            Path(row["videoname"]).stem: row for row in reader if row.get("videoname")
        }


def process_ffpp(
    metadata_csv: str | Path,
    faces_dir: str | Path,
    output_dir: str | Path,
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    link_mode: str = "symlink",
    verify_images: bool = False,
    max_samples: int | None = None,
) -> list[ManifestRow]:
    metadata_path = resolve_project_path(metadata_csv)
    faces_path = resolve_project_path(faces_dir)
    output = resolve_project_path(output_dir)
    metadata = _read_ffpp_metadata(metadata_path)
    rows: list[ManifestRow] = []
    images = _iter_images(faces_path)
    if max_samples is not None:
        images = images[:max_samples]
    for idx, img_path in enumerate(tqdm(images, desc="preprocess ffpp")):
        stem = img_path.stem
        meta = metadata.get(stem)
        if meta is None:
            continue
        label_text = meta.get("label", "").upper()
        label = 0 if label_text == "REAL" else 1
        label_name = "real" if label == 0 else "fake"
        group_key = Path(meta.get("original") or meta.get("videoname") or stem).stem
        split = stable_split(group_key, split_ratios)
        if verify_images and cv2.imread(str(img_path), cv2.IMREAD_COLOR) is None:
            continue
        sample_id = f"ffpp_{idx:06d}_{stem}"
        dst = output / split / label_name / f"{sample_id}{img_path.suffix.lower()}"
        _safe_link_or_copy(img_path, dst, link_mode)
        rows.append(
            ManifestRow(
                sample_id=sample_id,
                image_path=str(dst.relative_to(resolve_project_path("."))),
                label=label,
                source_dataset="ffpp",
                original_video=group_key,
                split=split,
            )
        )
    return rows


def process_140k(
    base_dir: str | Path,
    output_dir: str | Path,
    link_mode: str = "symlink",
    verify_images: bool = False,
    max_samples_per_split_label: int | None = None,
) -> list[ManifestRow]:
    base = resolve_project_path(base_dir)
    output = resolve_project_path(output_dir)
    mapping = {
        "train/real": ("train", 0, "real"),
        "train/fake": ("train", 1, "fake"),
        "valid/real": ("val", 0, "real"),
        "valid/fake": ("val", 1, "fake"),
        "validation/real": ("val", 0, "real"),
        "validation/fake": ("val", 1, "fake"),
        "test/real": ("test", 0, "real"),
        "test/fake": ("test", 1, "fake"),
    }
    rows: list[ManifestRow] = []
    for rel, (split, label, label_name) in mapping.items():
        src_dir = base / rel
        images = _iter_images(src_dir)
        if max_samples_per_split_label is not None:
            images = images[:max_samples_per_split_label]
        for idx, img_path in enumerate(tqdm(images, desc=f"preprocess 140k {rel}")):
            if verify_images and cv2.imread(str(img_path), cv2.IMREAD_COLOR) is None:
                continue
            sample_id = f"rf140k_{split}_{label_name}_{idx:06d}_{img_path.stem}"
            dst = output / split / label_name / f"{sample_id}{img_path.suffix.lower()}"
            _safe_link_or_copy(img_path, dst, link_mode)
            rows.append(
                ManifestRow(
                    sample_id=sample_id,
                    image_path=str(dst.relative_to(resolve_project_path("."))),
                    label=label,
                    source_dataset="140k",
                    split=split,
                )
            )
    return rows


def process_ciplab(
    base_dir: str | Path,
    output_dir: str | Path,
    split: str = "val",
    link_mode: str = "symlink",
    verify_images: bool = False,
    max_samples: int | None = None,
) -> list[ManifestRow]:
    base = resolve_project_path(base_dir)
    output = resolve_project_path(output_dir)
    mapping = {
        "training_real": (0, "real"),
        "training_fake": (1, "fake"),
        "real": (0, "real"),
        "fake": (1, "fake"),
    }
    rows: list[ManifestRow] = []
    for rel, (label, label_name) in mapping.items():
        images = _iter_images(base / rel)
        if max_samples is not None:
            images = images[:max_samples]
        for idx, img_path in enumerate(tqdm(images, desc=f"preprocess ciplab {rel}")):
            if verify_images and cv2.imread(str(img_path), cv2.IMREAD_COLOR) is None:
                continue
            sample_id = f"ciplab_{label_name}_{idx:05d}_{img_path.stem}"
            dst = output / split / label_name / f"{sample_id}{img_path.suffix.lower()}"
            _safe_link_or_copy(img_path, dst, link_mode)
            rows.append(
                ManifestRow(
                    sample_id=sample_id,
                    image_path=str(dst.relative_to(resolve_project_path("."))),
                    label=label,
                    source_dataset="ciplab",
                    split=split,
                )
            )
    return rows


def process_antispoof(
    base_dir: str | Path,
    output_dir: str | Path,
    detector: FaceDetector,
    frame_stride: int = 30,
    max_frames_per_video: int = 50,
    split: str = "test",
) -> list[ManifestRow]:
    """Extract aligned face crops from anti-spoofing videos/images for physical attack testing."""
    base = resolve_project_path(base_dir)
    output = resolve_project_path(output_dir)
    real_dirs = {"live_selfie", "live_video"}
    rows: list[ManifestRow] = []
    for child in sorted(p for p in base.iterdir() if p.is_dir()):
        label = 0 if child.name in real_dirs else 1
        label_name = "real" if label == 0 else "fake"
        for img_path in _iter_images(child):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            face = detector.detect_best(img)
            if face is None:
                continue
            crop = align_face(img, face.landmarks, (224, 224))
            sample_id = f"antispoof_{child.name.replace(' ', '_')}_{img_path.stem}"
            dst = output / split / label_name / f"{sample_id}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), crop)
            rows.append(
                ManifestRow(
                    sample_id=sample_id,
                    image_path=str(dst.relative_to(resolve_project_path("."))),
                    label=label,
                    source_dataset="antispoof",
                    original_video=img_path.name,
                    split=split,
                    face_confidence=float(face.confidence),
                )
            )
        for video_path in sorted(
            f
            for f in child.rglob("*")
            if f.is_file() and f.suffix.lower() in VIDEO_EXTS
        ):
            cap = cv2.VideoCapture(str(video_path))
            frame_idx = 0
            saved = 0
            while cap.isOpened() and saved < max_frames_per_video:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % frame_stride == 0:
                    face = detector.detect_best(frame)
                    if face is not None:
                        crop = align_face(frame, face.landmarks, (224, 224))
                        sample_id = f"antispoof_{child.name.replace(' ', '_')}_{video_path.stem}_{frame_idx:06d}"
                        dst = output / split / label_name / f"{sample_id}.jpg"
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(dst), crop)
                        rows.append(
                            ManifestRow(
                                sample_id=sample_id,
                                image_path=str(
                                    dst.relative_to(resolve_project_path("."))
                                ),
                                label=label,
                                source_dataset="antispoof",
                                original_video=video_path.name,
                                split=split,
                                face_confidence=float(face.confidence),
                            )
                        )
                        saved += 1
                frame_idx += 1
            cap.release()
    return rows


def write_manifest(rows: list[ManifestRow], save_path: str | Path) -> None:
    output = resolve_project_path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows([row.as_dict() for row in rows])


def prepare_deepfake_data(
    link_mode: str = "symlink",
    verify_images: bool = False,
    include_antispoof: bool = False,
    max_samples: int | None = None,
) -> list[ManifestRow]:
    cfg = get_dataset_config()
    raw = cfg["raw"]
    output = resolve_project_path(cfg["processed"]["deepfake_faces"])
    manifest = cfg["splits"]["manifest"]
    ratios = (
        float(cfg["splits"]["train_ratio"]),
        float(cfg["splits"]["val_ratio"]),
        float(cfg["splits"]["test_ratio"]),
    )
    output.mkdir(parents=True, exist_ok=True)

    rows: list[ManifestRow] = []
    ffpp_root = resolve_project_path(raw["faceforensicspp"])
    celebdf_root = resolve_project_path(raw["celebdf"])
    rows.extend(
        process_ffpp(
            ffpp_root / "metadata.csv",
            ffpp_root / "faces_224",
            output,
            ratios,
            link_mode,
            verify_images,
            max_samples,
        )
    )
    rows.extend(
        process_140k(
            celebdf_root / "real_vs_fake" / "real-vs-fake",
            output,
            link_mode,
            verify_images,
            max_samples,
        )
    )
    rows.extend(
        process_ciplab(
            celebdf_root / "real_and_fake_face",
            output,
            "val",
            link_mode,
            verify_images,
            max_samples,
        )
    )
    if include_antispoof:
        detector = FaceDetector(min_confidence=0.8, ctx_id=-1)
        rows.extend(process_antispoof(raw["custom_single"], output, detector))
    write_manifest(rows, manifest)
    LOGGER.info("Prepared %d samples and wrote manifest to %s", len(rows), manifest)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare processed deepfake dataset and manifest."
    )
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="Symlink saves disk; copy creates independent files.",
    )
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Open every source image with OpenCV before adding it.",
    )
    parser.add_argument(
        "--include-antispoof",
        action="store_true",
        help="Extract RetinaFace crops from anti-spoofing videos/images into test split.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Debug limit per source/split. Omit for full dataset.",
    )
    args = parser.parse_args()
    rows = prepare_deepfake_data(
        args.link_mode, args.verify_images, args.include_antispoof, args.max_samples
    )
    print(f"Prepared {len(rows)} samples")


if __name__ == "__main__":
    main()
