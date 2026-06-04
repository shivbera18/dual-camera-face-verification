"""Build the deepfake_manifest.csv from the processed dataset layout."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Iterable

from src.utils.config import get_dataset_config, resolve
from src.utils.logger import get_logger

logger = get_logger(__name__)

MANIFEST_COLUMNS = [
    "sample_id",
    "image_path",
    "label",
    "source_dataset",
    "original_video",
    "subject_id",
    "split",
    "face_confidence",
    "crop_quality_status",
]


def _hash_id(path: Path) -> str:
    return hashlib.md5(str(path).encode("utf-8")).hexdigest()[:12]


def _iter_images(processed_dir: Path) -> Iterable[dict]:
    for split in ("train", "val", "test"):
        for label_name, label_int in (("real", 0), ("fake", 1)):
            sub = processed_dir / split / label_name
            if not sub.exists():
                continue
            for img_path in sorted(sub.iterdir()):
                if not img_path.is_file():
                    continue
                rel = img_path.relative_to(resolve("data"))
                yield {
                    "sample_id": _hash_id(img_path),
                    "image_path": str(rel),
                    "label": label_int,
                    "split": split,
                    "source_dataset": "deepfake_faces",
                    "original_video": "",
                    "subject_id": "",
                    "face_confidence": 1.0,
                    "crop_quality_status": "ok",
                }


def build_manifest(processed_dir: str | Path, manifest_path: str | Path) -> int:
    processed_dir = resolve(processed_dir)
    manifest_path = resolve(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in _iter_images(processed_dir):
            writer.writerow(row)
            count += 1
    logger.info("manifest written: %s rows=%d", manifest_path, count)
    return count


def main() -> None:
    ds = get_dataset_config()
    n = build_manifest(
        processed_dir=ds["processed"]["deepfake_faces"],
        manifest_path=ds["splits"]["manifest"],
    )
    print(f"manifest rows: {n}")


if __name__ == "__main__":
    main()
