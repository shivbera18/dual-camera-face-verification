from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from src.utils.config import get_dataset_config, resolve_project_path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
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


def build_manifest(
    processed_dir: str | Path | None = None, save_path: str | Path | None = None
) -> int:
    cfg = get_dataset_config()
    processed = resolve_project_path(
        processed_dir or cfg["processed"]["deepfake_faces"]
    )
    output = resolve_project_path(save_path or cfg["splits"]["manifest"])
    rows: list[dict[str, str | int | float]] = []
    for split in ["train", "val", "test"]:
        for label_name, label in [("real", 0), ("fake", 1)]:
            folder = processed / split / label_name
            if not folder.exists():
                continue
            for img_path in tqdm(
                sorted(folder.rglob("*")), desc=f"manifest {split}/{label_name}"
            ):
                if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                stem = img_path.stem
                source = stem.split("_", 1)[0]
                rows.append(
                    {
                        "sample_id": stem,
                        "image_path": str(
                            img_path.relative_to(resolve_project_path("."))
                        ),
                        "label": label,
                        "source_dataset": source,
                        "original_video": "",
                        "subject_id": "",
                        "split": split,
                        "face_confidence": 1.0,
                        "crop_quality_status": "ok",
                    }
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a manifest by walking processed deepfake_faces folders."
    )
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--save-path", default=None)
    args = parser.parse_args()
    count = build_manifest(args.processed_dir, args.save_path)
    print(f"Wrote {count} rows")


if __name__ == "__main__":
    main()
