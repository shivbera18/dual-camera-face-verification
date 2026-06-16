from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.config import get_dataset_config, resolve_project_path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _count_files(path: Path, exts: set[str]) -> int | None:
    if not path.exists():
        return None
    return sum(1 for f in path.rglob("*") if f.is_file() and f.suffix.lower() in exts)


def _csv_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    with path.open("r", newline="", encoding="utf-8", errors="ignore") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    label_counts: Counter[str] = Counter()
    for row in rows:
        label = (row.get("label") or row.get("Label") or row.get("class") or row.get("target") or "").strip()
        label_counts[label] += 1
    return {"exists": True, "rows": len(rows), "columns": reader.fieldnames or [], "label_counts": dict(label_counts)}


def check_dataset() -> dict[str, Any]:
    cfg = get_dataset_config()
    raw = cfg["raw"]
    ffpp_root = resolve_project_path(raw["faceforensicspp"])
    celebdf_root = resolve_project_path(raw["celebdf"])
    single_root = resolve_project_path(raw["custom_single"])
    lfw_dir = resolve_project_path(raw["lfw"])
    lfw_pairs = resolve_project_path(raw["lfw_pairs"])
    processed = resolve_project_path(cfg["processed"]["deepfake_faces"])
    manifest = resolve_project_path(cfg["splits"]["manifest"])

    ffpp_faces = ffpp_root / "faces_224"
    real_vs_fake = celebdf_root / "real_vs_fake" / "real-vs-fake"
    ciplab = celebdf_root / "real_and_fake_face"

    report: dict[str, Any] = {
        "paths": {},
        "counts": {},
        "status": {},
        "generated_artifacts": {},
    }

    expected_paths = {
        "ffpp_faces": ffpp_faces,
        "ffpp_metadata": ffpp_root / "metadata.csv",
        "140k_base": real_vs_fake,
        "ciplab_base": ciplab,
        "anti_spoof": single_root,
        "lfw_dir": lfw_dir,
        "lfw_pairs": lfw_pairs,
    }
    for key, path in expected_paths.items():
        report["paths"][key] = {"path": str(path), "exists": path.exists()}

    report["counts"]["ffpp_images"] = _count_files(ffpp_faces, IMAGE_EXTS)
    report["counts"]["ffpp_metadata"] = _csv_summary(ffpp_root / "metadata.csv")
    report["counts"]["140k_images_total"] = _count_files(real_vs_fake, IMAGE_EXTS)
    report["counts"]["140k_splits"] = {
        rel: _count_files(real_vs_fake / rel, IMAGE_EXTS)
        for rel in ["train/real", "train/fake", "valid/real", "valid/fake", "test/real", "test/fake"]
        if (real_vs_fake / rel).exists()
    }
    report["counts"]["ciplab"] = {
        rel: _count_files(ciplab / rel, IMAGE_EXTS)
        for rel in ["training_real", "training_fake", "real", "fake"]
        if (ciplab / rel).exists()
    }
    report["counts"]["anti_spoof_videos"] = _count_files(single_root, VIDEO_EXTS)
    report["counts"]["anti_spoof_images"] = _count_files(single_root, IMAGE_EXTS)
    if single_root.exists():
        report["counts"]["anti_spoof_by_dir"] = {
            child.name: {
                "images": _count_files(child, IMAGE_EXTS),
                "videos": _count_files(child, VIDEO_EXTS),
            }
            for child in sorted(single_root.iterdir())
            if child.is_dir()
        }
    report["counts"]["lfw_images"] = _count_files(lfw_dir, IMAGE_EXTS)
    if lfw_pairs.exists():
        lines = [line for line in lfw_pairs.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
        report["counts"]["lfw_pairs_lines"] = len(lines)

    report["generated_artifacts"]["processed_images"] = _count_files(processed, IMAGE_EXTS)
    report["generated_artifacts"]["manifest"] = _csv_summary(manifest)

    required_ok = all(v["exists"] for v in report["paths"].values())
    counts_ok = (
        (report["counts"].get("ffpp_images") or 0) >= 95_000
        and (report["counts"].get("140k_images_total") or 0) >= 110_000
        and (report["counts"].get("ciplab", {}).get("training_real") or 0) >= 1_000
        and (report["counts"].get("ciplab", {}).get("training_fake") or 0) >= 900
        and (report["counts"].get("lfw_images") or 0) >= 13_000
        and (report["counts"].get("lfw_pairs_lines") or 0) >= 6_001
    )
    report["status"]["raw_dataset_complete"] = bool(required_ok and counts_ok)
    report["status"]["processed_dataset_ready"] = bool(
        (report["generated_artifacts"].get("processed_images") or 0) > 0
        and report["generated_artifacts"].get("manifest", {}).get("exists", False)
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check downloaded dataset completeness.")
    parser.add_argument("--save", default="artifacts/metrics/dataset_check_report.json")
    args = parser.parse_args()
    report = check_dataset()
    out = resolve_project_path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved dataset report to {out}")


if __name__ == "__main__":
    main()
