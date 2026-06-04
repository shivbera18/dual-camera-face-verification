"""YAML configuration loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open("r") as f:
        return yaml.safe_load(f)


def get_dataset_config() -> dict[str, Any]:
    return load_config(CONFIGS_DIR / "dataset.yaml")


def get_model_config() -> dict[str, Any]:
    return load_config(CONFIGS_DIR / "model.yaml")


def get_pipeline_config() -> dict[str, Any]:
    return load_config(CONFIGS_DIR / "pipeline.yaml")


def get_repo_root() -> Path:
    return REPO_ROOT


def resolve(path_str: str | Path) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p
