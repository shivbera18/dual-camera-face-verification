from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a project-relative or absolute path to an absolute Path."""
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return a plain dictionary."""
    config_path = resolve_project_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return data


def get_dataset_config() -> dict[str, Any]:
    return load_config(CONFIG_DIR / "dataset.yaml")


def get_model_config() -> dict[str, Any]:
    return load_config(CONFIG_DIR / "model.yaml")


def get_pipeline_config() -> dict[str, Any]:
    return load_config(CONFIG_DIR / "pipeline.yaml")


def load_all_configs() -> dict[str, dict[str, Any]]:
    return {
        "dataset": get_dataset_config(),
        "model": get_model_config(),
        "pipeline": get_pipeline_config(),
    }
