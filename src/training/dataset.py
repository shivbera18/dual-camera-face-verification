from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from src.training.augmentation import get_val_transform
from src.utils.config import resolve_project_path

MANIFEST_DTYPES = {
    "sample_id": "string",
    "image_path": "string",
    "source_dataset": "string",
    "original_video": "string",
    "subject_id": "string",
    "split": "string",
}


class DeepfakeDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Manifest-backed dataset for binary real/fake face classification."""

    def __init__(
        self,
        manifest_csv: str | Path,
        split: str,
        transform: Any | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.manifest_csv = resolve_project_path(manifest_csv)
        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_csv}")
        df = pd.read_csv(self.manifest_csv, dtype=MANIFEST_DTYPES, low_memory=False)
        if (
            "split" not in df.columns
            or "label" not in df.columns
            or "image_path" not in df.columns
        ):
            raise ValueError(
                "Manifest must include split, label, and image_path columns"
            )
        self.df = df[df["split"] == split].reset_index(drop=True)
        if max_samples is not None:
            self.df = self.df.sample(
                n=min(max_samples, len(self.df)), random_state=42
            ).reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows for split={split} in {self.manifest_csv}")
        self.transform = transform or get_val_transform()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        path = resolve_project_path(str(row["image_path"]))
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Unreadable image: {path}")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            rgb = self.transform(image=rgb)["image"]
        if isinstance(rgb, np.ndarray):
            tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
        else:
            tensor = rgb.float()
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return tensor, label

    def labels(self) -> np.ndarray:
        return self.df["label"].astype(int).to_numpy()


def make_weighted_sampler(dataset: DeepfakeDataset) -> WeightedRandomSampler:
    labels = dataset.labels()
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    weights = 1.0 / np.maximum(counts, 1.0)
    sample_weights = weights[labels]
    return WeightedRandomSampler(
        sample_weights.tolist(), num_samples=len(sample_weights), replacement=True
    )
