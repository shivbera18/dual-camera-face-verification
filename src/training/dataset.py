"""PyTorch Dataset backed by the deepfake manifest."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.config import resolve


class DeepfakeDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        split: str,
        transform: Optional[object] = None,
    ) -> None:
        manifest_csv = resolve(manifest_csv)
        if not manifest_csv.exists():
            raise FileNotFoundError(f"manifest not found: {manifest_csv}")
        df = pd.read_csv(manifest_csv)
        df = df[df["split"] == split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"no rows for split={split} in {manifest_csv}")
        self.df = df
        self.transform = transform
        self.data_root = resolve("data")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        rel = row["image_path"]
        img_path = self.data_root / rel
        if not img_path.exists():
            raise FileNotFoundError(f"image missing: {img_path}")
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed["image"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        label = int(row["label"])
        return img, label
