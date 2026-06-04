"""Face alignment + preprocessing for EfficientNet and ArcFace."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch

from src.utils.config import get_model_config

_ARCFACE_REF = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _get_output_size() -> Tuple[int, int]:
    cfg = get_model_config().get("retinaface", {}).get("align_output_size", [224, 224])
    return int(cfg[0]), int(cfg[1])


def align_face(
    img: np.ndarray,
    landmarks: np.ndarray,
    output_size: Tuple[int, int] | None = None,
) -> np.ndarray:
    if img is None or landmarks is None or landmarks.shape != (5, 2):
        raise ValueError("align_face requires 5x2 landmarks")
    if output_size is None:
        output_size = _get_output_size()
    ref = _ARCFACE_REF.copy()
    ref[:, 0] *= output_size[0] / 112.0
    ref[:, 1] *= output_size[1] / 112.0
    transform = cv2.estimateAffinePartial2D(
        landmarks.astype(np.float32), ref, method=cv2.LMEDS
    )[0]
    if transform is None:
        raise RuntimeError("Failed to estimate alignment transform")
    aligned = cv2.warpAffine(
        img, transform, output_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    return aligned


def preprocess_for_efficientnet(crop: np.ndarray) -> torch.Tensor:
    if crop is None:
        raise ValueError("crop is None")
    if crop.shape[2] == 3:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    else:
        rgb = crop
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(arr).unsqueeze(0).contiguous()
    return tensor


def preprocess_for_arcface(crop: np.ndarray) -> np.ndarray:
    if crop is None:
        raise ValueError("crop is None")
    if crop.shape[0] != 112 or crop.shape[1] != 112:
        crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)
    if crop.shape[2] == 3:
        bgr = crop
    else:
        bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    arr = bgr.astype(np.float32)
    arr = (arr - 127.5) / 127.5
    arr = np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(arr)
