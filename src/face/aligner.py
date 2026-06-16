from __future__ import annotations

import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# InsightFace 5-point landmark template for 112x112 crops.
_ARCFACE_REF_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def align_face(
    img: np.ndarray, landmarks: np.ndarray, output_size: tuple[int, int] = (224, 224)
) -> np.ndarray:
    """Align a BGR face crop using 5 landmarks and return BGR uint8 output."""
    if landmarks.shape != (5, 2):
        raise ValueError(f"Expected landmarks with shape (5, 2), got {landmarks.shape}")
    out_w, out_h = output_size[0], output_size[1]
    scale_x = out_w / 112.0
    scale_y = out_h / 112.0
    dst = _ARCFACE_REF_112.copy()
    dst[:, 0] *= scale_x
    dst[:, 1] *= scale_y
    matrix, _ = cv2.estimateAffinePartial2D(
        landmarks.astype(np.float32), dst, method=cv2.LMEDS
    )
    if matrix is None:
        x1, y1 = np.maximum(landmarks.min(axis=0) - 40, 0).astype(int)
        x2, y2 = np.minimum(
            landmarks.max(axis=0) + 40, [img.shape[1], img.shape[0]]
        ).astype(int)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError("Could not align face and fallback crop is empty")
        return cv2.resize(crop, output_size, interpolation=cv2.INTER_AREA)
    aligned = cv2.warpAffine(
        img,
        matrix,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return aligned.astype(np.uint8)


def preprocess_for_efficientnet(crop: np.ndarray) -> torch.Tensor:
    """Normalize a BGR crop to an EfficientNet tensor with shape [1, 3, 224, 224]."""
    resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor


def preprocess_for_arcface(crop: np.ndarray) -> np.ndarray:
    """Return a 112x112 BGR crop suitable for InsightFace ArcFace extraction."""
    return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA).astype(np.uint8)
