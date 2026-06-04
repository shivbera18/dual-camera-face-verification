"""Albumentations pipelines for train/val."""
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.utils.config import get_model_config

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transform(image_size: int = 224) -> A.Compose:
    cfg = get_model_config().get("augmentation", {})
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=cfg.get("horizontal_flip", 0.5)),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=cfg.get("brightness_contrast", 0.3),
            ),
            A.ImageCompression(quality_range=(60, 100), p=cfg.get("jpeg_compression", 0.3)),
            A.GaussianBlur(blur_limit=(3, 5), p=cfg.get("gaussian_blur", 0.2)),
            A.GaussNoise(p=cfg.get("gaussian_noise", 0.2)),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=cfg.get("resize_crop", 0.2)
            ),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transform(image_size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    )
