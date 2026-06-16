from __future__ import annotations

import albumentations as A

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _random_resized_crop(image_size: tuple[int, int], p: float) -> A.BasicTransform:
    try:
        return A.RandomResizedCrop(
            size=image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1), p=p
        )
    except Exception:
        return A.RandomResizedCrop(
            height=image_size[0],
            width=image_size[1],
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
            p=p,
        )


def _image_compression(p: float) -> A.BasicTransform:
    try:
        return A.ImageCompression(quality_range=(60, 100), p=p)
    except Exception:
        return A.ImageCompression(quality_lower=60, quality_upper=100, p=p)


def _coarse_dropout(p: float) -> A.BasicTransform:
    try:
        return A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(8, 24),
            hole_width_range=(8, 24),
            p=p,
        )
    except Exception:
        return A.CoarseDropout(
            max_holes=3,
            max_height=24,
            max_width=24,
            min_holes=1,
            min_height=8,
            min_width=8,
            p=p,
        )


def get_train_transform(image_size: tuple[int, int] = (224, 224)) -> A.Compose:
    """Albumentations pipeline for EfficientNet training."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=15,
                border_mode=0,
                p=0.4,
            ),
            _random_resized_crop(image_size, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.2
            ),
            _image_compression(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(p=0.2),
            _coarse_dropout(p=0.1),
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transform(image_size: tuple[int, int] = (224, 224)) -> A.Compose:
    """Validation/test transforms: resize and ImageNet normalization only."""
    return A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
