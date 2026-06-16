"""Model definitions and wrappers."""

from src.models.arcface import ArcFaceExtractor
from src.models.efficientnet import DeepfakeClassifier

__all__ = ["DeepfakeClassifier", "ArcFaceExtractor"]
