"""ArcFace embedding extractor wrapping InsightFace."""
from __future__ import annotations

from typing import Optional

import numpy as np
from insightface.app import FaceAnalysis

from src.utils.config import get_model_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ArcFaceExtractor:
    def __init__(self, model_pack: str = "buffalo_l", ctx_id: int = -1) -> None:
        self.app = FaceAnalysis(name=model_pack, allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=ctx_id)
        cfg = get_model_config().get("arcface", {})
        self.embedding_dim: int = int(cfg.get("embedding_dim", 512))
        logger.info("ArcFaceExtractor ready pack=%s", model_pack)

    def get_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        if aligned_face is None:
            raise ValueError("aligned_face is None")
        faces = self.app.get(aligned_face)
        if not faces:
            raise RuntimeError("ArcFace: no face detected in aligned crop")
        emb = np.asarray(faces[0].normed_embedding, dtype=np.float32)
        if emb.shape[0] != self.embedding_dim:
            emb = emb[: self.embedding_dim]
        norm = float(np.linalg.norm(emb)) + 1e-12
        return emb / norm

    def get_embedding_from_face_app(self, faces_list) -> Optional[np.ndarray]:
        if not faces_list:
            return None
        emb = np.asarray(faces_list[0].normed_embedding, dtype=np.float32)
        if emb.shape[0] != self.embedding_dim:
            emb = emb[: self.embedding_dim]
        norm = float(np.linalg.norm(emb)) + 1e-12
        return emb / norm

    @staticmethod
    def similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        a = np.asarray(emb1, dtype=np.float32).ravel()
        b = np.asarray(emb2, dtype=np.float32).ravel()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.dot(a, b) / denom)
