from __future__ import annotations

import numpy as np


class ArcFaceExtractor:
    """ArcFace embedding wrapper around InsightFace buffalo_l."""

    def __init__(
        self,
        model_pack: str = "buffalo_l",
        ctx_id: int = -1,
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        from insightface.app import FaceAnalysis

        self.model_pack = model_pack
        self.app = FaceAnalysis(
            name=model_pack, providers=["CPUExecutionProvider"] if ctx_id < 0 else None
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def get_embedding(self, img_bgr: np.ndarray) -> np.ndarray | None:
        """Return a normalized 512-D embedding for the largest detected face."""
        faces = self.app.get(img_bgr)
        if not faces:
            return None
        face = max(
            faces,
            key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
        )
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            emb = getattr(face, "embedding", None)
        if emb is None:
            return None
        arr = np.asarray(emb, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm <= 0:
            return None
        return arr / norm

    @staticmethod
    def similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        a = np.asarray(emb1, dtype=np.float32)
        b = np.asarray(emb2, dtype=np.float32)
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)
        return float(np.dot(a, b))
