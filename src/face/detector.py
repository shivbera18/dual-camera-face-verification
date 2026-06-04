"""RetinaFace wrapper around InsightFace's FaceAnalysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from insightface.app import FaceAnalysis

from src.utils.config import get_model_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FaceResult:
    bbox: np.ndarray
    landmarks: np.ndarray
    confidence: float
    area: float

    def as_dict(self) -> dict:
        return {
            "bbox": self.bbox.tolist(),
            "landmarks": self.landmarks.tolist(),
            "confidence": float(self.confidence),
            "area": float(self.area),
        }


class FaceDetector:
    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        min_confidence: float = 0.8,
        ctx_id: int = -1,
    ) -> None:
        cfg = get_model_config().get("retinaface", {})
        self.min_confidence = float(cfg.get("detection_threshold", min_confidence))
        self.min_face_size = int(cfg.get("min_face_size", 40))
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        logger.info(
            "FaceDetector ready: model=%s det_size=%s min_conf=%.2f",
            model_name, det_size, self.min_confidence,
        )

    def _to_result(self, face) -> Optional[FaceResult]:
        bbox = np.asarray(face.bbox, dtype=np.float32)
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        if min(w, h) < self.min_face_size:
            return None
        if face.det_score < self.min_confidence:
            return None
        landmarks = np.asarray(face.kps, dtype=np.float32)
        if landmarks.shape != (5, 2):
            return None
        return FaceResult(
            bbox=bbox,
            landmarks=landmarks,
            confidence=float(face.det_score),
            area=float(w * h),
        )

    def detect(self, img: np.ndarray) -> list[FaceResult]:
        if img is None or img.size == 0:
            return []
        faces = self.app.get(img)
        results = [self._to_result(f) for f in faces]
        results = [r for r in results if r is not None]
        results.sort(key=lambda r: r.area, reverse=True)
        return results

    def detect_best(self, img: np.ndarray) -> Optional[FaceResult]:
        results = self.detect(img)
        if not results:
            return None
        if len(results) == 1:
            return results[0]
        h, w = img.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        def center_score(r: FaceResult) -> float:
            bx1, by1, bx2, by2 = r.bbox
            face_cx = (bx1 + bx2) / 2.0
            face_cy = (by1 + by2) / 2.0
            dist = ((face_cx - cx) ** 2 + (face_cy - cy) ** 2) ** 0.5
            return r.area - dist * dist * 0.5
        return max(results, key=center_score)
