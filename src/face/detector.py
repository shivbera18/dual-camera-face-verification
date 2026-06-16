from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaceResult:
    bbox: np.ndarray
    landmarks: np.ndarray
    confidence: float
    embedding: np.ndarray | None = None

    @property
    def width(self) -> float:
        return float(max(0.0, self.bbox[2] - self.bbox[0]))

    @property
    def height(self) -> float:
        return float(max(0.0, self.bbox[3] - self.bbox[1]))

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def face_size(self) -> int:
        return int(min(self.width, self.height))


class FaceDetector:
    """RetinaFace wrapper via InsightFace FaceAnalysis."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        min_confidence: float = 0.8,
        ctx_id: int = -1,
    ) -> None:
        from insightface.app import FaceAnalysis

        self.model_name = model_name
        self.det_size = det_size
        self.min_confidence = min_confidence
        import onnxruntime
        providers = ["CPUExecutionProvider"]
        if ctx_id >= 0:
            avail = onnxruntime.get_available_providers()
            providers = [p for p in ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"] if p in avail]

        self.app = FaceAnalysis(
            name=model_name, providers=providers
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def detect(self, img: np.ndarray) -> list[FaceResult]:
        """Return detected faces sorted by area descending."""
        faces = self.app.get(img)
        results: list[FaceResult] = []
        for face in faces:
            score = float(getattr(face, "det_score", 0.0))
            if score < self.min_confidence:
                continue
            bbox = np.asarray(face.bbox, dtype=np.float32)
            landmarks = np.asarray(face.kps, dtype=np.float32)
            if bbox.shape != (4,) or landmarks.shape != (5, 2):
                continue
            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                emb = getattr(face, "embedding", None)
            if emb is not None:
                emb = np.asarray(emb, dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
            results.append(FaceResult(bbox=bbox, landmarks=landmarks, confidence=score, embedding=emb))
        return sorted(results, key=lambda f: f.area, reverse=True)

    def detect_best(self, img: np.ndarray) -> FaceResult | None:
        """Return the best face by size and closeness to image center."""
        faces = self.detect(img)
        if not faces:
            return None
        h, w = img.shape[:2]
        center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        max_area = max(face.area for face in faces) or 1.0
        max_dist = float(np.linalg.norm(center)) or 1.0

        def score(face: FaceResult) -> float:
            bbox_center = np.array(
                [
                    (face.bbox[0] + face.bbox[2]) / 2.0,
                    (face.bbox[1] + face.bbox[3]) / 2.0,
                ]
            )
            area_score = face.area / max_area
            center_score = 1.0 - min(
                1.0, float(np.linalg.norm(bbox_center - center)) / max_dist
            )
            return 0.75 * area_score + 0.25 * center_score

        return max(faces, key=score)
