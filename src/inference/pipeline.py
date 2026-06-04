"""End-to-end single-camera verification pipeline."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from src.enrollment.store import EnrollmentStore
from src.face.aligner import align_face, preprocess_for_efficientnet
from src.face.detector import FaceDetector
from src.inference.decision import DecisionEngine
from src.inference.result import VerificationResult, ViewResult
from src.input.camera import CameraInput
from src.models.arcface import ArcFaceExtractor
from src.models.efficientnet import build_classifier
from src.utils.config import get_pipeline_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SingleCamPipeline:
    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        deepfake_model: Optional[torch.nn.Module] = None,
        arcface: Optional[ArcFaceExtractor] = None,
        store: Optional[EnrollmentStore] = None,
        decision_engine: Optional[DecisionEngine] = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.detector = detector or FaceDetector()
        self.deepfake = deepfake_model or build_classifier(device=self.device)
        self.deepfake.eval()
        self.arcface = arcface or ArcFaceExtractor()
        self.store = store or EnrollmentStore()
        self.engine = decision_engine or DecisionEngine(mode="single_camera")
        cfg = get_pipeline_config().get("single_camera", {})
        self.face_size = tuple(self.deepfake.input_size)

    def _view_result(self, img: np.ndarray) -> tuple[ViewResult, Optional[np.ndarray]]:
        t0 = time.time()
        best = self.detector.detect_best(img)
        if best is None:
            return (
                ViewResult(
                    face_detected=False,
                    fake_score=0.0,
                    match_score=0.0,
                    latency_ms=(time.time() - t0) * 1000.0,
                ),
                None,
            )
        try:
            crop = align_face(img, best.landmarks, output_size=self.face_size)
        except Exception:
            return (
                ViewResult(
                    face_detected=False,
                    fake_score=0.0,
                    match_score=0.0,
                    latency_ms=(time.time() - t0) * 1000.0,
                ),
                None,
            )
        x = preprocess_for_efficientnet(crop).to(self.device)
        with torch.no_grad():
            fake_score = float(self.deepfake(x).item())
        emb = self.arcface.get_embedding(crop)
        return (
            ViewResult(
                face_detected=True,
                fake_score=fake_score,
                match_score=0.0,
                latency_ms=(time.time() - t0) * 1000.0,
                bbox=best.bbox.tolist(),
            ),
            emb,
        )

    def run(self, img: np.ndarray, user_id: Optional[str] = None) -> VerificationResult:
        view, emb = self._view_result(img)
        if not view.face_detected or emb is None:
            return self.engine.decide(0.0, 0.0, False, user_id=user_id)
        match_score = 0.0
        if user_id is not None:
            tmpl = self.store.get_template(user_id)
            if tmpl is not None:
                match_score = ArcFaceExtractor.similarity(emb, tmpl)
        view.match_score = match_score
        return self.engine.decide(view.fake_score, match_score, True, user_id=user_id)

    def _draw_overlay(
        self, frame: np.ndarray, view: ViewResult, result: VerificationResult
    ) -> np.ndarray:
        disp = frame.copy()
        if view.bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in view.bbox]
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text_lines = [
            f"decision: {result.decision}",
            f"fake: {result.fake_score:.3f}",
            f"match: {result.match_score:.3f}",
        ]
        if result.user_id:
            text_lines.append(f"user: {result.user_id}")
        for i, line in enumerate(text_lines):
            cv2.putText(
                disp,
                line,
                (10, 25 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        return disp

    def run_live(self, camera_index: int = 0, user_id: Optional[str] = None) -> None:
        with CameraInput(source=camera_index) as cam:
            window = "single-cam verify"
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            for frame, _ in cam.frames():
                view, emb = self._view_result(frame)
                if not view.face_detected or emb is None:
                    result = self.engine.decide(0.0, 0.0, False, user_id=user_id)
                else:
                    match_score = 0.0
                    if user_id is not None:
                        tmpl = self.store.get_template(user_id)
                        if tmpl is not None:
                            match_score = ArcFaceExtractor.similarity(emb, tmpl)
                    view.match_score = match_score
                    result = self.engine.decide(
                        view.fake_score, match_score, True, user_id=user_id
                    )
                disp = self._draw_overlay(frame, view, result)
                cv2.imshow(window, disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyWindow(window)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--user_id", type=str, default=None)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--image", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = SingleCamPipeline()
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise FileNotFoundError(args.image)
        result = pipeline.run(img, user_id=args.user_id)
        logger.info("result: %s", result.to_dict())
        print(result.to_dict())
    else:
        pipeline.run_live(camera_index=args.camera, user_id=args.user_id)


if __name__ == "__main__":
    main()
