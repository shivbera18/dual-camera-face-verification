"""Dual-camera verification pipeline with score fusion."""
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
from src.inference.pipeline import SingleCamPipeline
from src.inference.result import VerificationResult, ViewResult
from src.input.dual_camera import DualCameraCapture
from src.input.frame import Frame
from src.models.arcface import ArcFaceExtractor
from src.models.efficientnet import build_classifier
from src.utils.config import get_pipeline_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DualCamPipeline:
    def __init__(
        self,
        single_pipeline: Optional[SingleCamPipeline] = None,
        fusion: Optional[str] = None,
        save_image_pairs: bool = False,
    ) -> None:
        self.single = single_pipeline or SingleCamPipeline()
        cfg = get_pipeline_config()
        dual_cfg = cfg.get("dual_camera", {})
        self.engine = DecisionEngine(
            fake_threshold=dual_cfg.get("fake_threshold"),
            match_threshold=dual_cfg.get("match_threshold"),
            mode="dual_camera",
        )
        self.save_image_pairs = bool(
            save_image_pairs if save_image_pairs is not None else cfg.get("logging", {}).get("save_image_pairs", False)
        )
        self.fallback_single_view = bool(dual_cfg.get("fallback_single_view", True))
        self.collection_dir = "data/raw/custom/dual_cam/evaluation/real"

    def _process_view(self, frame: Frame) -> tuple[ViewResult, Optional[np.ndarray]]:
        t0 = time.time()
        best = self.single.detector.detect_best(frame.img)
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
            crop = align_face(frame.img, best.landmarks, output_size=self.single.face_size)
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
        x = preprocess_for_efficientnet(crop).to(self.single.device)
        with torch.no_grad():
            fake_score = float(self.single.deepfake(x).item())
        emb = self.single.arcface.get_embedding(crop)
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

    def run(
        self,
        left_frame: Frame,
        right_frame: Frame,
        user_id: Optional[str] = None,
    ) -> VerificationResult:
        lv, le = self._process_view(left_frame)
        rv, re = self._process_view(right_frame)
        tmpl = self.single.store.get_template(user_id) if user_id else None
        if tmpl is not None:
            if le is not None:
                lv.match_score = ArcFaceExtractor.similarity(le, tmpl)
            if re is not None:
                rv.match_score = ArcFaceExtractor.similarity(re, tmpl)
        if not self.fallback_single_view and (not lv.face_detected or not rv.face_detected):
            result = self.engine.decide_dual(
                {"left": lv, "right": rv}, user_id=user_id
            )
            if not lv.face_detected or not rv.face_detected:
                result.reason = "dual_view_required_but_missing"
            return result
        result = self.engine.decide_dual(
            {"left": lv, "right": rv}, user_id=user_id
        )
        result.latency_ms = float(lv.latency_ms + rv.latency_ms)
        return result

    def _draw_overlay(
        self,
        left: Frame,
        right: Frame,
        result: VerificationResult,
        lv: ViewResult,
        rv: ViewResult,
    ) -> np.ndarray:
        h, w = left.img.shape[:2]
        canvas = np.zeros((h, 2 * w + 20, 3), dtype=np.uint8)
        canvas[:, :w] = left.img
        canvas[:, w + 20:] = right.img
        for i, (frame, view, label) in enumerate(
            ((left, lv, "LEFT"), (right, rv, "RIGHT"))
        ):
            x_offset = i * (w + 20)
            if view.bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in view.bbox]
                cv2.rectangle(canvas, (x_offset + x1, y1), (x_offset + x2, y2), (0, 255, 0), 2)
            lines = [
                f"{label} face={view.face_detected}",
                f"fake={view.fake_score:.3f}",
                f"match={view.match_score:.3f}",
            ]
            for j, line in enumerate(lines):
                cv2.putText(
                    canvas,
                    line,
                    (x_offset + 10, 25 + j * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
        info = [
            f"decision: {result.decision}",
            f"fused_fake: {result.fake_score:.3f}",
            f"fused_match: {result.match_score:.3f}",
        ]
        for j, line in enumerate(info):
            cv2.putText(
                canvas,
                line,
                (10, h - 60 + j * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
            )
        return canvas

    def run_live(
        self,
        user_id: Optional[str] = None,
        left_idx: int = 0,
        right_idx: int = 1,
        save_pairs: bool = False,
    ) -> None:
        cap = DualCameraCapture(left_idx=left_idx, right_idx=right_idx)
        cap.start()
        window = "dual-cam verify"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        try:
            pair_idx = 0
            while True:
                pair = cap.get_pair()
                if pair is None:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue
                lf, rf = pair
                lv, le = self._process_view(lf)
                rv, re = self._process_view(rf)
                tmpl = self.single.store.get_template(user_id) if user_id else None
                if tmpl is not None:
                    if le is not None:
                        lv.match_score = ArcFaceExtractor.similarity(le, tmpl)
                    if re is not None:
                        rv.match_score = ArcFaceExtractor.similarity(re, tmpl)
                result = self.engine.decide_dual(
                    {"left": lv, "right": rv}, user_id=user_id
                )
                result.latency_ms = float(lv.latency_ms + rv.latency_ms)
                if save_pairs and (lv.face_detected or rv.face_detected):
                    meta = {
                        "session_id": "live",
                        "pair_index": pair_idx,
                        "label": "real",
                        "attack_type": "none",
                        "face_detected_left": lv.face_detected,
                        "face_detected_right": rv.face_detected,
                        "fake_score_left": lv.fake_score,
                        "fake_score_right": rv.fake_score,
                        "match_score_left": lv.match_score,
                        "match_score_right": rv.match_score,
                        "fused_fake_score": result.fake_score,
                        "fused_match_score": result.match_score,
                        "decision": result.decision,
                    }
                    cap.save_pair(lf, rf, meta, self.collection_dir)
                    pair_idx += 1
                disp = self._draw_overlay(lf, rf, result, lv, rv)
                cv2.imshow(window, disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.stop()
            cv2.destroyWindow(window)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--user_id", type=str, default=None)
    p.add_argument("--left", type=int, default=0)
    p.add_argument("--right", type=int, default=1)
    p.add_argument("--save_pairs", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = DualCamPipeline(save_image_pairs=args.save_pairs)
    pipeline.run_live(
        user_id=args.user_id,
        left_idx=args.left,
        right_idx=args.right,
        save_pairs=args.save_pairs,
    )


if __name__ == "__main__":
    main()
