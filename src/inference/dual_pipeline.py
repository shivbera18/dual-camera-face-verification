from __future__ import annotations

import argparse
import json
import time

import cv2

from src.inference.decision import DecisionEngine
from src.inference.pipeline import SingleCamPipeline, build_pipeline
from src.inference.result import VerificationResult
from src.input.dual_camera import DualCameraCapture
from src.input.frame import Frame
from src.utils.config import get_pipeline_config


def fuse_scores(
    left: VerificationResult,
    right: VerificationResult,
    fake_method: str = "mean",
    match_method: str = "max",
) -> tuple[float, float, bool, float, int]:
    valid = [res for res in [left, right] if res.face_detected]
    if not valid:
        return 0.0, 0.0, False, 0.0, 0
    fake_values = [res.fake_score for res in valid]
    match_values = [res.match_score for res in valid]
    fake_score = (
        max(fake_values)
        if fake_method == "max"
        else sum(fake_values) / len(fake_values)
    )
    match_score = (
        sum(match_values) / len(match_values)
        if match_method == "mean"
        else max(match_values)
    )
    face_conf = max(res.face_confidence for res in valid)
    face_size = max(res.face_size for res in valid)
    return float(fake_score), float(match_score), True, float(face_conf), int(face_size)


class DualCamPipeline:
    def __init__(
        self,
        single_pipeline: SingleCamPipeline,
        decision_engine: DecisionEngine,
        fake_fusion: str = "mean",
        match_fusion: str = "max",
        fallback_single_view: bool = True,
    ) -> None:
        self.single_pipeline = single_pipeline
        self.decision_engine = decision_engine
        self.fake_fusion = fake_fusion
        self.match_fusion = match_fusion
        self.fallback_single_view = fallback_single_view

    def run(
        self, left_frame: Frame, right_frame: Frame, user_id: str
    ) -> VerificationResult:
        started = time.perf_counter()
        left = self.single_pipeline.run(left_frame.img, user_id)
        right = self.single_pipeline.run(right_frame.img, user_id)
        if not self.fallback_single_view and (
            not left.face_detected or not right.face_detected
        ):
            return VerificationResult(
                "RETRY",
                0.0,
                0.0,
                user_id,
                False,
                (time.perf_counter() - started) * 1000.0,
                "dual_view_required",
            )
        fake, match, detected, conf, size = fuse_scores(
            left, right, self.fake_fusion, self.match_fusion
        )
        result = self.decision_engine.decide(
            fake_score=fake,
            match_score=match,
            user_id=user_id,
            face_detected=detected,
            face_confidence=conf,
            face_size=size,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
        result.reason = (
            f"dual_fused:{result.reason};left={left.decision};right={right.decision}"
        )
        return result

    def run_live(self, user_id: str, left_idx: int = 0, right_idx: int = 1) -> None:
        capture = DualCameraCapture(
            left_idx=left_idx,
            right_idx=right_idx,
            sync_delta_ms=float(get_pipeline_config()["dual_camera"]["sync_delta_ms"]),
        )
        capture.start()
        try:
            while True:
                pair = capture.get_pair()
                if pair is None:
                    if (cv2.waitKey(1) & 0xFF) in {27, ord("q")}:
                        break
                    continue
                left, right = pair
                result = self.run(left, right, user_id)
                view = cv2.hconcat([left.img, right.img])
                color = (0, 255, 0) if result.decision == "ACCEPT" else (0, 0, 255)
                cv2.putText(
                    view,
                    f"{result.decision} fake={result.fake_score:.3f} match={result.match_score:.3f} {result.latency_ms:.1f}ms",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    color,
                    2,
                )
                cv2.imshow("dual-camera verification", view)
                if (cv2.waitKey(1) & 0xFF) in {27, ord("q")}:
                    break
        finally:
            capture.stop()
            cv2.destroyAllWindows()


def build_dual_pipeline(
    device: str = "auto", checkpoint: str | None = None, ctx_id: int = -1
) -> DualCamPipeline:
    cfg = get_pipeline_config()
    device = get_device(device)
    if ctx_id < 0 and device.type in ("cuda", "mps"):
        ctx_id = 0
    single = build_pipeline(device, checkpoint, ctx_id)
    decision = DecisionEngine(
        fake_threshold=float(cfg["dual_camera"]["fake_threshold"]),
        match_threshold=float(cfg["dual_camera"]["match_threshold"]),
        min_face_confidence=float(cfg["single_camera"]["min_face_confidence"]),
        min_face_size=int(cfg["single_camera"]["min_face_size"]),
    )
    return DualCamPipeline(
        single,
        decision,
        fake_fusion=str(cfg["dual_camera"]["fusion"]["fake_score"]),
        match_fusion=str(cfg["dual_camera"]["fusion"]["match_score"]),
        fallback_single_view=bool(cfg["dual_camera"].get("fallback_single_view", True)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dual-camera verification demo.")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--left-camera", type=int, default=0)
    parser.add_argument("--right-camera", type=int, default=1)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ctx-id", type=int, default=-1)
    args = parser.parse_args()
    pipeline = build_dual_pipeline(args.device, args.checkpoint, args.ctx_id)
    pipeline.run_live(args.user_id, args.left_camera, args.right_camera)


if __name__ == "__main__":
    main()
