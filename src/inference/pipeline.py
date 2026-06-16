from __future__ import annotations

import argparse
import json
import time

import cv2
import torch

from src.enrollment.store import EnrollmentStore
from src.face.aligner import align_face, preprocess_for_efficientnet
from src.face.detector import FaceDetector
from src.inference.decision import DecisionEngine
from src.inference.result import VerificationResult
from src.models.arcface import ArcFaceExtractor
from src.models.efficientnet import DeepfakeClassifier, load_deepfake_checkpoint
from src.training.train_baseline import get_device
from src.utils.config import get_model_config, get_pipeline_config, resolve_project_path


class SingleCamPipeline:
    def __init__(
        self,
        detector: FaceDetector,
        deepfake_model: torch.nn.Module,
        arcface: ArcFaceExtractor,
        store: EnrollmentStore,
        decision_engine: DecisionEngine,
        device: torch.device,
    ) -> None:
        self.detector = detector
        self.deepfake_model = deepfake_model.to(device).eval()
        self.arcface = arcface
        self.store = store
        self.decision_engine = decision_engine
        self.device = device

    @torch.no_grad()
    def fake_probability(self, crop_bgr) -> float:
        tensor = preprocess_for_efficientnet(crop_bgr).to(self.device)
        logits = self.deepfake_model(tensor)
        return float(torch.sigmoid(logits).detach().cpu().item())

    def run(self, img, user_id: str | None) -> VerificationResult:
        started = time.perf_counter()
        face = self.detector.detect_best(img)
        if face is None:
            latency = (time.perf_counter() - started) * 1000.0
            return self.decision_engine.decide(
                fake_score=0.0,
                match_score=0.0,
                user_id=user_id,
                face_detected=False,
                latency_ms=latency,
            )
        crop = align_face(img, face.landmarks, (224, 224))
        fake_score = self.fake_probability(crop)
        match_score = 0.0
        no_template = False
        if user_id:
            template = self.store.get_template(user_id)
            if template is None:
                no_template = True
            else:
                emb = self.arcface.get_embedding(crop)
                if emb is not None:
                    match_score = self.arcface.similarity(template, emb)
        latency = (time.perf_counter() - started) * 1000.0
        return self.decision_engine.decide(
            fake_score=fake_score,
            match_score=match_score,
            user_id=user_id,
            face_detected=True,
            face_confidence=float(face.confidence),
            face_size=face.face_size,
            latency_ms=latency,
            no_template=no_template,
        )

    def run_live(self, user_id: str, camera_index: int = 0) -> None:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue
                result = self.run(frame, user_id)
                color = (0, 255, 0) if result.decision == "ACCEPT" else (0, 0, 255)
                overlay = f"{result.decision} fake={result.fake_score:.3f} match={result.match_score:.3f} {result.latency_ms:.1f}ms"
                cv2.putText(
                    frame, overlay, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2
                )
                cv2.imshow("single-camera verification", frame)
                if (cv2.waitKey(1) & 0xFF) in {27, ord("q")}:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


def build_pipeline(
    device_name: str = "auto", checkpoint: str | None = None, ctx_id: int = -1
) -> SingleCamPipeline:
    model_cfg = get_model_config()
    pipe_cfg = get_pipeline_config()
    device = get_device(device_name)
    detector = FaceDetector(
        model_name=model_cfg["arcface"]["model"],
        min_confidence=float(pipe_cfg["single_camera"]["min_face_confidence"]),
        ctx_id=ctx_id,
    )
    arcface = ArcFaceExtractor(model_pack=model_cfg["arcface"]["model"], ctx_id=ctx_id)
    deepfake, _ = load_deepfake_checkpoint(
        resolve_project_path(checkpoint or model_cfg["efficientnet"]["checkpoint"]),
        device=device,
        pretrained=False,
    )
    store = EnrollmentStore(model_cfg["arcface"]["enrollment_db"])
    decision = DecisionEngine(
        fake_threshold=float(pipe_cfg["single_camera"]["fake_threshold"]),
        match_threshold=float(pipe_cfg["single_camera"]["match_threshold"]),
        min_face_confidence=float(pipe_cfg["single_camera"]["min_face_confidence"]),
        min_face_size=int(pipe_cfg["single_camera"]["min_face_size"]),
    )
    return SingleCamPipeline(detector, deepfake, arcface, store, decision, device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-camera face verification.")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--image", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ctx-id", type=int, default=-1)
    args = parser.parse_args()
    pipeline = build_pipeline(args.device, args.checkpoint, args.ctx_id)
    if args.image:
        img = cv2.imread(str(resolve_project_path(args.image)), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Unreadable image: {args.image}")
        result = pipeline.run(img, args.user_id)
        print(json.dumps(result.to_dict(), indent=2))
    else:
        pipeline.run_live(args.user_id, args.camera)


if __name__ == "__main__":
    main()
