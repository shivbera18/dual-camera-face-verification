from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.enrollment.store import EnrollmentStore
from src.face.aligner import align_face
from src.face.detector import FaceDetector
from src.models.arcface import ArcFaceExtractor
from src.utils.config import get_model_config, resolve_project_path
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def enroll_from_images(
    user_id: str,
    image_paths: list[str | Path],
    detector: FaceDetector,
    extractor: ArcFaceExtractor,
    store: EnrollmentStore,
) -> int:
    embeddings: list[np.ndarray] = []
    for path_like in image_paths:
        path = resolve_project_path(path_like)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            LOGGER.warning("Skipping unreadable image: %s", path)
            continue
        face = detector.detect_best(img)
        if face is None:
            LOGGER.warning("Skipping image with no detected face: %s", path)
            continue
        crop = align_face(img, face.landmarks, (224, 224))
        emb = extractor.get_embedding(crop)
        if emb is None:
            LOGGER.warning("Skipping image with no ArcFace embedding: %s", path)
            continue
        embeddings.append(emb)
    store.enroll(user_id, embeddings, model_name=extractor.model_pack)
    LOGGER.info("Enrolled user %s with %d samples", user_id, len(embeddings))
    return len(embeddings)


def enroll_from_webcam(
    user_id: str,
    n_samples: int,
    detector: FaceDetector,
    extractor: ArcFaceExtractor,
    store: EnrollmentStore,
    camera_index: int = 0,
) -> int:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")
    embeddings: list[np.ndarray] = []
    try:
        while len(embeddings) < n_samples:
            ok, frame = cap.read()
            if not ok:
                continue
            face = detector.detect_best(frame)
            view = frame.copy()
            if face is not None:
                x1, y1, x2, y2 = face.bbox.astype(int)
                cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                view,
                f"Samples: {len(embeddings)}/{n_samples} - press SPACE",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("enroll", view)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == 32 and face is not None:
                crop = align_face(frame, face.landmarks, (224, 224))
                emb = extractor.get_embedding(crop)
                if emb is not None:
                    embeddings.append(emb)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    store.enroll(user_id, embeddings, model_name=extractor.model_pack)
    return len(embeddings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll a user from images or webcam.")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--images", nargs="*", default=[])
    parser.add_argument("--webcam", action="store_true")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--ctx-id", type=int, default=-1)
    args = parser.parse_args()

    model_cfg = get_model_config()
    store = EnrollmentStore(args.db or model_cfg["arcface"]["enrollment_db"])
    detector = FaceDetector(
        model_name=model_cfg["arcface"]["model"],
        min_confidence=float(model_cfg["retinaface"]["detection_threshold"]),
        ctx_id=args.ctx_id,
    )
    extractor = ArcFaceExtractor(
        model_pack=model_cfg["arcface"]["model"], ctx_id=args.ctx_id
    )
    if args.webcam:
        count = enroll_from_webcam(
            args.user_id,
            int(args.samples or model_cfg["arcface"]["enrollment_samples"]),
            detector,
            extractor,
            store,
            args.camera,
        )
    else:
        count = enroll_from_images(
            args.user_id, args.images, detector, extractor, store
        )
    print(
        f"Enrolled {args.user_id} with {count} embeddings. Users: {store.list_users()}"
    )


if __name__ == "__main__":
    main()
