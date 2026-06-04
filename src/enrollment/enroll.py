"""Enrollment flows: from images, from a webcam session."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.enrollment.store import EnrollmentStore
from src.face.aligner import align_face
from src.face.detector import FaceDetector
from src.models.arcface import ArcFaceExtractor
from src.utils.config import get_model_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img


def _crop_for_arcface(detector: FaceDetector, img: np.ndarray) -> np.ndarray:
    best = detector.detect_best(img)
    if best is None:
        raise RuntimeError("no face detected")
    return align_face(img, best.landmarks, output_size=(112, 112))


def enroll_from_images(
    user_id: str,
    image_paths: list[str | Path],
    detector: FaceDetector,
    extractor: ArcFaceExtractor,
    store: EnrollmentStore,
) -> int:
    embeddings: list[np.ndarray] = []
    for p in image_paths:
        path = Path(p)
        try:
            img = _read_image(path)
            crop = _crop_for_arcface(detector, img)
            emb = extractor.get_embedding(crop)
        except Exception as exc:
            logger.warning("skip %s: %s", path, exc)
            continue
        embeddings.append(emb)
    if not embeddings:
        raise RuntimeError(f"no usable faces in {len(image_paths)} images")
    store.enroll(user_id, embeddings)
    return len(embeddings)


def enroll_from_webcam(
    user_id: str,
    n_samples: int,
    detector: FaceDetector,
    extractor: ArcFaceExtractor,
    store: EnrollmentStore,
    camera_index: int = 0,
    window_name: str = "enroll",
    show_preview: bool = True,
) -> int:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {camera_index}")
    cfg = get_model_config().get("arcface", {})
    target = int(cfg.get("enrollment_samples", n_samples))
    if n_samples > 0:
        target = n_samples
    embeddings: list[np.ndarray] = []
    captured = 0
    last_capture_ts = 0.0
    try:
        while captured < target:
            ok, frame = cap.read()
            if not ok:
                break
            best = detector.detect_best(frame)
            now = time.time()
            if best is not None and (now - last_capture_ts) > 0.5:
                try:
                    crop = align_face(frame, best.landmarks, output_size=(112, 112))
                    emb = extractor.get_embedding(crop)
                    embeddings.append(emb)
                    captured += 1
                    last_capture_ts = now
                    logger.info("captured %d/%d", captured, target)
                except Exception as exc:
                    logger.warning("capture failed: %s", exc)
            if show_preview:
                disp = frame.copy()
                if best is not None:
                    x1, y1, x2, y2 = best.bbox.astype(int)
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    disp,
                    f"Enroll {user_id}: {captured}/{target}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(window_name, disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if show_preview:
            cv2.destroyWindow(window_name)
    if not embeddings:
        raise RuntimeError("no embeddings captured from webcam")
    store.enroll(user_id, embeddings)
    return len(embeddings)
