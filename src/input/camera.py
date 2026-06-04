"""Single OpenCV camera wrapper."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator, Optional, Union

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CameraInput:
    def __init__(self, source: Union[int, str, Path] = 0, width: int = 1280, height: int = 720) -> None:
        self.source = int(source) if isinstance(source, (int, np.integer)) else str(source)
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera source: {self.source}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        logger.info("camera opened: source=%s %dx%d", self.source, self.width, self.height)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("camera closed: source=%s", self.source)

    def __enter__(self) -> "CameraInput":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read(self) -> tuple[bool, np.ndarray, float]:
        if self.cap is None:
            raise RuntimeError("camera not open")
        ok, frame = self.cap.read()
        ts = time.time()
        return ok, frame, ts

    def frames(self) -> Iterator[tuple[np.ndarray, float]]:
        if self.cap is None:
            self.open()
        assert self.cap is not None
        while True:
            ok, frame, ts = self.read()
            if not ok:
                break
            yield frame, ts
