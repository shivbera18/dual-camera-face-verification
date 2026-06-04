"""Two-threaded camera capture with timestamp-based frame pairing."""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.input.frame import Frame
from src.utils.config import get_pipeline_config, resolve
from src.utils.logger import get_logger

logger = get_logger(__name__)


class _CameraThread(threading.Thread):
    def __init__(self, camera_id: int, source: int, width: int, height: int, buffer: deque, stop_event: threading.Event) -> None:
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.source = source
        self.width = width
        self.height = height
        self.buffer = buffer
        self.stop_event = stop_event
        self.frame_index = 0
        self.cap: Optional[cv2.VideoCapture] = None

    def run(self) -> None:
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error("camera %d failed to open (source=%s)", self.camera_id, self.source)
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        logger.info("camera thread %d started (source=%s)", self.camera_id, self.source)
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            ts_ms = time.time() * 1000.0
            self.frame_index += 1
            self.buffer.append(Frame(img=frame, timestamp_ms=ts_ms, camera_id=self.camera_id, frame_index=self.frame_index))
        if self.cap is not None:
            self.cap.release()
        logger.info("camera thread %d stopped", self.camera_id)


class DualCameraCapture:
    def __init__(
        self,
        left_idx: int = 0,
        right_idx: int = 1,
        sync_delta_ms: int = 50,
        width: int = 1280,
        height: int = 720,
        buffer_size: int = 16,
    ) -> None:
        cfg = get_pipeline_config().get("dual_camera", {})
        if sync_delta_ms is None:
            sync_delta_ms = int(cfg.get("sync_delta_ms", 50))
        self.left_idx = int(left_idx)
        self.right_idx = int(right_idx)
        self.sync_delta_ms = int(sync_delta_ms)
        self.width = width
        self.height = height
        self.buffer_size = buffer_size
        self._left_buf: deque = deque(maxlen=buffer_size)
        self._right_buf: deque = deque(maxlen=buffer_size)
        self._stop_event = threading.Event()
        self._left_thread: Optional[_CameraThread] = None
        self._right_thread: Optional[_CameraThread] = None

    def start(self) -> None:
        self._stop_event.clear()
        self._left_thread = _CameraThread(0, self.left_idx, self.width, self.height, self._left_buf, self._stop_event)
        self._right_thread = _CameraThread(1, self.right_idx, self.width, self.height, self._right_buf, self._stop_event)
        self._left_thread.start()
        self._right_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        for t in (self._left_thread, self._right_thread):
            if t is not None:
                t.join(timeout=2.0)
        self._left_thread = None
        self._right_thread = None

    def _drop_stale(self) -> None:
        now_ms = time.time() * 1000.0
        while self._left_buf and (now_ms - self._left_buf[0].timestamp_ms) > 1000.0:
            self._left_buf.popleft()
        while self._right_buf and (now_ms - self._right_buf[0].timestamp_ms) > 1000.0:
            self._right_buf.popleft()

    def get_pair(self) -> Optional[tuple[Frame, Frame]]:
        self._drop_stale()
        if not self._left_buf or not self._right_buf:
            return None
        best: Optional[tuple[Frame, Frame, float]] = None
        for lf in self._left_buf:
            for rf in self._right_buf:
                delta = abs(lf.timestamp_ms - rf.timestamp_ms)
                if delta > self.sync_delta_ms:
                    continue
                if best is None or delta < best[2]:
                    best = (lf, rf, delta)
        if best is None:
            return None
        lf, rf, _ = best
        try:
            self._left_buf.remove(lf)
        except ValueError:
            pass
        try:
            self._right_buf.remove(rf)
        except ValueError:
            pass
        return lf, rf

    def save_pair(
        self,
        left: Frame,
        right: Frame,
        metadata: dict,
        output_dir: str | Path,
    ) -> dict:
        out_dir = resolve(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        session = metadata.get("session_id", "session")
        pair_idx = int(metadata.get("pair_index", 0))
        left_path = out_dir / f"{session}_{pair_idx}_left_{int(left.timestamp_ms)}.jpg"
        right_path = out_dir / f"{session}_{pair_idx}_right_{int(right.timestamp_ms)}.jpg"
        cv2.imwrite(str(left_path), left.img)
        cv2.imwrite(str(right_path), right.img)
        meta_path = out_dir / f"{session}_{pair_idx}.json"
        full_meta = {
            **metadata,
            "left_image_path": str(left_path.relative_to(resolve("."))),
            "right_image_path": str(right_path.relative_to(resolve("."))),
            "left_timestamp": left.timestamp_ms,
            "right_timestamp": right.timestamp_ms,
            "sync_delta_ms": abs(left.timestamp_ms - right.timestamp_ms),
            "camera_left_index": self.left_idx,
            "camera_right_index": self.right_idx,
        }
        with meta_path.open("w") as f:
            json.dump(full_meta, f, indent=2, default=str)
        return full_meta
