from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

import cv2

from src.input.frame import Frame
from src.utils.config import resolve_project_path


class DualCameraCapture:
    def __init__(
        self,
        left_idx: int = 0,
        right_idx: int = 1,
        sync_delta_ms: float = 50,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        self.left_idx = left_idx
        self.right_idx = right_idx
        self.sync_delta_ms = sync_delta_ms
        self.width = width
        self.height = height
        self.fps = fps
        self.left_q: queue.Queue[Frame] = queue.Queue(maxsize=4)
        self.right_q: queue.Queue[Frame] = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def _open_camera(self, idx: int) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _reader(self, camera_id: int, index: int) -> None:
        cap = self._open_camera(index)
        if not cap.isOpened():
            return
        q = self.left_q if camera_id == 0 else self.right_q
        frame_index = 0
        while not self._stop.is_set():
            ok, img = cap.read()
            if not ok:
                continue
            frame = Frame(
                img=img,
                timestamp_ms=time.monotonic() * 1000.0,
                camera_id=camera_id,
                frame_index=frame_index,
            )
            try:
                q.put_nowait(frame)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                q.put_nowait(frame)
            frame_index += 1
        cap.release()

    def start(self) -> None:
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._reader, args=(0, self.left_idx), daemon=True),
            threading.Thread(
                target=self._reader, args=(1, self.right_idx), daemon=True
            ),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2.0)

    def get_pair(self) -> tuple[Frame, Frame] | None:
        if self.left_q.empty() or self.right_q.empty():
            return None
        left = self.left_q.get()
        candidates = [self.right_q.get()]
        while not self.right_q.empty():
            candidates.append(self.right_q.get())
        right = min(
            candidates, key=lambda frame: abs(frame.timestamp_ms - left.timestamp_ms)
        )
        if abs(left.timestamp_ms - right.timestamp_ms) > self.sync_delta_ms:
            return None
        return left, right

    def save_pair(
        self, left: Frame, right: Frame, metadata: dict, output_dir: str | Path
    ) -> None:
        out = resolve_project_path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        session_id = metadata.get("session_id", "session")
        pair_index = int(metadata.get("pair_index", 0))
        left_path = (
            out / f"{session_id}_{pair_index:06d}_left_{int(left.timestamp_ms)}.jpg"
        )
        right_path = (
            out / f"{session_id}_{pair_index:06d}_right_{int(right.timestamp_ms)}.jpg"
        )
        cv2.imwrite(str(left_path), left.img)
        cv2.imwrite(str(right_path), right.img)
        metadata = dict(metadata)
        metadata.update(
            {
                "left_image_path": str(left_path),
                "right_image_path": str(right_path),
                "left_timestamp": left.timestamp_ms,
                "right_timestamp": right.timestamp_ms,
                "sync_delta_ms": abs(left.timestamp_ms - right.timestamp_ms),
            }
        )
        with (out / f"{session_id}_{pair_index:06d}.json").open(
            "w", encoding="utf-8"
        ) as fh:
            json.dump(metadata, fh, indent=2)
