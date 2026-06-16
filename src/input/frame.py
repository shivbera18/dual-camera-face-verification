from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    img: np.ndarray
    timestamp_ms: float
    camera_id: int
    frame_index: int
