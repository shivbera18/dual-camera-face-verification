"""VerificationResult + per-view result containers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class VerificationResult:
    decision: str
    fake_score: float
    match_score: float
    user_id: Optional[str]
    face_detected: bool
    latency_ms: float
    reason: str
    thresholds: dict = field(default_factory=dict)
    model_versions: dict = field(default_factory=dict)
    mode: str = "single_camera"
    per_view: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ViewResult:
    face_detected: bool
    fake_score: float
    match_score: float
    latency_ms: float
    bbox: Optional[list[float]] = None

    def to_dict(self) -> dict:
        return asdict(self)
