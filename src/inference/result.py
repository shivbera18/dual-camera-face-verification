from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class VerificationResult:
    decision: str
    fake_score: float
    match_score: float
    user_id: str | None
    face_detected: bool
    latency_ms: float
    reason: str
    face_confidence: float = 0.0
    face_size: int = 0

    def to_dict(self) -> dict[str, float | str | bool | None | int]:
        return asdict(self)
