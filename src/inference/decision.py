from __future__ import annotations

from src.inference.result import VerificationResult


class DecisionEngine:
    def __init__(
        self,
        fake_threshold: float = 0.5,
        match_threshold: float = 0.60,
        min_face_confidence: float = 0.8,
        min_face_size: int = 40,
    ) -> None:
        self.fake_threshold = fake_threshold
        self.match_threshold = match_threshold
        self.min_face_confidence = min_face_confidence
        self.min_face_size = min_face_size

    def decide(
        self,
        *,
        fake_score: float,
        match_score: float,
        user_id: str | None,
        face_detected: bool,
        face_confidence: float = 0.0,
        face_size: int = 0,
        latency_ms: float = 0.0,
        no_template: bool = False,
    ) -> VerificationResult:
        if not face_detected:
            return VerificationResult(
                "RETRY",
                0.0,
                0.0,
                user_id,
                False,
                latency_ms,
                "no_face",
                face_confidence,
                face_size,
            )
        if face_confidence < self.min_face_confidence:
            return VerificationResult(
                "RETRY",
                fake_score,
                match_score,
                user_id,
                True,
                latency_ms,
                f"low_face_confidence_{face_confidence:.3f}",
                face_confidence,
                face_size,
            )
        if face_size < self.min_face_size:
            return VerificationResult(
                "RETRY",
                fake_score,
                match_score,
                user_id,
                True,
                latency_ms,
                f"small_face_{face_size}px",
                face_confidence,
                face_size,
            )
        if fake_score > self.fake_threshold:
            return VerificationResult(
                "REJECT_FAKE",
                fake_score,
                match_score,
                user_id,
                True,
                latency_ms,
                f"fake_score_{fake_score:.3f}_above_{self.fake_threshold:.3f}",
                face_confidence,
                face_size,
            )
        if no_template:
            return VerificationResult(
                "REJECT_IDENTITY",
                fake_score,
                match_score,
                user_id,
                True,
                latency_ms,
                "no_enrollment_template",
                face_confidence,
                face_size,
            )
        if match_score >= self.match_threshold:
            return VerificationResult(
                "ACCEPT",
                fake_score,
                match_score,
                user_id,
                True,
                latency_ms,
                "matched",
                face_confidence,
                face_size,
            )
        return VerificationResult(
            "REJECT_IDENTITY",
            fake_score,
            match_score,
            user_id,
            True,
            latency_ms,
            f"match_score_{match_score:.3f}_below_{self.match_threshold:.3f}",
            face_confidence,
            face_size,
        )
