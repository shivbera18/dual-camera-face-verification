"""Decision engine applying fake and identity thresholds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.inference.result import VerificationResult, ViewResult
from src.utils.config import get_pipeline_config


@dataclass
class _Scores:
    fake: float
    match: float
    face_detected: bool


class DecisionEngine:
    DECISIONS = ("ACCEPT", "REJECT_FAKE", "REJECT_IDENTITY", "RETRY")

    def __init__(
        self,
        fake_threshold: Optional[float] = None,
        match_threshold: Optional[float] = None,
        mode: str = "single_camera",
    ) -> None:
        cfg = get_pipeline_config()
        section = cfg.get(mode, cfg.get("single_camera", {}))
        self.fake_threshold = float(
            fake_threshold if fake_threshold is not None else section.get("fake_threshold", 0.5)
        )
        self.match_threshold = float(
            match_threshold if match_threshold is not None else section.get("match_threshold", 0.60)
        )
        self.mode = mode
        self.fusion_fake = cfg.get("dual_camera", {}).get("fusion", {}).get("fake_score", "mean")
        self.fusion_match = cfg.get("dual_camera", {}).get("fusion", {}).get("match_score", "max")

    def _fuse(self, values: list[float], mode: str) -> float:
        if not values:
            return 0.0
        if mode == "max":
            return max(values)
        if mode == "min":
            return min(values)
        return sum(values) / len(values)

    def decide(
        self,
        fake_score: float,
        match_score: float,
        face_detected: bool,
        user_id: Optional[str] = None,
    ) -> VerificationResult:
        thresholds = {
            "fake_threshold": self.fake_threshold,
            "match_threshold": self.match_threshold,
        }
        if not face_detected:
            return VerificationResult(
                decision="RETRY",
                fake_score=float(fake_score),
                match_score=float(match_score),
                user_id=user_id,
                face_detected=False,
                latency_ms=0.0,
                reason="no_face_detected",
                thresholds=thresholds,
                mode=self.mode,
            )
        if fake_score >= self.fake_threshold:
            return VerificationResult(
                decision="REJECT_FAKE",
                fake_score=float(fake_score),
                match_score=float(match_score),
                user_id=user_id,
                face_detected=True,
                latency_ms=0.0,
                reason="fake_above_threshold",
                thresholds=thresholds,
                mode=self.mode,
            )
        if user_id is not None and match_score < self.match_threshold:
            return VerificationResult(
                decision="REJECT_IDENTITY",
                fake_score=float(fake_score),
                match_score=float(match_score),
                user_id=user_id,
                face_detected=True,
                latency_ms=0.0,
                reason="identity_below_threshold",
                thresholds=thresholds,
                mode=self.mode,
            )
        return VerificationResult(
            decision="ACCEPT",
            fake_score=float(fake_score),
            match_score=float(match_score),
            user_id=user_id,
            face_detected=True,
            latency_ms=0.0,
            reason="all_checks_passed",
            thresholds=thresholds,
            mode=self.mode,
        )

    def decide_dual(
        self,
        views: dict[str, ViewResult],
        user_id: Optional[str] = None,
    ) -> VerificationResult:
        thresholds = {
            "fake_threshold": self.fake_threshold,
            "match_threshold": self.match_threshold,
            "fusion_fake": self.fusion_fake,
            "fusion_match": self.fusion_match,
        }
        detected = [v for v in views.values() if v.face_detected]
        if not detected:
            return VerificationResult(
                decision="RETRY",
                fake_score=0.0,
                match_score=0.0,
                user_id=user_id,
                face_detected=False,
                latency_ms=0.0,
                reason="no_face_any_view",
                thresholds=thresholds,
                mode="dual_camera",
                per_view={k: v.to_dict() for k, v in views.items()},
            )
        fake_scores = [v.fake_score for v in detected]
        match_scores = [v.match_score for v in detected]
        fused_fake = self._fuse(fake_scores, self.fusion_fake)
        fused_match = self._fuse(match_scores, self.fusion_match)
        result = self.decide(fused_fake, fused_match, True, user_id=user_id)
        result.mode = "dual_camera"
        result.per_view = {k: v.to_dict() for k, v in views.items()}
        return result
