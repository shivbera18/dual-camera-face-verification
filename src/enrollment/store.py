"""Persistent enrollment database (one template per user)."""
from __future__ import annotations

import datetime as _dt
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.config import get_model_config, resolve
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EnrollmentRecord:
    user_id: str
    embedding_template: np.ndarray
    num_samples: int
    created_at: str
    model_name: str = "buffalo_l"
    model_version: str = "insightface-1.0"
    raw_embeddings: list[np.ndarray] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["embedding_template"] = self.embedding_template.tolist()
        d["raw_embeddings"] = [e.tolist() for e in self.raw_embeddings]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EnrollmentRecord":
        return cls(
            user_id=d["user_id"],
            embedding_template=np.asarray(d["embedding_template"], dtype=np.float32),
            num_samples=int(d.get("num_samples", 0)),
            created_at=d.get("created_at", _dt.datetime.utcnow().isoformat()),
            model_name=d.get("model_name", "buffalo_l"),
            model_version=d.get("model_version", "insightface-1.0"),
            raw_embeddings=[np.asarray(e, dtype=np.float32) for e in d.get("raw_embeddings", [])],
        )


class EnrollmentStore:
    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        cfg = get_model_config().get("arcface", {})
        self.db_path = resolve(db_path or cfg.get("enrollment_db", "artifacts/models/enrollment_db.pkl"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, EnrollmentRecord] = {}
        if self.db_path.exists():
            self.load()

    def enroll(self, user_id: str, embeddings: list[np.ndarray]) -> EnrollmentRecord:
        if not embeddings:
            raise ValueError("no embeddings provided")
        arr = np.stack([np.asarray(e, dtype=np.float32) for e in embeddings], axis=0)
        template = arr.mean(axis=0)
        template = template / (np.linalg.norm(template) + 1e-12)
        record = EnrollmentRecord(
            user_id=user_id,
            embedding_template=template.astype(np.float32),
            num_samples=int(arr.shape[0]),
            created_at=_dt.datetime.utcnow().isoformat(),
            raw_embeddings=[e.astype(np.float32) for e in embeddings],
        )
        self.records[user_id] = record
        self.save()
        logger.info("enrolled user_id=%s n_samples=%d", user_id, record.num_samples)
        return record

    def get_template(self, user_id: str) -> Optional[np.ndarray]:
        rec = self.records.get(user_id)
        return None if rec is None else rec.embedding_template

    def list_users(self) -> list[str]:
        return sorted(self.records.keys())

    def delete_user(self, user_id: str) -> bool:
        if user_id in self.records:
            del self.records[user_id]
            self.save()
            return True
        return False

    def save(self) -> None:
        payload = {uid: rec.to_dict() for uid, rec in self.records.items()}
        with self.db_path.open("wb") as f:
            pickle.dump(payload, f)
        logger.info("enrollment db saved: %s users=%d", self.db_path, len(self.records))

    def load(self) -> None:
        with self.db_path.open("rb") as f:
            payload = pickle.load(f)
        self.records = {uid: EnrollmentRecord.from_dict(d) for uid, d in payload.items()}
        logger.info("enrollment db loaded: %s users=%d", self.db_path, len(self.records))
