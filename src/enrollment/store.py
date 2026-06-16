from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.models.arcface import ArcFaceExtractor
from src.utils.config import resolve_project_path


@dataclass
class EnrollmentRecord:
    user_id: str
    embedding_template: np.ndarray
    num_samples: int
    created_at: str
    model_name: str = "buffalo_l"
    model_version: str = "insightface"

    def to_serializable(self) -> dict[str, Any]:
        data = asdict(self)
        data["embedding_template"] = self.embedding_template.astype(np.float32)
        return data


class EnrollmentStore:
    def __init__(
        self, db_path: str | Path = "artifacts/models/enrollment_db.pkl"
    ) -> None:
        self.db_path = resolve_project_path(db_path)
        self.records: dict[str, EnrollmentRecord] = {}
        self.load()

    def enroll(
        self, user_id: str, embeddings: list[np.ndarray], model_name: str = "buffalo_l"
    ) -> EnrollmentRecord:
        if not embeddings:
            raise ValueError("At least one embedding is required for enrollment")
        stacked = np.stack(
            [emb / (np.linalg.norm(emb) + 1e-12) for emb in embeddings]
        ).astype(np.float32)
        template = stacked.mean(axis=0)
        template = template / (np.linalg.norm(template) + 1e-12)
        record = EnrollmentRecord(
            user_id=user_id,
            embedding_template=template,
            num_samples=len(embeddings),
            created_at=datetime.now(timezone.utc).isoformat(),
            model_name=model_name,
        )
        self.records[user_id] = record
        self.save()
        return record

    def get_template(self, user_id: str) -> np.ndarray | None:
        record = self.records.get(user_id)
        return None if record is None else record.embedding_template

    def list_users(self) -> list[str]:
        return sorted(self.records.keys())

    def delete_user(self, user_id: str) -> None:
        self.records.pop(user_id, None)
        self.save()

    def verify(self, user_id: str, embedding: np.ndarray) -> float | None:
        template = self.get_template(user_id)
        if template is None:
            return None
        return ArcFaceExtractor.similarity(template, embedding)

    def save(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.db_path.open("wb") as fh:
            pickle.dump(
                {uid: record.to_serializable() for uid, record in self.records.items()},
                fh,
            )

    def load(self) -> None:
        if not self.db_path.exists():
            self.records = {}
            return
        with self.db_path.open("rb") as fh:
            raw = pickle.load(fh)
        self.records = {
            uid: EnrollmentRecord(
                user_id=data["user_id"],
                embedding_template=np.asarray(
                    data["embedding_template"], dtype=np.float32
                ),
                num_samples=int(data["num_samples"]),
                created_at=data["created_at"],
                model_name=data.get("model_name", "buffalo_l"),
                model_version=data.get("model_version", "insightface"),
            )
            for uid, data in raw.items()
        }
