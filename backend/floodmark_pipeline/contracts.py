from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class CaptureJob:
    camera_id: str
    source: str
    source_camera_id: str
    source_view_id: str
    scheduled_at: datetime
    capture_id: str

    def payload(self) -> dict[str, str]:
        data = asdict(self)
        data["scheduled_at"] = self.scheduled_at.isoformat()
        return data

    @classmethod
    def from_payload(cls, payload: dict[str, str]) -> "CaptureJob":
        return cls(**{**payload, "scheduled_at": datetime.fromisoformat(payload["scheduled_at"])})


@dataclass(frozen=True, slots=True)
class Prediction:
    model_version: dict[str, str]
    status: str
    confidence: float
    stage_a_probabilities: dict[str, float]
    stage_b_probabilities: dict[str, float]
    stage_probabilities: dict[str, float]
    thresholds: dict[str, float]
    note: str | None
    heatmap_bytes: bytes | None
