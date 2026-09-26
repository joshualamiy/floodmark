from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime


@dataclass(frozen=True, slots=True)
class CaptureJob:
    camera_id: str
    source: str
    source_camera_id: str
    source_view_id: str
    scheduled_at: datetime
    capture_id: str
    # optional so jobs queued before coordinates were added still deserialize
    latitude: float | None = None
    longitude: float | None = None
    # how many fast re-polls preceded this capture; 0 for a regular five-minute capture
    fast_poll: int = 0

    def follow_up(self, run_at: datetime) -> "CaptureJob":
        """The next fast re-poll of this camera, keyed on its own run time."""
        return replace(
            self,
            scheduled_at=run_at,
            capture_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.capture_id}:fast:{self.fast_poll + 1}")),
            fast_poll=self.fast_poll + 1,
        )

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
    heatmap_status: str = "disabled"
    heatmap_note: str | None = None
    alert_status: str | None = None
    alert_note: str | None = None
