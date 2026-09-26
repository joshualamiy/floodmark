"""Per-camera temporal smoothing: N consecutive raw "flooded" frames before
reporting flooded, plus a blocklist for cameras with known false-alarm views
(e.g. 511GA camera 11372 -- reports/EVALUATION.md).
"""
from __future__ import annotations

from typing import NamedTuple


class SmoothedStatus(NamedTuple):
    status: str
    note: str | None = None


class TemporalSmoother:
    def __init__(self, n: int = 3, blocklist=()):
        self.n = n
        self.blocklist = set(blocklist)
        self._streaks: dict[str, int] = {}

    @staticmethod
    def _raw_status(prediction_or_status) -> str:
        return getattr(prediction_or_status, "status", prediction_or_status)

    def update(self, camera_id: str, prediction_or_status) -> SmoothedStatus:
        status = self._raw_status(prediction_or_status)
        streak = self._streaks.get(camera_id, 0)
        streak = streak + 1 if status == "flooded" else 0
        self._streaks[camera_id] = streak

        if camera_id in self.blocklist:
            if status == "dry":
                return SmoothedStatus("dry")
            return SmoothedStatus("wet", "camera blocklisted; flood alerts suppressed")
        if status == "flooded" and streak < self.n:
            return SmoothedStatus("wet", f"flooded frame {streak}/{self.n}; awaiting confirmation")
        return SmoothedStatus(status)

    def skip(self, camera_id: str) -> None:
        # e.g. camera moved: registers the camera but doesn't touch its streak
        self._streaks.setdefault(camera_id, 0)

    def to_dict(self) -> dict:
        return {"n": self.n, "blocklist": sorted(self.blocklist), "streaks": dict(self._streaks)}

    @classmethod
    def from_dict(cls, d: dict) -> TemporalSmoother:
        obj = cls(n=d.get("n", 3), blocklist=d.get("blocklist", ()))
        obj._streaks = dict(d.get("streaks", {}))
        return obj
