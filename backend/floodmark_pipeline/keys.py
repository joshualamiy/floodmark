from __future__ import annotations

from datetime import datetime, timezone

def capture_slot(value: datetime) -> datetime:
    value = value.astimezone(timezone.utc)
    return value.replace(minute=value.minute - value.minute % 5, second=0, microsecond=0)


def capture_job_id(source: str, camera_id: str, scheduled_at: datetime) -> str:
    slot = capture_slot(scheduled_at)
    return f"capture:{source.lower()}:{camera_id}:{slot.strftime('%Y-%m-%dT%H:%MZ')}"


def fast_poll_job_id(source: str, camera_id: str, run_at: datetime) -> str:
    # fast re-polls run between slots, so they get their own ID at second resolution
    return f"{capture_job_id(source, camera_id, run_at)}:fast:{run_at.astimezone(timezone.utc):%H%M%S}"


def _timestamp(captured_at: datetime, seconds: bool) -> str:
    timestamp = captured_at.astimezone(timezone.utc)
    return f"{timestamp:%Y%m%dT%H%M%SZ}" if seconds else f"{timestamp:%Y%m%dT%H%MZ}"


def object_key(kind: str, view_id: str, captured_at: datetime, seconds: bool = False) -> str:
    if kind not in {"captures", "heatmaps"}:
        raise ValueError("kind must be captures or heatmaps")
    if not view_id or "/" in view_id:
        raise ValueError("view ID must be a non-empty path segment")
    extension = "jpg" if kind == "captures" else "png"
    return f"{kind}/{view_id}/{_timestamp(captured_at, seconds)}.{extension}"


def skipped_key(view_id: str, captured_at: datetime, seconds: bool = False) -> str:
    if not view_id or "/" in view_id:
        raise ValueError("view ID must be a non-empty path segment")
    return f"skipped/{view_id}/{_timestamp(captured_at, seconds)}.jpg"
