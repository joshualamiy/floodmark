"""Per-camera alert decisions on top of the raw model status.

The raw model status is stored as-is in ``predictions.status``. The alert status
shown on the map and used for emails is decided here from the camera's own
history in the database, so it survives worker restarts and is shared between
worker processes:

1. Blocklisted cameras never alert.
2. A frame byte-identical to the previous capture (a frozen feed) is not counted.
3. A water frame only counts when its flood score is clearly above the camera's
   own recent median, so a scene that always looks a bit like water cannot
   confirm itself.
4. ``n`` counted water frames in a row are needed before ``flooded``; fewer
   when it is raining nearby (storm mode).
5. A confirmed flood is only alerted when it has actually rained nearby.

A pending decision tells the worker to re-poll the camera within a minute
instead of waiting for the next five-minute cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

WATER_STATUSES = frozenset({"wet", "flooded"})


@dataclass(frozen=True, slots=True)
class HistoryFrame:
    """One earlier capture of the same camera, newest first in a history list."""

    sha256: str | None
    status: str
    alert_status: str | None
    flood_score: float


@dataclass(frozen=True, slots=True)
class AlertDecision:
    status: str
    note: str | None = None
    # True when the streak is complete and only the rain check remains.
    confirmed_flood: bool = False
    # True while confirmation is in progress, so the camera is worth re-polling soon.
    pending: bool = False


def is_frozen(sha256: str | None, previous_sha256: str | None) -> bool:
    return bool(sha256) and sha256 == previous_sha256


def water_threshold(baseline: float | None, margin: float) -> float:
    return 0.0 if baseline is None else baseline + margin


def counts_as_water(status: str, flood_score: float, threshold: float) -> bool:
    return status in WATER_STATUSES and flood_score >= threshold


def water_streak(history: list[HistoryFrame], threshold: float) -> int:
    """Consecutive counted water frames before the current one, skipping frozen repeats."""
    streak = 0
    for index, frame in enumerate(history):
        older = history[index + 1] if index + 1 < len(history) else None
        if older is not None and is_frozen(frame.sha256, older.sha256):
            continue
        if counts_as_water(frame.status, frame.flood_score, threshold):
            streak += 1
        else:
            break
    return streak


def decide_alert(
    *,
    camera_id: str,
    status: str,
    flood_score: float,
    sha256: str | None,
    history: list[HistoryFrame],
    baseline: float | None,
    streak_frames: int,
    baseline_margin: float,
    blocklist: frozenset[str] | set[str],
) -> AlertDecision:
    if status == "dry":
        return AlertDecision("dry")
    if camera_id in blocklist:
        return AlertDecision("wet", "camera blocklisted; flood alerts suppressed")

    previous = history[0] if history else None
    if previous is not None and is_frozen(sha256, previous.sha256):
        carried = previous.alert_status or "wet"
        return AlertDecision(
            carried,
            "frame identical to the previous capture; not counted toward confirmation",
            pending=carried != "flooded",
        )

    threshold = water_threshold(baseline, baseline_margin)
    if not counts_as_water(status, flood_score, threshold):
        return AlertDecision(
            "wet",
            f"possible flooding; flood score {flood_score:.2f} is within this camera's normal range "
            f"(median {baseline:.2f}); not counted toward confirmation",
        )

    streak = water_streak(history, threshold) + 1
    if streak < streak_frames:
        return AlertDecision(
            "wet", f"potential flooding; water frame {streak}/{streak_frames}; awaiting confirmation", pending=True
        )
    if status == "wet":
        return AlertDecision("wet")
    return AlertDecision("flooded", confirmed_flood=True)


def apply_rain_gate(
    decision: AlertDecision,
    rain_mm: float | None,
    *,
    min_rain_mm: float,
    window_hours: int,
    streak_frames: int,
) -> AlertDecision:
    """Downgrade a confirmed flood to ``wet`` when it has not rained. Unknown rain fails open."""
    if not decision.confirmed_flood:
        return decision
    if rain_mm is None:
        return replace(decision, note="rain check unavailable; alert allowed")
    if rain_mm < min_rain_mm:
        return AlertDecision(
            "wet",
            f"flood signal in {streak_frames} frames but only {rain_mm:.1f} mm of rain in the last "
            f"{window_hours} h; alert suppressed",
        )
    return replace(decision, note=f"{rain_mm:.1f} mm of rain in the last {window_hours} h")
