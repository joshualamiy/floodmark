"""Poll 511GA events, flag ones that mention flooding/water, and capture
frames from nearby enabled cameras the moment a new flagged event appears.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from ga511 import snapshot
from ga511.api import get_events
from ga511.cameras import load_atlanta_rows
from ga511.geo import haversine_km
from ga511.paths import (
    CAMERAS_ATLANTA_CSV,
    FLOOD_EVENT_FRAMES_DIR,
    FLOOD_EVENTS_LOG,
    SEEN_EVENTS_PATH,
    ensure_dirs,
    setup_logging,
)
from ga511.ratelimit import get_default_limiter, locked_file, read_json_fd, write_json_fd

log = setup_logging("ga511_events")

FLOOD_KEYWORDS = ["flooding", "flood", "high water", "water"]
_KEYWORD_PATTERNS = [(kw, re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)) for kw in FLOOD_KEYWORDS]
_FALSE_POSITIVE_PATTERNS = [re.compile(r"\bwater\s*main\b", re.IGNORECASE)]

CAPTURE_RADIUS_KM = 1.0
DEFAULT_POLL_INTERVAL_S = 300.0


def match_flood_keyword(text: str | None) -> str | None:
    """Return the first matching flood-ish keyword (word-boundary,
    case-insensitive), checking longer/more-specific phrases first, or None.
    """
    if not text:
        return None
    for kw, pattern in _KEYWORD_PATTERNS:
        if pattern.search(text):
            return kw
    return None


def is_likely_false_positive(text: str | None) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in _FALSE_POSITIVE_PATTERNS)


def flag_event(event: dict) -> dict | None:
    """Check one raw event dict; return a flag record or None."""
    text = " ".join(
        str(event.get(f) or "") for f in ("Description", "EventType", "Subtype")
    ).strip()
    matched = match_flood_keyword(text)
    if matched is None:
        return None
    return {
        "matched_keyword": matched,
        "likely_false_positive": is_likely_false_positive(text),
        "text": text,
    }


def _event_id(event: dict) -> str:
    return str(event.get("Id") or event.get("ID") or event.get("EventId") or hash(json.dumps(event, sort_keys=True, default=str)))


def _load_seen(path: Path = SEEN_EVENTS_PATH) -> set:
    with locked_file(path) as fd:
        state = read_json_fd(fd, {"seen": []})
    return set(state.get("seen", []))


def _save_seen(seen: set, path: Path = SEEN_EVENTS_PATH) -> None:
    with locked_file(path) as fd:
        write_json_fd(fd, {"seen": sorted(seen)})


def _log_line(line: str, path: Path = FLOOD_EVENTS_LOG) -> None:
    ensure_dirs()
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def _nearby_views(event: dict, views: list[dict], radius_km: float = CAPTURE_RADIUS_KM) -> list[dict]:
    lat, lon = event.get("Latitude"), event.get("Longitude")
    if lat is None or lon is None:
        return []
    lat, lon = float(lat), float(lon)
    nearby = []
    for v in views:
        try:
            d = haversine_km(lat, lon, float(v["lat"]), float(v["lon"]))
        except (TypeError, ValueError, KeyError):
            continue
        if d <= radius_km:
            nearby.append({**v, "distance_km": d})
    return nearby


def capture_event_frames(event: dict, flag: dict, views: list[dict]) -> Path:
    event_id = _event_id(event)
    ts = int(time.time())
    out_dir = FLOOD_EVENT_FRAMES_DIR / f"{event_id}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    nearby = _nearby_views(event, views)
    frame_meta = []
    for v in nearby:
        try:
            result = snapshot.fetch_view(v["view_id"], max_wait_s=60.0)
        except TimeoutError:
            frame_meta.append({**v, "captured": False, "reason": "throttled"})
            continue
        if result.ok:
            fname = f"{v['view_id']}_{result.timestamp_utc}.jpg"
            result.image.convert("RGB").save(out_dir / fname, "JPEG", quality=90)
            frame_meta.append(
                {
                    **v,
                    "captured": True,
                    "path": fname,
                    "timestamp_utc": result.timestamp_utc,
                }
            )
        else:
            frame_meta.append({**v, "captured": False, "reason": result.error_reason})

    event_record = {
        "event": event,
        "flag": flag,
        "captured_at_utc": ts,
        "nearby_view_count": len(nearby),
        "frames": frame_meta,
    }
    with open(out_dir / "event.json", "w", encoding="utf-8") as f:
        json.dump(event_record, f, indent=2, default=str)

    n_captured = sum(1 for f in frame_meta if f.get("captured"))
    _log_line(
        f"{ts} event_id={event_id} keyword={flag['matched_keyword']} "
        f"false_positive_guess={flag['likely_false_positive']} "
        f"nearby_views={len(nearby)} captured={n_captured} dir={out_dir.name}"
    )
    log.info(
        "flagged event %s (%s): captured %d/%d nearby frames -> %s",
        event_id,
        flag["matched_keyword"],
        n_captured,
        len(nearby),
        out_dir,
    )
    return out_dir


def poll_once(views: list[dict] | None = None) -> dict:
    if views is None:
        views = load_atlanta_rows(CAMERAS_ATLANTA_CSV)
    limiter = get_default_limiter()
    events = get_events(limiter=limiter)
    seen = _load_seen()
    new_flagged = 0
    for event in events:
        flag = flag_event(event)
        if flag is None:
            continue
        eid = _event_id(event)
        if eid in seen:
            continue
        seen.add(eid)
        new_flagged += 1
        capture_event_frames(event, flag, views)
    _save_seen(seen)
    return {"n_events": len(events), "n_new_flagged": new_flagged}


def run_forever(interval_s: float = DEFAULT_POLL_INTERVAL_S, stop_flag=None) -> None:
    import threading

    stop_flag = stop_flag or threading.Event()
    views = load_atlanta_rows(CAMERAS_ATLANTA_CSV)
    while not stop_flag.is_set():
        try:
            poll_once(views=views)
        except Exception as e:  # noqa: BLE001
            log.error("event poll failed: %s", e)
        stop_flag.wait(interval_s)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Poll 511GA events for flood keywords")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval-s", type=float, default=DEFAULT_POLL_INTERVAL_S)
    args = parser.parse_args()
    if args.once:
        summary = poll_once()
        print(json.dumps(summary))
    else:
        run_forever(interval_s=args.interval_s)


if __name__ == "__main__":
    _main()
