"""Fetch 511GA camera snapshot images.

Snapshot URLs (https://511ga.org/map/Cctv/{view_id}?t={ts}) carry no API key,
so they don't touch the shared `RateLimiter` directly. We still throttle
them politely: at most one fetch per view every 5 minutes, persisted to disk
so the limit holds across restarts and across processes, plus a soft global
cap of ~1-2 requests/second.
"""
from __future__ import annotations

import argparse
import io
import time
from dataclasses import dataclass

import requests
from PIL import Image

from ga511.paths import LAST_FETCH_PATH, setup_logging
from ga511.ratelimit import get_default_limiter, locked_file, read_json_fd, redact, write_json_fd

SNAPSHOT_URL_TMPL = "https://511ga.org/map/Cctv/{view_id}?t={ts}"
USER_AGENT = "floodmark-research/0.1 (+https://github.com/joshualamiy/floodmark)"

VIEW_MIN_INTERVAL_S = 300.0  # at most one fetch per camera view per 5 minutes
GLOBAL_MIN_INTERVAL_S = 0.7  # ~1.4 req/s overall, within the 1-2 req/s budget

log = setup_logging("ga511_snapshot")


@dataclass
class SnapshotResult:
    ok: bool
    view_id: str
    timestamp_utc: int
    http_status: int | None = None
    content_type: str | None = None
    nbytes: int = 0
    width: int | None = None
    height: int | None = None
    image: Image.Image | None = None
    error_reason: str | None = None  # "http_error" | "non_image" | "tiny_response"


def seconds_since_last_fetch(view_id, state_path=LAST_FETCH_PATH) -> float | None:
    with locked_file(state_path) as fd:
        state = read_json_fd(fd, {})
    last = state.get(str(view_id))
    if last is None:
        return None
    return time.time() - last


def _wait_for_view_slot(
    view_id,
    min_interval: float = VIEW_MIN_INTERVAL_S,
    global_min_interval: float = GLOBAL_MIN_INTERVAL_S,
    state_path=LAST_FETCH_PATH,
    max_wait_s: float = 330.0,
) -> None:
    """Block (cross-process) until this view is allowed to be fetched again,
    then reserve the slot. Raises TimeoutError past `max_wait_s`.
    """
    start = time.time()
    while True:
        with locked_file(state_path) as fd:
            state = read_json_fd(fd, {})
            now = time.time()
            last_view = state.get(str(view_id))
            last_global = state.get("__global__")
            wait = 0.0
            if last_view is not None and now - last_view < min_interval:
                wait = max(wait, min_interval - (now - last_view))
            if last_global is not None and now - last_global < global_min_interval:
                wait = max(wait, global_min_interval - (now - last_global))
            if wait <= 0:
                state[str(view_id)] = now
                state["__global__"] = now
                write_json_fd(fd, state)
                return
        if time.time() - start > max_wait_s:
            raise TimeoutError(f"snapshot: view {view_id} throttled past max_wait_s")
        time.sleep(min(wait, 5.0) + 0.01)


def fetch_view(
    view_id,
    timeout: float = 15.0,
    enforce_interval: bool = True,
    min_interval: float = VIEW_MIN_INTERVAL_S,
    max_wait_s: float = 330.0,
) -> SnapshotResult:
    """Fetch one snapshot for `view_id`, decoding it with PIL to confirm it's
    really an image. Never logs the full URL (it carries no key, but we keep
    the habit consistent).
    """
    if enforce_interval:
        _wait_for_view_slot(view_id, min_interval=min_interval, max_wait_s=max_wait_s)
    ts = int(time.time())
    url = SNAPSHOT_URL_TMPL.format(view_id=view_id, ts=ts)
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    except requests.RequestException as e:
        log.warning("view %s: request error %s", view_id, redact(str(e)))
        return SnapshotResult(
            ok=False, view_id=str(view_id), timestamp_utc=ts, error_reason="http_error"
        )
    content_type = resp.headers.get("Content-Type", "")
    nbytes = len(resp.content)
    if resp.status_code != 200:
        log.info("view %s: http %s", view_id, resp.status_code)
        return SnapshotResult(
            ok=False,
            view_id=str(view_id),
            timestamp_utc=ts,
            http_status=resp.status_code,
            content_type=content_type,
            nbytes=nbytes,
            error_reason="http_error",
        )
    if not content_type.lower().startswith("image/"):
        return SnapshotResult(
            ok=False,
            view_id=str(view_id),
            timestamp_utc=ts,
            http_status=resp.status_code,
            content_type=content_type,
            nbytes=nbytes,
            error_reason="non_image",
        )
    try:
        img = Image.open(io.BytesIO(resp.content))
        img.load()
        width, height = img.size
    except Exception as e:  # noqa: BLE001 - any PIL decode failure counts as non_image
        log.info("view %s: PIL decode failed: %s", view_id, e)
        return SnapshotResult(
            ok=False,
            view_id=str(view_id),
            timestamp_utc=ts,
            http_status=resp.status_code,
            content_type=content_type,
            nbytes=nbytes,
            error_reason="non_image",
        )
    return SnapshotResult(
        ok=True,
        view_id=str(view_id),
        timestamp_utc=ts,
        http_status=resp.status_code,
        content_type=content_type,
        nbytes=nbytes,
        width=width,
        height=height,
        image=img,
    )


def check_image_fetch_throttle(view_ids: list, n: int = 15) -> dict:
    """Empirical check: fetch `n` different camera views' snapshots (no API
    key involved) within about a minute, then make one real API call through
    the shared limiter and confirm it doesn't get a 429.

    Returns a small dict summarizing the result for the phase report.
    """
    from ga511.api import get_events

    sample = view_ids[:n]
    t0 = time.time()
    ok_count = 0
    for vid in sample:
        result = fetch_view(vid, enforce_interval=True)
        ok_count += int(result.ok)
    elapsed = time.time() - t0
    limiter = get_default_limiter()
    api_ok = True
    api_error = None
    try:
        events = get_events(limiter=limiter)
        api_event_count = len(events)
    except Exception as e:  # noqa: BLE001
        api_ok = False
        api_error = redact(str(e))
        api_event_count = None
    return {
        "n_requested": len(sample),
        "n_ok": ok_count,
        "elapsed_s": elapsed,
        "api_call_ok": api_ok,
        "api_error": api_error,
        "api_event_count": api_event_count,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a single 511GA snapshot")
    parser.add_argument("view_id")
    parser.add_argument("--no-throttle", action="store_true")
    args = parser.parse_args()
    result = fetch_view(args.view_id, enforce_interval=not args.no_throttle)
    print(
        f"ok={result.ok} status={result.http_status} content_type={result.content_type} "
        f"bytes={result.nbytes} size={result.width}x{result.height} "
        f"error_reason={result.error_reason}"
    )


if __name__ == "__main__":
    _main()
