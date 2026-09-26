# 511ga api client: cameras (cached daily) + events
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from ga511.paths import CAMERAS_CACHE_PATH, ENV_PATH, ensure_dirs, setup_logging
from ga511.ratelimit import RateLimiter, backoff_delay, get_default_limiter, redact

CAMERAS_URL = "https://511ga.org/api/v2/get/cameras"
EVENTS_URL = "https://511ga.org/api/v2/get/event"
USER_AGENT = "floodmark-research/0.1 (+https://github.com/joshualamiy/floodmark)"

CAMERAS_MAX_AGE_S = 86400

log = setup_logging("ga511_api")

_dotenv_loaded = False


def _load_env() -> None:
    global _dotenv_loaded
    if not _dotenv_loaded:
        load_dotenv(ENV_PATH)
        _dotenv_loaded = True


def _get_api_key() -> str:
    import os

    _load_env()
    key = os.environ.get("GA511_API_KEY")
    if not key:
        raise RuntimeError(
            f"GA511_API_KEY not set (looked for .env at {ENV_PATH})"
        )
    return key


def _call_api(
    url: str,
    limiter: RateLimiter,
    params: dict | None = None,
    max_retries: int = 5,
    timeout: float = 30.0,
):
    params = dict(params or {})
    params["key"] = _get_api_key()
    params["format"] = "json"
    last_err = None
    for attempt in range(max_retries):
        limiter.acquire()
        try:
            resp = requests.get(
                url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT}
            )
        except requests.RequestException as e:
            last_err = redact(str(e))
            delay = backoff_delay(attempt)
            log.warning("request error (%s), retrying in %.1fs", last_err, delay)
            time.sleep(delay)
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            delay = backoff_delay(attempt)
            log.warning("http %s, retrying in %.1fs", resp.status_code, delay)
            time.sleep(delay)
            continue
        try:
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(redact(str(e))) from None
        return resp.json()
    raise RuntimeError(f"511ga api failed after {max_retries} attempts: {last_err}")


def get_cameras(
    limiter: RateLimiter | None = None,
    force_refresh: bool = False,
    max_age_s: int = CAMERAS_MAX_AGE_S,
    cache_path: Path = CAMERAS_CACHE_PATH,
):
    ensure_dirs()
    if not force_refresh and cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < max_age_s:
            with open(cache_path, encoding="utf-8") as f:
                cached = json.load(f)
            log.info("get_cameras: using cache (%.0fs old, %d cameras)",
                      age, len(cached.get("cameras", [])))
            return cached["cameras"]
    limiter = limiter or get_default_limiter()
    log.info("get_cameras: fetching from API")
    data = _call_api(CAMERAS_URL, limiter)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"fetched_at": time.time(), "cameras": data}, f)
    log.info("get_cameras: fetched %d cameras", len(data))
    return data


def get_events(limiter: RateLimiter | None = None):
    limiter = limiter or get_default_limiter()
    data = _call_api(EVENTS_URL, limiter)
    log.info("get_events: fetched %d events", len(data))
    return data


def _main() -> None:
    parser = argparse.ArgumentParser(description="Ad-hoc 511GA API calls")
    parser.add_argument("--cameras", action="store_true", help="fetch/cache cameras")
    parser.add_argument("--events", action="store_true", help="fetch events")
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()
    if args.cameras:
        cams = get_cameras(force_refresh=args.force_refresh)
        print(f"cameras: {len(cams)}")
    if args.events:
        evs = get_events()
        print(f"events: {len(evs)}")
    if not (args.cameras or args.events):
        parser.print_help()


if __name__ == "__main__":
    _main()

