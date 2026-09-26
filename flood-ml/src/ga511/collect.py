# sweeps camera views, saves good frames, logs dead ones
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import imagehash

from ga511 import quality, snapshot
from ga511.cameras import load_atlanta_rows
from ga511.paths import (
    CAMERAS_ATLANTA_CSV,
    DATA_DIR,
    FRAMES_CSV,
    FRAMES_CSV_LOCK,
    FRAMES_DIR,
    PREV_PHASH_PATH,
    ensure_dirs,
    setup_logging,
)
from ga511.ratelimit import locked_file, read_json_fd, write_json_fd

log = setup_logging("ga511_collect")

FRAME_CSV_FIELDS = [
    "frame_id",
    "path",
    "camera_id",
    "view_id",
    "lat",
    "lon",
    "roadway",
    "county",
    "timestamp_utc",
    "http_status",
    "bytes",
    "width",
    "height",
    "phash",
    "dead_reason",
    "weak_label",
    "precip_1h_mm",
    "precip_3h_mm",
    "precip_source",
]

DEFAULT_INTERVAL_MIN = 60.0
DEFAULT_MAX_WORKERS = 10


def _append_frame_rows(rows: list[dict], csv_path: Path = FRAMES_CSV) -> None:
    ensure_dirs()
    with locked_file(FRAMES_CSV_LOCK):
        is_new = not csv_path.exists() or csv_path.stat().st_size == 0
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FRAME_CSV_FIELDS)
            if is_new:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)


def _load_prev_phashes(path: Path = PREV_PHASH_PATH) -> dict:
    with locked_file(path) as fd:
        return read_json_fd(fd, {})


def _save_prev_phashes(state: dict, path: Path = PREV_PHASH_PATH) -> None:
    with locked_file(path) as fd:
        write_json_fd(fd, state)


def _process_view(view: dict, placeholder_hashes: list, prev_phash_hex: str | None, with_weather: bool) -> dict:
    view_id = view["view_id"]
    result = snapshot.fetch_view(view_id)
    row = {
        "frame_id": f"{view_id}_{result.timestamp_utc}",
        "path": "",
        "camera_id": view.get("camera_id"),
        "view_id": view_id,
        "lat": view.get("lat"),
        "lon": view.get("lon"),
        "roadway": view.get("roadway"),
        "county": view.get("county"),
        "timestamp_utc": result.timestamp_utc,
        "http_status": result.http_status if result.http_status is not None else "",
        "bytes": result.nbytes,
        "width": result.width if result.width is not None else "",
        "height": result.height if result.height is not None else "",
        "phash": "",
        "dead_reason": "",
        "weak_label": "",
        "precip_1h_mm": "",
        "precip_3h_mm": "",
        "precip_source": "",
        "_dead_reason": None,
        "_new_phash": None,
    }

    if not result.ok:
        row["dead_reason"] = result.error_reason
        row["_dead_reason"] = result.error_reason
        return row

    prev_phash = imagehash.hex_to_hash(prev_phash_hex) if prev_phash_hex else None
    reason, phash = quality.classify(
        result.image,
        result.nbytes,
        prev_phash=prev_phash,
        placeholder_hashes=placeholder_hashes,
    )
    if phash is not None:
        row["phash"] = str(phash)
        row["_new_phash"] = str(phash)

    if reason is not None:
        row["dead_reason"] = reason
        row["_dead_reason"] = reason
        quality.save_dead_sample(reason, view_id, result.timestamp_utc, result.image)
        return row

    out_dir = FRAMES_DIR / str(view_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{result.timestamp_utc}.jpg"
    result.image.convert("RGB").save(out_path, "JPEG", quality=90)
    row["path"] = str(out_path.relative_to(DATA_DIR))

    if with_weather and row["lat"] and row["lon"]:
        try:
            from ga511 import weather

            info = weather.get_precip_and_label(row["lat"], row["lon"], result.timestamp_utc)
            row["weak_label"] = info["weak_label"]
            row["precip_1h_mm"] = "" if info["precip_1h_mm"] is None else info["precip_1h_mm"]
            row["precip_3h_mm"] = "" if info["precip_3h_mm"] is None else info["precip_3h_mm"]
            row["precip_source"] = info["precip_source"] or ""
        except Exception as e:  # noqa: BLE001
            log.warning("weather lookup failed for view %s: %s", view_id, e)

    return row


def sweep_once(
    views: list[dict] | None = None,
    with_weather: bool = False,
    max_views: int | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    stop_flag=None,
    offset: int = 0,
) -> dict:
    if views is None:
        views = load_atlanta_rows(CAMERAS_ATLANTA_CSV)
    if offset:
        views = views[offset:]
    if max_views is not None:
        views = views[:max_views]

    placeholder_hashes = quality.load_placeholder_hashes()
    prev_state = _load_prev_phashes()

    counts = {"ok": 0, "dead": {}}
    pending_rows = []
    flush_every = 25
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for view in views:
            if stop_flag is not None and stop_flag.is_set():
                break
            view_id = str(view["view_id"])
            fut = executor.submit(
                _process_view, view, placeholder_hashes, prev_state.get(view_id), with_weather
            )
            futures[fut] = view_id

        for fut in as_completed(futures):
            view_id = futures[fut]
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001
                log.error("view %s: processing failed: %s", view_id, e)
                continue
            new_phash = row.pop("_new_phash")
            dead_reason = row.pop("_dead_reason")
            if new_phash is not None:
                prev_state[view_id] = new_phash
            if dead_reason is not None:
                counts["dead"][dead_reason] = counts["dead"].get(dead_reason, 0) + 1
            else:
                counts["ok"] += 1
            pending_rows.append(row)
            if len(pending_rows) >= flush_every:
                _append_frame_rows(pending_rows)
                _save_prev_phashes(prev_state)
                pending_rows = []

    if pending_rows:
        _append_frame_rows(pending_rows)
    _save_prev_phashes(prev_state)
    log.info("sweep complete: ok=%d dead=%s (of %d views)", counts["ok"], counts["dead"], len(views))
    return {"n_views": len(views), "ok": counts["ok"], "dead": counts["dead"]}


def run_forever(
    interval_min: float = DEFAULT_INTERVAL_MIN,
    with_weather: bool = False,
    max_workers: int = DEFAULT_MAX_WORKERS,
    stop_flag=None,
) -> None:
    import threading

    stop_flag = stop_flag or threading.Event()
    while not stop_flag.is_set():
        try:
            sweep_once(with_weather=with_weather, max_workers=max_workers, stop_flag=stop_flag)
        except Exception as e:  # noqa: BLE001
            log.error("sweep failed: %s", e)
        stop_flag.wait(interval_min * 60.0)


def _main() -> None:
    parser = argparse.ArgumentParser(description="511GA frame collector")
    parser.add_argument("--once", action="store_true", help="run a single sweep and exit")
    parser.add_argument(
        "--interval-min", type=float, default=DEFAULT_INTERVAL_MIN, help="minutes between sweeps"
    )
    parser.add_argument(
        "--max-views", type=int, default=None, help="limit views per sweep (chunking/debugging)"
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="skip this many views (for resumable chunked sweeps)"
    )
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument(
        "--with-weather", action="store_true", help="fetch weather labels inline (slower)"
    )
    args = parser.parse_args()
    if args.once:
        summary = sweep_once(
            with_weather=args.with_weather,
            max_views=args.max_views,
            max_workers=args.max_workers,
            offset=args.offset,
        )
        print(json.dumps(summary))
    else:
        run_forever(interval_min=args.interval_min, with_weather=args.with_weather)


if __name__ == "__main__":
    _main()

