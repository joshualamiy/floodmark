# iowa dot rwis cams from the iem archive: rainy + dry days
from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import imagehash
import requests

from ga511 import quality
from ga511.ratelimit import backoff_delay, locked_file, read_json_fd, write_json_fd
from othercams import paths, schema

SOURCE = "iowa_rwis"
USER_AGENT = "floodmark-research/0.1 (+https://github.com/joshualamiy/floodmark)"

CAMERA_LIST_URL = "https://mesonet.agron.iastate.edu/RWIS/camera.phtml"
SITE_GEOJSON_URL = "https://mesonet.agron.iastate.edu/geojson/network.php?network=IA_RWIS"
ARCHIVE_LIST_URL = "https://mesonet.agron.iastate.edu/json/webcam.py"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

VIEW_MIN_INTERVAL_S = 300.0
GLOBAL_MIN_INTERVAL_S = 0.7

DAYTIME_UTC_HOURS = range(13, 23)
WET_MONTHS = (4, 5, 6, 7, 8, 9, 10)
WET_DAY_MIN_MM = 3.0

_PACE_LOCK = threading.Lock()
_LAST_CALL_MONO = [0.0]
PACE_MIN_INTERVAL_S = 0.6


def _pace() -> None:
    with _PACE_LOCK:
        now = time.monotonic()
        wait = PACE_MIN_INTERVAL_S - (now - _LAST_CALL_MONO[0])
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL_MONO[0] = time.monotonic()


def _get_with_retry(url: str, params: dict, timeout: float, max_retries: int = 4) -> requests.Response:
    last_exc = None
    for attempt in range(max_retries):
        _pace()
        try:
            resp = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        except requests.RequestException as e:
            last_exc = e
            time.sleep(backoff_delay(attempt))
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == max_retries - 1:
                resp.raise_for_status()
            time.sleep(backoff_delay(attempt, base=2.0, cap=30.0))
            continue
        resp.raise_for_status()
        return resp
    raise last_exc or RuntimeError(f"gave up on {url}")

_OPTION_RE = re.compile(
    r"<option value='(IDOT-\d+-\d+)'>([^<]*?)\s*--\s*\(([\d-]+)\)</option>"
)
ROUTE_RE = re.compile(r"\b(I-?\d{1,3}|US-?\s?\d{1,3}|IA-?\s?\d{1,3}|Hwy\s?\d{1,3})\b", re.IGNORECASE)
_SKIP_VIEW_WORDS = ("bridge deck", "zoom", "sensor", "fp2000")
_NAME_STOP = {
    "bridge", "deck", "zoom", "sensor", "nb", "sb", "eb", "wb", "north", "south",
    "east", "west", "int", "interchange", "ramp", "approach", "hwy", "us", "ia",
    "i", "fp2000", "view", "cam", "camera",
}


def parse_camera_options(html_text: str) -> list[dict]:
    out = []
    for m in _OPTION_RE.finditer(html_text):
        cid, label, first_seen = m.groups()
        out.append({
            "view_id": cid,
            "group": cid.rsplit("-", 1)[0],
            "label": html.unescape(label).strip(),
            "first_seen": first_seen,
        })
    return out


def parse_site_features(geojson_obj: dict) -> list[dict]:
    out = []
    for feat in geojson_obj.get("features", []):
        p = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates")
        if not coords:
            continue
        out.append({
            "sid": p.get("sid"),
            "sname": p.get("sname") or "",
            "county": p.get("county") or "",
            "lat": coords[1],
            "lon": coords[0],
        })
    return out


def _normalize(s: str) -> set[str]:
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return {t for t in s.split() if t and t not in _NAME_STOP and not t.isdigit()}


def _first_word(s: str) -> str:
    m = re.match(r"^([A-Za-z .'-]+)", s)
    return (m.group(1).strip() if m else s).split()[0].strip("(),") if s else ""


def match_group_to_site(label: str, sites: list[dict], min_score: float = 0.5) -> tuple[dict | None, float]:
    ltoks = _normalize(label)
    if not ltoks:
        return None, 0.0
    best, best_score = None, 0.0
    label_town = _first_word(label).lower()
    for site in sites:
        stoks = _normalize(site["sname"])
        if not stoks:
            continue
        overlap = len(ltoks & stoks)
        score = overlap / max(1, min(len(ltoks), len(stoks)))
        if _first_word(site["sname"]).lower() == label_town:
            score += 0.5
        if score > best_score:
            best, best_score = site, score
    if best is not None and best_score >= min_score:
        return best, best_score
    return None, 0.0


def parse_roadway(label: str) -> str:
    m = ROUTE_RE.search(label)
    return m.group(1) if m else ""


def choose_preferred_view(options: list[dict]) -> dict:
    plain = [o for o in options if not any(w in o["label"].lower() for w in _SKIP_VIEW_WORDS)]
    pool = plain or options
    return min(pool, key=lambda o: len(o["label"]))


def build_camera_table(options: list[dict], sites: list[dict]) -> list[dict]:
    by_group: dict[str, list[dict]] = {}
    for o in options:
        by_group.setdefault(o["group"], []).append(o)

    rows = []
    for group, opts in by_group.items():
        rep_label = min((o["label"] for o in opts), key=len)
        site, score = match_group_to_site(rep_label, sites)
        if site is None:
            continue
        chosen = choose_preferred_view(opts)
        rows.append({
            "camera_id": group,
            "view_id": chosen["view_id"],
            "label": chosen["label"],
            "lat": site["lat"],
            "lon": site["lon"],
            "county": site["county"],
            "roadway": parse_roadway(chosen["label"]) or parse_roadway(site["sname"]),
            "match_score": round(score, 2),
        })
    return rows


def fetch_camera_list() -> str:
    resp = requests.get(CAMERA_LIST_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.text


def fetch_site_geojson() -> dict:
    resp = requests.get(SITE_GEOJSON_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def load_camera_table(refresh: bool = False) -> list[dict]:
    paths.ensure_dirs(SOURCE)
    cache = paths.cache_dir(SOURCE) / "camera_table.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    options = parse_camera_options(fetch_camera_list())
    sites = parse_site_features(fetch_site_geojson())
    rows = build_camera_table(options, sites)
    cache.write_text(json.dumps(rows))
    return rows


def daily_precip(lat: float, lon: float, start: date, end: date, cache_path: Path | None = None) -> list[tuple[str, float | None]]:
    if cache_path and cache_path.exists():
        obj = json.loads(cache_path.read_text())
    else:
        resp = _get_with_retry(
            OPEN_METEO_ARCHIVE_URL,
            params={
                "latitude": round(lat, 3), "longitude": round(lon, 3),
                "start_date": start.isoformat(), "end_date": end.isoformat(),
                "daily": "precipitation_sum", "timezone": "UTC",
            },
            timeout=30,
        )
        obj = resp.json()
        if cache_path:
            cache_path.write_text(json.dumps(obj))
    daily = obj.get("daily") or {}
    return list(zip(daily.get("time") or [], daily.get("precipitation_sum") or []))


def pick_target_days(series: list[tuple[str, float | None]]) -> dict[str, list[str]]:
    by_date = {d: mm for d, mm in series if mm is not None and int(d[5:7]) in WET_MONTHS}
    ordered = sorted(by_date)
    wet, dry = [], []
    for i, d in enumerate(ordered):
        mm = by_date[d]
        if mm >= WET_DAY_MIN_MM:
            wet.append((d, mm))
        elif mm == 0.0:
            d0 = date.fromisoformat(d)
            prevs_dry = all(
                by_date.get((d0 - timedelta(days=k)).isoformat(), 1.0) == 0.0 for k in (1, 2)
            )
            if prevs_dry:
                dry.append((d, mm))
    wet.sort(key=lambda x: -x[1])
    dry.sort(key=lambda x: x[0], reverse=True)
    return {"wet": [d for d, _ in wet], "dry": [d for d, _ in dry]}


def list_archive_images(cid: str, date_str: str) -> list[dict]:
    resp = _get_with_retry(ARCHIVE_LIST_URL, params={"cid": cid, "date": date_str}, timeout=20)
    return resp.json().get("images", [])


def _valid_to_ts(valid_iso: str) -> int:
    dt = datetime.strptime(valid_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def is_daytime(valid_iso: str) -> bool:
    hour = int(valid_iso[11:13])
    return hour in DAYTIME_UTC_HOURS


def subsample(items: list, k: int) -> list:
    if len(items) <= k:
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def _wait_for_view_slot(view_id: str, state_path: Path) -> None:
    start = time.time()
    while True:
        with locked_file(state_path) as fd:
            state = read_json_fd(fd, {})
            now = time.time()
            last_view = state.get(view_id)
            last_global = state.get("__global__")
            wait = 0.0
            if last_view is not None and now - last_view < VIEW_MIN_INTERVAL_S:
                wait = max(wait, VIEW_MIN_INTERVAL_S - (now - last_view))
            if last_global is not None and now - last_global < GLOBAL_MIN_INTERVAL_S:
                wait = max(wait, GLOBAL_MIN_INTERVAL_S - (now - last_global))
            if wait <= 0:
                state[view_id] = now
                state["__global__"] = now
                write_json_fd(fd, state)
                return
        if time.time() - start > 600.0:
            raise TimeoutError(f"othercams: view {view_id} throttled past budget")
        time.sleep(min(wait, 5.0) + 0.01)


def fetch_and_build_row(cam: dict, ts: int, href: str, weak_info: dict, prev_phash: dict) -> dict:
    cid = cam["view_id"]
    _wait_for_view_slot(cid, paths.last_fetch_path(SOURCE))
    row = {
        "frame_id": f"{cid}_{ts}", "path": "", "camera_id": cam["camera_id"], "view_id": cid,
        "lat": cam["lat"], "lon": cam["lon"], "roadway": cam["roadway"], "county": cam["county"],
        "timestamp_utc": ts, "http_status": "", "bytes": 0, "width": "", "height": "",
        "phash": "", "dead_reason": "", "weak_label": weak_info["weak_label"],
        "precip_1h_mm": weak_info["precip_1h_mm"] if weak_info["precip_1h_mm"] is not None else "",
        "precip_3h_mm": weak_info["precip_3h_mm"] if weak_info["precip_3h_mm"] is not None else "",
        "precip_source": weak_info["precip_source"] or "",
    }
    try:
        resp = requests.get(href, headers={"User-Agent": USER_AGENT}, timeout=20)
    except requests.RequestException:
        row["dead_reason"] = "http_error"
        return row
    row["http_status"] = resp.status_code
    row["bytes"] = len(resp.content)
    if resp.status_code != 200:
        row["dead_reason"] = "http_error"
        return row
    try:
        img = quality.bytes_to_image(resp.content)
    except Exception:  # noqa: BLE001
        row["dead_reason"] = "non_image"
        return row
    row["width"], row["height"] = img.size
    reason, phash = quality.classify(
        img, len(resp.content), prev_phash=prev_phash.get(cid), placeholder_hashes=[],
    )
    if phash is not None:
        row["phash"] = str(phash)
        prev_phash[cid] = phash
    if reason is not None:
        row["dead_reason"] = reason
        return row
    out_dir = paths.frames_dir(SOURCE) / cid
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ts}.jpg"
    img.convert("RGB").save(out_path, "JPEG", quality=90)
    row["path"] = str(out_path.relative_to(paths.source_dir(SOURCE)))
    return row


def collect_group(
    cam: dict, wet_target: int, dry_target: int, max_wet_days: int, max_dry_days: int,
    prev_phash: dict, log, existing: list[dict] | None = None, on_row=None,
) -> list[dict]:
    from ga511 import weather

    existing = existing or []
    seen = {r["frame_id"] for r in existing}
    counts = {label: sum(not r.get("dead_reason") and r.get("weak_label") == label
                         for r in existing) for label in ("likely_wet", "likely_dry")}
    if counts["likely_wet"] >= wet_target and counts["likely_dry"] >= dry_target:
        return []
    cache_path = paths.cache_dir(SOURCE) / f"daily_{cam['camera_id']}.json"
    today = datetime.now(timezone.utc).date()
    series = daily_precip(cam["lat"], cam["lon"], today - timedelta(days=395), today - timedelta(days=1), cache_path)
    days = pick_target_days(series)

    rows: list[dict] = []
    for label, target, day_list, max_days in (
        ("likely_wet", wet_target, days["wet"], max_wet_days),
        ("likely_dry", dry_target, days["dry"], max_dry_days),
    ):
        got = counts[label]
        for d in day_list[:max_days]:
            if got >= target:
                break
            try:
                imgs = list_archive_images(cam["view_id"], d.replace("-", ""))
            except requests.RequestException as e:
                log.warning("archive list failed %s %s: %s", cam["view_id"], d, e)
                continue
            candidates = subsample([im for im in imgs if is_daytime(im.get("valid", ""))], 6)
            for im in candidates:
                if got >= target:
                    break
                ts = _valid_to_ts(im["valid"])
                frame_id = f"{cam['view_id']}_{ts}"
                if frame_id in seen:
                    continue
                _pace()
                info = weather.get_precip_and_label(cam["lat"], cam["lon"], ts)
                if info["weak_label"] != label:
                    continue
                row = fetch_and_build_row(cam, ts, im["href"], info, prev_phash)
                if on_row:
                    on_row(row)
                rows.append(row)
                seen.add(frame_id)
                if not row["dead_reason"]:
                    got += 1
    return rows


def _run_collection(
    max_groups: int = 60, wet_target: int = 4, dry_target: int = 4,
    max_wet_days: int = 5, max_dry_days: int = 3, max_workers: int = 20, seed: int = 7,
) -> dict:
    log = paths.setup_logging("othercams_iowa_rwis")
    paths.ensure_dirs(SOURCE)
    all_cams = load_camera_table()
    rng = random.Random(seed)
    rng.shuffle(all_cams)
    cams = all_cams[:max_groups]
    log.info("iowa_rwis: %d matched camera groups selected (of %d)", len(cams), len(all_cams))

    prev_phash: dict = {}
    existing_by_camera: dict[str, list[dict]] = {}
    for row in schema.read_frame_rows(paths.frames_csv(SOURCE)):
        existing_by_camera.setdefault(row["camera_id"], []).append(row)
        if row.get("phash"):
            prev_phash[row["view_id"]] = imagehash.hex_to_hash(row["phash"])

    def save_row(row):
        schema.append_frame_rows([row], paths.frames_csv(SOURCE), paths.frames_csv_lock(SOURCE))

    totals = {"wet_ok": 0, "dry_ok": 0, "dead": {}, "groups_with_frames": 0}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(collect_group, cam, wet_target, dry_target, max_wet_days, max_dry_days,
                          prev_phash, log, existing_by_camera.get(cam["camera_id"], []), save_row): cam
                for cam in cams}
        for fut in as_completed(futs):
            cam = futs[fut]
            try:
                rows = fut.result()
            except Exception as e:  # noqa: BLE001
                log.error("group %s failed: %s", cam["camera_id"], e)
                continue
            if not rows:
                continue
            n_wet = sum(1 for r in rows if not r["dead_reason"] and r["weak_label"] == "likely_wet")
            n_dry = sum(1 for r in rows if not r["dead_reason"] and r["weak_label"] == "likely_dry")
            totals["wet_ok"] += n_wet
            totals["dry_ok"] += n_dry
            if n_wet or n_dry:
                totals["groups_with_frames"] += 1
            for r in rows:
                if r["dead_reason"]:
                    totals["dead"][r["dead_reason"]] = totals["dead"].get(r["dead_reason"], 0) + 1
            log.info("group %s: wet=%d dry=%d", cam["camera_id"], n_wet, n_dry)
    log.info("run_collection done: %s", totals)
    return totals


def run_collection(**kwargs) -> dict:
    from othercams.jobs import collection_lock

    with collection_lock(paths.pid_path(SOURCE)):
        return _run_collection(**kwargs)


def status() -> dict:
    from othercams import jobs

    rows = schema.read_frame_rows(paths.frames_csv(SOURCE))
    counts = {"total": len(rows), "weak_label": {}, "dead_reason": {}, "cameras": set()}
    for r in rows:
        counts["cameras"].add(r.get("camera_id"))
        if r.get("dead_reason"):
            counts["dead_reason"][r["dead_reason"]] = counts["dead_reason"].get(r["dead_reason"], 0) + 1
        elif r.get("weak_label"):
            counts["weak_label"][r["weak_label"]] = counts["weak_label"].get(r["weak_label"], 0) + 1
    counts["cameras"] = len(counts["cameras"])
    pid = jobs.read_pid(paths.pid_path(SOURCE))
    counts.update(pid=pid, running=pid is not None and jobs.pid_alive(pid))
    return counts


def _main() -> None:
    from othercams import jobs

    ap = argparse.ArgumentParser(description="Iowa RWIS (IEM archive) collector")
    sub = ap.add_subparsers(dest="cmd", required=True)

    cams_p = sub.add_parser("cameras", help="build+print the matched camera table")
    cams_p.add_argument("--refresh", action="store_true")

    run_p = sub.add_parser("run", help="run one collection pass in the foreground")
    run_p.add_argument("--max-groups", type=int, default=60)
    run_p.add_argument("--wet-target", type=int, default=4)
    run_p.add_argument("--dry-target", type=int, default=4)
    run_p.add_argument("--max-workers", type=int, default=20)

    sub.add_parser("start", help="run a collection pass detached")
    sub.add_parser("stop", help="stop a detached run")
    sub.add_parser("status", help="print frames.csv counts")

    args = ap.parse_args()
    if args.cmd == "cameras":
        rows = load_camera_table(refresh=args.refresh)
        print(f"matched {len(rows)} camera groups")
        for r in rows[:10]:
            print(r)
        out_csv = paths.source_dir(SOURCE) / "cameras.csv"
        paths.ensure_dirs(SOURCE)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_csv}")
    elif args.cmd == "run":
        summary = run_collection(
            max_groups=args.max_groups, wet_target=args.wet_target,
            dry_target=args.dry_target, max_workers=args.max_workers,
        )
        print(json.dumps(summary))
    elif args.cmd == "start":
        paths.ensure_dirs(SOURCE)
        with locked_file(paths.pid_path(SOURCE).with_suffix(".start.lock")):
            previous = jobs.read_pid(paths.pid_path(SOURCE))
            if previous and jobs.pid_alive(previous):
                print(f"already running pid={previous}")
                return
            pid = jobs.start_detached(
                "othercams.iowa_rwis", ["run"],
                paths.LOGS_JOBS_DIR / "othercams_iowa_rwis_run.log",
                paths.FLOOD_ML_DIR,
            )
            paths.pid_path(SOURCE).write_text(str(pid))
            print(f"started pid={pid}")
    elif args.cmd == "stop":
        ok = jobs.stop(paths.pid_path(SOURCE))
        print("stopped" if ok else "not running")
    elif args.cmd == "status":
        print(json.dumps(status(), indent=2))


if __name__ == "__main__":
    _main()

