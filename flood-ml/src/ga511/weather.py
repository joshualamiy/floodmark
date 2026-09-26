# weak rain labels from nws / open-meteo
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

from ga511.paths import FRAMES_CSV, WEATHER_CACHE_DIR, ensure_dirs, setup_logging

log = setup_logging("ga511_weather")

USER_AGENT = "floodmark-research/0.1 (+https://github.com/joshualamiy/floodmark)"
NWS_TIMEOUT = 10.0
OPEN_METEO_TIMEOUT = 10.0

WET_1H_MM = 0.2
WET_3H_MM = 1.0
DRY_6H_MM = 0.0


def classify_weak_label(
    precip_1h_mm: float | None,
    precip_3h_mm: float | None,
    precip_6h_mm: float | None,
) -> str:
    if precip_1h_mm is not None and precip_1h_mm >= WET_1H_MM:
        return "likely_wet"
    if precip_3h_mm is not None and precip_3h_mm >= WET_3H_MM:
        return "likely_wet"
    if precip_6h_mm is not None and precip_6h_mm <= DRY_6H_MM:
        return "likely_dry"
    return "uncertain"


def _cache_path(kind: str, key: str, hour_key: str) -> Path:
    d = WEATHER_CACHE_DIR / kind
    d.mkdir(parents=True, exist_ok=True)
    safe_key = key.replace("/", "_")
    return d / f"{safe_key}_{hour_key}.json"


def _read_cache(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(path: Path, obj) -> None:
    ensure_dirs()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)
    except OSError as e:
        log.warning("could not write weather cache %s: %s", path, e)


def _round_grid(lat: float, lon: float, step: float = 0.05) -> str:
    return f"{round(lat / step) * step:.3f}_{round(lon / step) * step:.3f}"


def _to_mm(value, unit_code: str | None) -> float | None:
    if value is None:
        return None
    unit = (unit_code or "").lower()
    if "mm" in unit:
        return float(value)
    if unit.endswith((":m", "meter")) or unit == "wmounit:m":
        return float(value) * 1000.0
    if "in" in unit:
        return float(value) * 25.4
    return float(value) * 1000.0


def _nearest_station(lat: float, lon: float) -> str | None:
    hour_key = "static"
    key = _round_grid(lat, lon, step=0.05)
    cache_path = _cache_path("nws_station", key, hour_key)
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached.get("station_id")

    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    try:
        points_resp = requests.get(
            f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}",
            headers=headers,
            timeout=NWS_TIMEOUT,
        )
        points_resp.raise_for_status()
        stations_url = points_resp.json()["properties"]["observationStations"]
        stations_resp = requests.get(stations_url, headers=headers, timeout=NWS_TIMEOUT)
        stations_resp.raise_for_status()
        features = stations_resp.json().get("features", [])
        station_id = None
        if features:
            station_id = features[0]["properties"]["stationIdentifier"]
    except (requests.RequestException, KeyError, IndexError, ValueError) as e:
        log.info("nws station lookup failed: %s", e)
        station_id = None
    _write_cache(cache_path, {"station_id": station_id})
    return station_id


def _nws_precip(lat: float, lon: float, ts_utc: int) -> dict | None:
    station_id = _nearest_station(lat, lon)
    if not station_id:
        return None
    dt = datetime.fromtimestamp(ts_utc, tz=timezone.utc)
    hour_key = dt.strftime("%Y%m%d%H")
    cache_path = _cache_path("nws_obs", station_id, hour_key)
    cached = _read_cache(cache_path)
    if cached is None:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
        try:
            resp = requests.get(
                f"https://api.weather.gov/stations/{station_id}/observations",
                headers=headers,
                params={"limit": 50},
                timeout=NWS_TIMEOUT,
            )
            resp.raise_for_status()
            cached = resp.json()
        except requests.RequestException as e:
            log.info("nws observations fetch failed for %s: %s", station_id, e)
            cached = {"features": []}
        _write_cache(cache_path, cached)

    best = None
    best_dt_diff = None
    for feat in cached.get("features", []):
        props = feat.get("properties", {})
        obs_time = props.get("timestamp")
        if not obs_time:
            continue
        try:
            obs_dt = datetime.fromisoformat(obs_time.replace("Z", "+00:00"))
        except ValueError:
            continue
        diff = abs((obs_dt - dt).total_seconds())
        if diff > 3 * 3600:
            continue
        if best_dt_diff is None or diff < best_dt_diff:
            best, best_dt_diff = props, diff
    if best is None:
        return None

    def field(name):
        f = best.get(name) or {}
        return _to_mm(f.get("value"), f.get("unitCode"))

    p1 = field("precipitationLastHour")
    p3 = field("precipitationLast3Hours")
    p6 = field("precipitationLast6Hours")
    if p1 is None and p3 is None and p6 is None:
        return None
    return {
        "precip_1h_mm": p1,
        "precip_3h_mm": p3,
        "precip_6h_mm": p6,
        "source": f"nws:{station_id}",
    }


def _open_meteo_precip(lat: float, lon: float, ts_utc: int) -> dict | None:
    dt = datetime.fromtimestamp(ts_utc, tz=timezone.utc)
    now = datetime.now(tz=timezone.utc)
    age_days = (now - dt).days
    key = _round_grid(lat, lon, step=0.1)
    hour_key = dt.strftime("%Y%m%d%H")
    cache_path = _cache_path("open_meteo", key, hour_key)
    cached = _read_cache(cache_path)
    if cached is None:
        params = {
            "latitude": round(lat, 3),
            "longitude": round(lon, 3),
            "hourly": "precipitation",
            "timezone": "UTC",
        }
        try:
            if age_days > 5:
                url = "https://archive-api.open-meteo.com/v1/archive"
                day = dt.strftime("%Y-%m-%d")
                params["start_date"] = day
                params["end_date"] = day
            else:
                url = "https://api.open-meteo.com/v1/forecast"
                params["past_days"] = min(max(age_days + 1, 1), 92)
                params["forecast_days"] = 1
            resp = requests.get(url, params=params, timeout=OPEN_METEO_TIMEOUT)
            resp.raise_for_status()
            cached = resp.json()
        except requests.RequestException as e:
            log.info("open-meteo fetch failed: %s", e)
            cached = {}
        _write_cache(cache_path, cached)

    hourly = cached.get("hourly") or {}
    times = hourly.get("time") or []
    precip = hourly.get("precipitation") or []
    if not times or not precip:
        return None
    target = dt.replace(minute=0, second=0, microsecond=0)
    idx_by_time = {t: i for i, t in enumerate(times)}
    target_str = target.strftime("%Y-%m-%dT%H:00")
    if target_str not in idx_by_time:
        return None
    idx = idx_by_time[target_str]

    def sum_window(hours):
        vals = precip[max(0, idx - hours + 1) : idx + 1]
        vals = [v for v in vals if v is not None]
        return float(sum(vals)) if vals else None

    return {
        "precip_1h_mm": precip[idx] if precip[idx] is not None else None,
        "precip_3h_mm": sum_window(3),
        "precip_6h_mm": sum_window(6),
        "source": "open_meteo",
    }


def get_precip_and_label(lat: float, lon: float, ts_utc: int) -> dict:
    lat, lon, ts_utc = float(lat), float(lon), int(ts_utc)
    result = None
    try:
        result = _nws_precip(lat, lon, ts_utc)
    except Exception as e:  # noqa: BLE001
        log.warning("nws lookup errored: %s", e)
    if result is None:
        try:
            result = _open_meteo_precip(lat, lon, ts_utc)
        except Exception as e:  # noqa: BLE001
            log.warning("open-meteo lookup errored: %s", e)
    if result is None:
        return {
            "precip_1h_mm": None,
            "precip_3h_mm": None,
            "precip_source": "none",
            "weak_label": "uncertain",
        }
    label = classify_weak_label(
        result.get("precip_1h_mm"), result.get("precip_3h_mm"), result.get("precip_6h_mm")
    )
    return {
        "precip_1h_mm": result.get("precip_1h_mm"),
        "precip_3h_mm": result.get("precip_3h_mm"),
        "precip_source": result.get("source"),
        "weak_label": label,
    }


def fill_weak_labels(csv_path: Path = FRAMES_CSV, limit: int | None = None) -> int:
    from ga511.paths import FRAMES_CSV_LOCK
    from ga511.ratelimit import locked_file, read_json_fd  # noqa: F401

    csv_path = Path(csv_path)
    with locked_file(FRAMES_CSV_LOCK):
        if not csv_path.exists():
            return 0
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

        updated = 0
        for row in rows:
            if limit is not None and updated >= limit:
                break
            if row.get("dead_reason"):
                continue
            if row.get("weak_label"):
                continue
            lat, lon, ts = row.get("lat"), row.get("lon"), row.get("timestamp_utc")
            if not lat or not lon or not ts:
                continue
            try:
                info = get_precip_and_label(lat, lon, ts)
            except Exception as e:  # noqa: BLE001
                log.warning("weather lookup failed for frame %s: %s", row.get("frame_id"), e)
                continue
            row["weak_label"] = info["weak_label"]
            row["precip_1h_mm"] = "" if info["precip_1h_mm"] is None else info["precip_1h_mm"]
            row["precip_3h_mm"] = "" if info["precip_3h_mm"] is None else info["precip_3h_mm"]
            row["precip_source"] = info["precip_source"] or ""
            updated += 1

        if updated:
            tmp_path = csv_path.with_suffix(csv_path.suffix + f".tmp{os.getpid()}")
            with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, csv_path)
    log.info("fill_weak_labels: updated %d rows", updated)
    return updated


def _main() -> None:
    parser = argparse.ArgumentParser(description="Fill weak weather labels in frames.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    n = fill_weak_labels(limit=args.limit)
    print(f"updated {n} rows")


if __name__ == "__main__":
    _main()

