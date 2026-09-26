"""Filter 511GA cameras to the Atlanta metro bbox and write the working
camera/view table used by every other ga511 module.
"""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass

from ga511.api import get_cameras
from ga511.geo import in_atlanta_bbox
from ga511.paths import CAMERAS_ATLANTA_CSV, ensure_dirs, setup_logging

log = setup_logging("ga511_cameras")

CSV_FIELDS = [
    "camera_id",
    "view_id",
    "lat",
    "lon",
    "roadway",
    "direction",
    "location",
    "county",
    "name",
    "view_status",
    "view_url",
]

_COUNTY_RE = re.compile(r"\(([^()]+)\)\s*$")


def parse_county(location: str | None) -> str | None:
    """Pull the county name out of a Location string's trailing parens, e.g.
    "I-75 NB at Spring St (Fulton)" -> "Fulton". Returns None if not found.
    """
    if not location:
        return None
    m = _COUNTY_RE.search(location.strip())
    if not m:
        return None
    county = m.group(1).strip()
    return county or None


@dataclass
class CameraSummary:
    total_cameras: int = 0
    total_views: int = 0
    bbox_cameras: int = 0
    bbox_views: int = 0
    bbox_enabled_views: int = 0


def build_atlanta_rows(cameras: list) -> tuple[list[dict], CameraSummary]:
    """Filter `cameras` (raw API dicts) to the Atlanta bbox and enabled
    views. Returns (rows, summary) where rows match CSV_FIELDS.
    """
    summary = CameraSummary()
    rows: list[dict] = []
    for cam in cameras:
        summary.total_cameras += 1
        views = cam.get("Views") or []
        summary.total_views += len(views)
        lat, lon = cam.get("Latitude"), cam.get("Longitude")
        if lat is None or lon is None or not in_atlanta_bbox(float(lat), float(lon)):
            continue
        summary.bbox_cameras += 1
        summary.bbox_views += len(views)
        location = cam.get("Location")
        county = parse_county(location)
        for view in views:
            status = view.get("Status")
            if status != "Enabled":
                continue
            summary.bbox_enabled_views += 1
            rows.append(
                {
                    "camera_id": cam.get("Id"),
                    "view_id": view.get("Id"),
                    "lat": lat,
                    "lon": lon,
                    "roadway": cam.get("Roadway"),
                    "direction": cam.get("Direction"),
                    "location": location,
                    "county": county,
                    "name": cam.get("Name"),
                    "view_status": status,
                    "view_url": view.get("Url"),
                }
            )
    return rows, summary


def write_atlanta_csv(rows: list[dict], out_path=CAMERAS_ATLANTA_CSV) -> None:
    ensure_dirs()
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_atlanta_rows(csv_path=CAMERAS_ATLANTA_CSV) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run(force_refresh: bool = False) -> CameraSummary:
    cameras = get_cameras(force_refresh=force_refresh)
    rows, summary = build_atlanta_rows(cameras)
    write_atlanta_csv(rows)
    log.info(
        "cameras: total=%d views=%d | bbox cameras=%d bbox views=%d enabled=%d",
        summary.total_cameras,
        summary.total_views,
        summary.bbox_cameras,
        summary.bbox_views,
        summary.bbox_enabled_views,
    )
    return summary


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build cameras_atlanta.csv")
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()
    summary = run(force_refresh=args.force_refresh)
    print(
        f"total_cameras={summary.total_cameras} total_views={summary.total_views} "
        f"bbox_cameras={summary.bbox_cameras} bbox_views={summary.bbox_views} "
        f"bbox_enabled_views={summary.bbox_enabled_views}"
    )
    print(f"wrote {CAMERAS_ATLANTA_CSV}")


if __name__ == "__main__":
    _main()
