# same frames.csv schema as ga511 so the label tool works
from __future__ import annotations

import csv
from pathlib import Path

from ga511.ratelimit import locked_file

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


def append_frame_rows(rows: list[dict], csv_path: Path, lock_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with locked_file(lock_path):
        seen = {r["frame_id"] for r in read_frame_rows(csv_path)}
        is_new = not csv_path.exists() or csv_path.stat().st_size == 0
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FRAME_CSV_FIELDS)
            if is_new:
                w.writeheader()
            for row in rows:
                if row["frame_id"] in seen:
                    continue
                w.writerow({k: row.get(k, "") for k in FRAME_CSV_FIELDS})
                seen.add(row["frame_id"])


def read_frame_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

