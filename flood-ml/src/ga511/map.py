# camera maps (folium html + scatter png)
from __future__ import annotations

import argparse

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ga511.cameras import load_atlanta_rows
from ga511.paths import CAMERAS_ATLANTA_CSV, FIGURES_DIR, REPORTS_DIR, ensure_dirs, setup_logging

log = setup_logging("ga511_map")

MAP_OUT_PATH = REPORTS_DIR / "ga511_camera_map.html"
SCATTER_OUT_PATH = FIGURES_DIR / "ga511_cameras.png"


def build_folium_map(rows: list[dict], out_path=MAP_OUT_PATH) -> None:
    ensure_dirs()
    if rows:
        lats = [float(r["lat"]) for r in rows]
        lons = [float(r["lon"]) for r in rows]
        center = [sum(lats) / len(lats), sum(lons) / len(lons)]
    else:
        center = [33.79, -84.39]
    fmap = folium.Map(location=center, zoom_start=10, tiles="OpenStreetMap")
    for r in rows:
        try:
            lat, lon = float(r["lat"]), float(r["lon"])
        except (TypeError, ValueError):
            continue
        popup = folium.Popup(
            html=(
                f"<b>{r.get('name') or ''}</b><br>"
                f"{r.get('roadway') or ''} {r.get('direction') or ''}<br>"
                f"{r.get('location') or ''}<br>"
                f"county: {r.get('county') or ''}<br>"
                f"view_id: {r.get('view_id')}"
            ),
            max_width=300,
        )
        folium.CircleMarker(
            location=[lat, lon], radius=4, color="#1f77b4", fill=True, fill_opacity=0.8, popup=popup
        ).add_to(fmap)
    fmap.save(str(out_path))
    log.info("wrote folium map with %d markers -> %s", len(rows), out_path)


def build_scatter_png(rows: list[dict], out_path=SCATTER_OUT_PATH) -> None:
    ensure_dirs()
    lats, lons = [], []
    for r in rows:
        try:
            lats.append(float(r["lat"]))
            lons.append(float(r["lon"]))
        except (TypeError, ValueError):
            continue
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(lons, lats, s=8, alpha=0.6, color="#1f77b4")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"511GA Atlanta camera views (n={len(lats)})")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("wrote scatter png with %d points -> %s", len(lats), out_path)


def run(csv_path=CAMERAS_ATLANTA_CSV) -> None:
    rows = load_atlanta_rows(csv_path)
    build_folium_map(rows)
    build_scatter_png(rows)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build camera location maps")
    parser.parse_args()
    run()
    print(f"wrote {MAP_OUT_PATH}")
    print(f"wrote {SCATTER_OUT_PATH}")


if __name__ == "__main__":
    _main()

