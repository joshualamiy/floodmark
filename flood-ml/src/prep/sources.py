"""Per-source raw row builders for the manifest.

Each `iter_*` function returns a list of plain dicts describing one image
each, before resizing, road-region labeling, CLIP filtering, dedup, or split
assignment. Common keys:

    source, orig_path (str, relative to flood-ml/), mask_path (str|None),
    mask_kind (one of "fred", "binary01", "grayscale_thresh", "bottom_band01",
    None), label (str|None -- set directly when already known, e.g. FRED dry
    or NYSDOT), label_source, group_id, camera_id (str|None), view_type,
    license, restricted (bool), weak_label (str|None), notes (str)

`label=None` + a `mask_path`/`mask_kind` means Phase 2's mask_rules module
must compute `water_frac_road` and derive the label later.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from prep.common import (
    LICENSE_FLOOD_AREA_SEGMENTATION,
    LICENSE_FMD,
    LICENSE_FRED,
    LICENSE_GA511,
    LICENSE_NYSDOT,
    LICENSE_ROADWAY_FLOODING,
    RAW_ROOT,
    RESTRICTED_ROOT,
    normalize_group_name,
)
from prep.mask_rules import parse_nysdot_group_id

ROADWAY_FLOODING_DIR = RAW_ROOT / "roadway_flooding" / "Dataset"
FLOOD_AREA_DIR = RAW_ROOT / "flood_area_segmentation"
FRED_DIR = RAW_ROOT / "fred"
NYSDOT_DIR = RAW_ROOT / "nysdot_road_surface"
FMD_INDEX_PATH = RESTRICTED_ROOT / "flood_master" / "index.csv"

# Datasets excluded per Phase 1 findings (no license grant / no road scenes).
EXCLUDED_FMD_SOURCES = {"Water Dataset"}


def load_fmd_index() -> list[dict]:
    if not FMD_INDEX_PATH.exists():
        return []
    with open(FMD_INDEX_PATH, newline="") as f:
        return list(csv.DictReader(f))


def iter_roadway_flooding() -> list[dict]:
    """441 images with their own {0,1} binary flood masks. FMD's index maps
    179 of these to an "FMD mask", but that fmd_mask_path is literally this
    same raw label file (verified by inspection: FMD's own readme.txt says
    "For images taken from the Roadway Flooding Image Dataset, we use the
    annotation path from the original dataset" -- not a distinct FMD
    product), so every row here uses the public CC BY 4.0 label directly and
    none are marked restricted.
    """
    images_dir = ROADWAY_FLOODING_DIR / "images"
    labels_dir = ROADWAY_FLOODING_DIR / "labels"
    rows = []
    for img_path in sorted(images_dir.glob("*.jpg")):
        stem = img_path.stem  # image_123
        idx = stem.split("_")[-1]
        label_path = labels_dir / f"label_{idx}.png"
        if not label_path.exists():
            continue
        rows.append({
            "source": "roadway_flooding",
            "orig_path": str(img_path),
            "mask_path": str(label_path),
            "mask_kind": "bottom_band01",
            "label": None,
            "label_source": "mask",
            "group_id": None,  # filled with dup_cluster after dedup
            "camera_id": None,
            "view_type": "ground",
            "license": LICENSE_ROADWAY_FLOODING,
            "restricted": False,
            "weak_label": None,
            "notes": "",
        })
    return rows


def iter_flood_area_segmentation() -> list[dict]:
    """290 images. Where the FMD index resolves an image (289 of them, via
    `local_image_path`), use FMD's cleaned {0,1} mask (restricted, since it
    is a distinct FMD-produced file, not the raw dataset's own Mask/*.png).
    The remainder falls back to the dataset's own Mask/*.png, which is a
    continuous grayscale PNG (JPEG-style compression noise near 0/255, not
    clean {0,1}), so it needs a threshold rather than a >0 test.

    NOT called by `build_manifest.gather_rows`: a Phase 2 visual audit (25
    random images plus the CLIP-margin extremes) found this source is
    almost entirely elevated drone/aerial flood footage, not the ground-level
    view this project targets (mean CLIP road-vs-aerial margin -0.017, max
    +0.057 across all 290 images -- see docs/phase_reports/phase2_prep.md).
    Left implemented and importable for documentation/reproducibility.
    """
    images_dir = FLOOD_AREA_DIR / "Image"
    own_masks_dir = FLOOD_AREA_DIR / "Mask"

    fmd_by_local_path: dict[str, dict] = {}
    for r in load_fmd_index():
        if r.get("source") == "Flood Area Segmentation" and r.get("local_image_path"):
            fmd_by_local_path[r["local_image_path"]] = r

    rows = []
    for img_path in sorted(images_dir.glob("*.jpg")):
        own_mask = own_masks_dir / f"{img_path.stem}.png"
        fmd_row = fmd_by_local_path.get(str(img_path))
        if fmd_row is not None:
            rows.append({
                "source": "flood_area_segmentation",
                "orig_path": str(img_path),
                "mask_path": fmd_row["fmd_mask_path"],
                "mask_kind": "bottom_band01",
                "label": None,
                "label_source": "mask",
                "group_id": None,
                "camera_id": None,
                "view_type": "ground",
                "license": LICENSE_FMD,
                "restricted": True,
                "weak_label": None,
                "notes": "label from Flood Master Database cleaned mask",
            })
        elif own_mask.exists():
            rows.append({
                "source": "flood_area_segmentation",
                "orig_path": str(img_path),
                "mask_path": str(own_mask),
                "mask_kind": "bottom_band_grayscale",
                "label": None,
                "label_source": "mask",
                "group_id": None,
                "camera_id": None,
                "view_type": "ground",
                "license": LICENSE_FLOOD_AREA_SEGMENTATION,
                "restricted": False,
                "weak_label": None,
                "notes": "no FMD match; used dataset's own Mask/*.png",
            })
    return rows


_FRED_SEQ_RE = re.compile(r"^(?P<loc>.+)_(?P<date>\d{8})_(?P<time>\d{6})$")


def _fred_group_id(seq_dir_name: str) -> str:
    m = _FRED_SEQ_RE.match(seq_dir_name)
    loc = m.group("loc") if m else seq_dir_name
    return normalize_group_name(loc)


def iter_fred() -> list[dict]:
    """FRED front-camera sequences. Flooded sequences get per-image labels
    from the road+water-hazard mask (2,674 of 2,726 flooded images have a
    label; the 52 without are excluded -- there is no safe sequence-level
    fallback for a flooded run, since water level varies frame to frame).
    Dry sequences ship no masks at all, so every dry frame is labeled `dry`
    via `sequence_condition`.
    """
    rows = []

    flooded_root = FRED_DIR / "flooded" / "KITTI-style"
    for seq_dir in sorted(p for p in flooded_root.iterdir() if p.is_dir()):
        group_id = _fred_group_id(seq_dir.name)
        imgs_dir = seq_dir / "front-imgs"
        labels_dir = seq_dir / "front-labels"
        if not imgs_dir.is_dir():
            continue
        for img_path in sorted(imgs_dir.glob("*.png")):
            label_path = labels_dir / img_path.name
            if not label_path.exists():
                continue
            rows.append({
                "source": "fred",
                "orig_path": str(img_path),
                "mask_path": str(label_path),
                "mask_kind": "fred",
                "label": None,
                "label_source": "mask",
                "group_id": group_id,
                "camera_id": None,
                "view_type": "vehicle",
                "license": LICENSE_FRED,
                "restricted": False,
                "weak_label": None,
                "notes": f"fred sequence {seq_dir.name}",
            })

    dry_root = FRED_DIR / "dry" / "KITTI-style"
    for seq_dir in sorted(p for p in dry_root.iterdir() if p.is_dir()):
        group_id = _fred_group_id(seq_dir.name)
        imgs_dir = seq_dir / "front-imgs"
        if not imgs_dir.is_dir():
            continue
        for img_path in sorted(imgs_dir.glob("*.png")):
            rows.append({
                "source": "fred",
                "orig_path": str(img_path),
                "mask_path": None,
                "mask_kind": None,
                "label": "dry",
                "label_source": "sequence_condition",
                "group_id": group_id,
                "camera_id": None,
                "view_type": "vehicle",
                "license": LICENSE_FRED,
                "restricted": False,
                "weak_label": None,
                "notes": f"fred sequence {seq_dir.name} (dry, no per-frame mask)",
            })
    return rows


def iter_nysdot(min_agreement: float = 0.67) -> list[dict]:
    """NYSDOT real DOT camera images with human dry/wet labels. Excludes
    snow, snow_severe, poor_vis, obstructed, and anything below the
    agreement threshold. Every kept image still needs `mask_rules.nysdot_crop`
    applied at resize time (build_manifest does this) to remove the
    label-leaking weather-metadata header.
    """
    rows = []
    labels_csv = NYSDOT_DIR / "labels.csv"
    with open(labels_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r["has_image"] != "1":
                continue
            label = r["majority_label"]
            if label not in ("dry", "wet"):
                continue
            if float(r["agreement_frac"]) < min_agreement:
                continue
            img_path = NYSDOT_DIR / r["rel_path"]
            if not img_path.exists():
                continue
            group_id = parse_nysdot_group_id(r["resolved_filename"])
            rows.append({
                "source": "nysdot_road_surface",
                "orig_path": str(img_path),
                "mask_path": None,
                "mask_kind": None,
                "label": label,
                "label_source": "dataset_label",
                "group_id": group_id,
                "camera_id": group_id,
                "view_type": "ground",
                "license": LICENSE_NYSDOT,
                "restricted": False,
                "weak_label": None,
                "notes": f"trial={r['trial']} agreement={r['agreement_frac']}",
                "needs_nysdot_crop": True,
            })
    return rows


def iter_fmd_test(include_sources: tuple[str, ...] | None = None) -> list[dict]:
    """Flood Master Database test videos: Greek (567 frames) and Italian
    (1,406 frames), 1280x720, each one group. No road-only mask exists, so
    water_frac_road uses the bottom-band prior over the FMD binary mask.

    Phase 2 re-verified the previous worker's blanket "both are aerial
    drone footage" call by looking at 20 frames spread across the Greek
    video and 24 across the Italian one (see docs/phase_reports/phase2_prep.md
    for the spot-check table):

    - Greek video: a FIXED, static-framing elevated camera (balcony/CCTV
      style, not panning or tilting like a drone) looking down a flooded
      street at submerged parked cars. Every sampled frame shows the same
      street from the same angle -- this is a real flooded-road scene, just
      from a higher, steeper vantage than a 511GA pole camera. Kept, label
      `flooded` from the bottom-band water mask.
    - Italian video: genuinely a drone/news b-roll clip (NYTimes on-screen
      credit, panning and tilting shots of vineyards and rooftops from a
      moving aircraft, no fixed framing). Excluded -- not a road-camera view.

    So only the Greek video is gathered by default; pass
    `include_sources=("greek video", "italian video")` to also get the
    (excluded-by-default) Italian rows for documentation/testing.
    """
    rows = []
    if include_sources is None:
        include_sources = ("greek video",)
    for r in load_fmd_index():
        if r.get("fmd_split") != "test":
            continue
        source_name = r["source"]
        if source_name not in include_sources:
            continue
        img_path = r["local_image_path"]
        if not img_path or not Path(img_path).exists():
            continue
        rows.append({
            "source": "flood_master_test",
            "orig_path": img_path,
            "mask_path": r["fmd_mask_path"],
            "mask_kind": "bottom_band01",
            "label": None,
            "label_source": "mask",
            "group_id": r["group_id"],
            "camera_id": r["group_id"],
            "view_type": "ground",
            "license": LICENSE_FMD,
            "restricted": True,
            "weak_label": None,
            "notes": f"fmd test frame, source={source_name}",
            "forced_split": "test",
        })
    return rows


def iter_ga511(frames_csv: Path, user_labels: dict[str, dict], ai_review_labels: dict[str, dict]) -> list[dict]:
    """511GA frames. `frames.csv` grows continuously while a separate
    collector runs, so we read it defensively: only rows with an empty
    `dead_reason` and an existing image file are used, and if the very last
    line is a partial write (collector was mid-append), it is dropped.

    `user_labels`/`ai_review_labels` are frame_id -> {"label": ..., ...},
    already reduced to "last write wins" by the caller. Manual labels always
    win over ai_review; both win over weak_precip.
    """
    rows = []
    with open(frames_csv, newline="") as f:
        text_rows = list(csv.DictReader(f))
    for i, r in enumerate(text_rows):
        # Defend against a partially-written last line from the concurrently
        # running collector: a good row always has frame_id and camera_id.
        if not r.get("frame_id") or not r.get("camera_id"):
            continue
        if r.get("dead_reason"):
            continue
        path = r.get("path") or ""
        if not path:
            continue
        img_path = Path("data/ga511") / path
        if not img_path.exists():
            continue

        frame_id = r["frame_id"]
        label = None
        label_source = None
        notes = ""
        if frame_id in user_labels and user_labels[frame_id].get("label"):
            ul = user_labels[frame_id]["label"]
            if ul in ("dry", "wet", "flooded"):
                label, label_source = ul, "manual"
        elif frame_id in ai_review_labels and ai_review_labels[frame_id].get("label"):
            al = ai_review_labels[frame_id]["label"]
            if al in ("wet", "flooded") and r.get("weak_label") == "likely_dry":
                # an ai_review "wet" on a frame with zero precip over the last
                # 6 h is almost always night headlight trails on dry pavement.
                # leave it unlabeled until the user confirms it in the tool.
                notes = f"ai_review {al} conflicts with weather likely_dry; needs manual check"
            elif al in ("dry", "wet", "flooded"):
                label, label_source = al, "ai_review"
                notes = ai_review_labels[frame_id].get("note", "")
        elif r.get("weak_label") == "likely_dry":
            label, label_source = "dry", "weak_precip"

        weak_label = r.get("weak_label") or None
        camera_id = r["camera_id"]
        rows.append({
            "source": "ga511",
            "orig_path": str(img_path),
            "mask_path": None,
            "mask_kind": None,
            "label": label,
            "label_source": label_source,
            "group_id": camera_id,
            "camera_id": camera_id,
            "view_type": "ground",
            "license": LICENSE_GA511,
            "restricted": False,
            "weak_label": weak_label,
            "notes": notes,
            "frame_id": frame_id,
        })
    return rows


def load_labels_csv(path: Path) -> dict[str, dict]:
    """Last-write-wins reduction of an append-only labels CSV
    (frame_id, label, labeled_at[, note])."""
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            fid = r.get("frame_id")
            if fid:
                out[fid] = r
    return out
