# per-source row builders for the manifest
from __future__ import annotations

import csv
import re
from pathlib import Path

from prep.common import (
    LICENSE_ALLEYFLOODNET,
    LICENSE_FLOOD_AREA_SEGMENTATION,
    LICENSE_FMD,
    LICENSE_FRED,
    LICENSE_GA511,
    LICENSE_IOWA_RWIS,
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
IOWA_RWIS_DIR = Path("data/othercams/iowa_rwis")
EU_FLOOD_2013_DIR = RAW_ROOT / "eu_flood_2013"
ALLEYFLOODNET_DIR = RAW_ROOT / "alleyfloodnet"

EXCLUDED_FMD_SOURCES = {"Water Dataset"}

EU_FLOOD_ROAD_SCORE_THRESHOLD = 0.0

INCLUDE_ALLEYFLOODNET = True


def load_fmd_index() -> list[dict]:
    if not FMD_INDEX_PATH.exists():
        return []
    with open(FMD_INDEX_PATH, newline="") as f:
        return list(csv.DictReader(f))


def iter_roadway_flooding() -> list[dict]:
    images_dir = ROADWAY_FLOODING_DIR / "images"
    labels_dir = ROADWAY_FLOODING_DIR / "labels"
    rows = []
    for img_path in sorted(images_dir.glob("*.jpg")):
        stem = img_path.stem
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
            "group_id": None,
            "camera_id": None,
            "view_type": "ground",
            "license": LICENSE_ROADWAY_FLOODING,
            "restricted": False,
            "weak_label": None,
            "notes": "",
        })
    return rows


def iter_flood_area_segmentation() -> list[dict]:
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


def iter_ga511(
    frames_csv: Path,
    user_labels: dict[str, dict],
    ai_review_labels: dict[str, dict],
    camera_splits: dict[str, str] | None = None,
) -> list[dict]:
    camera_splits = camera_splits or {}
    rows = []
    with open(frames_csv, newline="") as f:
        text_rows = list(csv.DictReader(f))
    for i, r in enumerate(text_rows):
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
        camera_id = r["camera_id"]
        label = None
        label_source = None
        notes = ""
        if frame_id in user_labels and user_labels[frame_id].get("label"):
            ul = user_labels[frame_id]["label"]
            if ul in ("dry", "wet", "flooded"):
                label, label_source = ul, "manual"
            elif ul == "unusable" and camera_splits.get(camera_id) in ("train", "val"):
                # unusable on train/val cams -> stage b negative
                label, label_source = "not_flooded", "manual"
                notes = "manual unusable on train/val camera -> not_flooded (stage b negative)"
        elif frame_id in ai_review_labels and ai_review_labels[frame_id].get("label"):
            al = ai_review_labels[frame_id]["label"]
            # ai 'wet' on a 0 mm rain frame was night headlight glare, wait for a human label
            if al in ("wet", "flooded") and r.get("weak_label") == "likely_dry":
                notes = f"ai_review {al} conflicts with weather likely_dry; needs manual check"
            elif al in ("dry", "wet", "flooded"):
                label, label_source = al, "ai_review"
                notes = ai_review_labels[frame_id].get("note", "")
        elif r.get("weak_label") == "likely_dry":
            label, label_source = "dry", "weak_precip"

        weak_label = r.get("weak_label") or None
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


def iter_iowa_rwis() -> list[dict]:
    frames_csv = IOWA_RWIS_DIR / "frames.csv"
    labels_csv = IOWA_RWIS_DIR / "labels.csv"
    if not frames_csv.exists():
        return []
    user_labels = load_labels_csv(labels_csv)
    rows = []
    with open(frames_csv, newline="") as f:
        text_rows = list(csv.DictReader(f))
    for r in text_rows:
        if not r.get("frame_id") or not r.get("camera_id"):
            continue
        if r.get("dead_reason"):
            continue
        fid = r["frame_id"]
        ul = user_labels.get(fid)
        if not ul or not ul.get("label"):
            continue
        label = ul["label"]
        if label not in ("dry", "wet", "flooded"):
            continue
        path = r.get("path") or ""
        if not path:
            continue
        img_path = IOWA_RWIS_DIR / path
        if not img_path.exists():
            continue
        camera_id = r["camera_id"]
        rows.append({
            "source": "iowa_rwis",
            "orig_path": str(img_path),
            "mask_path": None,
            "mask_kind": None,
            "label": label,
            "label_source": "manual",
            "group_id": camera_id,
            "camera_id": camera_id,
            "view_type": "ground",
            "license": LICENSE_IOWA_RWIS,
            "restricted": False,
            "weak_label": r.get("weak_label") or None,
            "notes": "",
            "frame_id": fid,
        })
    return rows


def iter_eu_flood_2013(road_score_threshold: float = EU_FLOOD_ROAD_SCORE_THRESHOLD) -> list[dict]:
    candidates_csv = EU_FLOOD_2013_DIR / "candidates.csv"
    if not candidates_csv.exists():
        return []
    rows = []
    with open(candidates_csv, newline="") as f:
        for r in csv.DictReader(f):
            label = r.get("label")
            if label not in ("flooded", "not_flooded"):
                continue
            license_str = r.get("license") or ""
            if "unverified" in license_str.lower():
                continue
            try:
                road_score = float(r["road_score"])
            except (TypeError, ValueError):
                continue
            if road_score <= road_score_threshold:
                continue
            img_path = Path(r["path"])
            if not img_path.exists():
                continue
            group_id = r.get("group_id") or "euflood_unknown"
            rows.append({
                "source": "eu_flood_2013",
                "orig_path": str(img_path),
                "mask_path": None,
                "mask_kind": None,
                "label": label,
                "label_source": "dataset_label",
                "group_id": group_id,
                "camera_id": None,
                "view_type": "ground",
                "license": license_str,
                "restricted": False,
                "weak_label": None,
                "notes": f"road_score={road_score:.4f}",
            })
    return rows


def iter_alleyfloodnet(include: bool = INCLUDE_ALLEYFLOODNET) -> list[dict]:
    if not include:
        return []
    candidates_csv = ALLEYFLOODNET_DIR / "candidates.csv"
    if not candidates_csv.exists():
        return []
    rows = []
    with open(candidates_csv, newline="") as f:
        for r in csv.DictReader(f):
            label = r.get("label")
            if label not in ("flooded", "not_flooded"):
                continue
            img_path = Path(r["path"])
            if not img_path.exists():
                continue
            rows.append({
                "source": "alleyfloodnet",
                "orig_path": str(img_path),
                "mask_path": None,
                "mask_kind": None,
                "label": label,
                "label_source": "dataset_label",
                "group_id": None,
                "camera_id": None,
                "view_type": "ground",
                "license": LICENSE_ALLEYFLOODNET,
                "restricted": False,
                "weak_label": None,
                "notes": "",
            })
    return rows


def load_labels_csv(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            fid = r.get("frame_id")
            if fid:
                out[fid] = r
    return out

