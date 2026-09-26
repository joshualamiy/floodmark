# prep constants + image helpers
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(".")
RAW_ROOT = Path("data/raw")
RESTRICTED_ROOT = Path("data/restricted")
GA511_ROOT = Path("data/ga511")
PROCESSED_ROOT = Path("data/processed")
IMAGES_ROOT = PROCESSED_ROOT / "images"
MANIFEST_PATH = PROCESSED_ROOT / "manifest.csv"
VERSION_PATH = PROCESSED_ROOT / "VERSION"
SPLITS_REPORT_PATH = PROCESSED_ROOT / "splits_report.json"
CAMERA_SPLITS_PATH = PROCESSED_ROOT / "ga511_camera_splits.json"
LOGS_JOBS = Path("logs/jobs")

REQUIRED_COLUMNS = [
    "path", "label", "source", "group_id", "camera_id", "view_type",
    "license", "restricted",
]
EXTRA_COLUMNS = [
    "split", "label_source", "orig_path", "mask_path", "water_frac_road",
    "phash", "width", "height", "dup_cluster", "weak_label", "notes",
]
MANIFEST_COLUMNS = REQUIRED_COLUMNS + EXTRA_COLUMNS

LABELS = ("dry", "wet", "flooded", "not_flooded")
LABEL_SOURCES = ("mask", "sequence_condition", "manual", "weak_precip", "dataset_label", "ai_review")
SPLITS = ("train", "val", "test")

LICENSE_ROADWAY_FLOODING = "CC BY 4.0 (Roadway Flooding Image Dataset, Sazara/Cetin/Iftekharuddin 2019)"
LICENSE_FLOOD_AREA_SEGMENTATION = "CC0 1.0 (Flood Area Segmentation, Kaggle faizalkarim)"
LICENSE_FMD = "Flood Master Database: non-commercial research, no redistribution (AIIA Lab, AUTH)"
LICENSE_FRED = "CC BY-NC-SA 4.0 (FRED, CMalone-Jupiter/FRED)"
LICENSE_NYSDOT = "CC BY 4.0 (NYSDOT Road Surface Conditions, Zenodo 10.5281/zenodo.8370665)"
LICENSE_GA511 = "511GA (Georgia DOT) public traffic camera feed; internal research use"
LICENSE_IOWA_RWIS = "Public domain (Iowa Environmental Mesonet)"
LICENSE_ALLEYFLOODNET = "CC BY 4.0 (dataset-level; some photos carry third-party watermarks, per-photo rights unverified)"

RESIZE_SHORT_SIDE = 256
JPEG_QUALITY = 95


def resize_short_side(img: Image.Image, short_side: int = RESIZE_SHORT_SIDE) -> Image.Image:
    w, h = img.size
    if min(w, h) <= short_side:
        return img
    if w < h:
        new_w = short_side
        new_h = round(h * short_side / w)
    else:
        new_h = short_side
        new_w = round(w * short_side / h)
    return img.resize((new_w, new_h), Image.LANCZOS)


def save_processed_image(img: Image.Image, dest: Path) -> tuple[int, int]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = img.convert("RGB")
    img = resize_short_side(img)
    img.save(dest, format="JPEG", quality=JPEG_QUALITY)
    return img.size


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash_fraction(key: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()
    return int(h[:8], 16) / 0x1_0000_0000


def normalize_group_name(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "")


def setup_job_logger(name: str) -> logging.Logger:
    LOGS_JOBS.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"prep.{name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(LOGS_JOBS / f"{name}.log", mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


def data_version_string(manifest_path: Path = MANIFEST_PATH) -> str:
    digest = sha256_file(manifest_path)[:8]
    return f"data_version = v1-{digest}"


def phash_hex_to_int(phash_hex: str) -> int:
    return int(phash_hex, 16)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def road_bottom_trapezoid_mask(height: int, width: int, top_frac: float = 0.40) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    y0 = int(height * top_frac)
    for y in range(y0, height):
        t = (y - y0) / max(1, (height - 1 - y0))
        margin = int(width * 0.5 * (1 - t) * 0.5)
        mask[y, margin: width - margin] = True
    return mask

