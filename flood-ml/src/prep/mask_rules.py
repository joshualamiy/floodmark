# water fraction on the road region -> label, plus the nysdot header crop
from __future__ import annotations

import math

import numpy as np
from PIL import Image

from prep.common import road_bottom_trapezoid_mask

# water fraction of the road region, picked by spot-checking
WET_LOW_THRESHOLD = 0.04
FLOODED_THRESHOLD = 0.10

DRY_AT_ZERO_VERIFIED_SOURCES = frozenset({"fred"})
# fred's wet band was distant floods, not wet roads, so it's off
WET_BAND_VERIFIED_SOURCES: frozenset[str] = frozenset()


def water_frac_from_binary_mask(mask: np.ndarray, road_region: np.ndarray | None = None) -> float:
    water = mask.astype(bool)
    if road_region is None:
        road_region = np.ones_like(water, dtype=bool)
    denom = int(road_region.sum())
    if denom == 0:
        return float("nan")
    return float((water & road_region).sum()) / denom


def water_frac_bottom_band(mask_path_or_array, top_frac: float = 0.40) -> float:
    if isinstance(mask_path_or_array, np.ndarray):
        arr = mask_path_or_array
    else:
        arr = np.array(Image.open(mask_path_or_array))
    if arr.ndim == 3:
        arr = arr[..., 0]
    water = arr > 0
    h, w = water.shape
    road_region = road_bottom_trapezoid_mask(h, w, top_frac=top_frac)
    return water_frac_from_binary_mask(water, road_region)


def water_frac_grayscale_mask(mask_path, threshold: int = 127, top_frac: float = 0.40) -> float:
    arr = np.array(Image.open(mask_path).convert("L"))
    water = arr > threshold
    h, w = water.shape
    road_region = road_bottom_trapezoid_mask(h, w, top_frac=top_frac)
    return water_frac_from_binary_mask(water, road_region)


def water_frac_fred(label_path) -> float:
    arr = np.array(Image.open(label_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    road_region = (arr == 1) | (arr == 2)
    water = arr == 2
    return water_frac_from_binary_mask(water, road_region)


def assign_label_from_water_frac(
    water_frac: float,
    wet_low: float = WET_LOW_THRESHOLD,
    threshold: float = FLOODED_THRESHOLD,
) -> str | None:
    if math.isnan(water_frac):
        return None
    if water_frac <= 0.0:
        return "dry"
    if water_frac < wet_low:
        return None
    if water_frac >= threshold:
        return "flooded"
    return "wet"


def assign_label_for_source(
    water_frac: float,
    source: str,
    wet_low: float = WET_LOW_THRESHOLD,
    threshold: float = FLOODED_THRESHOLD,
) -> str | None:
    label = assign_label_from_water_frac(water_frac, wet_low=wet_low, threshold=threshold)
    if label == "dry" and source not in DRY_AT_ZERO_VERIFIED_SOURCES:
        return None
    if label == "wet" and source not in WET_BAND_VERIFIED_SOURCES:
        return None
    return label


def nysdot_crop(img: Image.Image, row_thresh: float = 0.45, col_thresh: float = 0.5,
                 min_run_frac: float = 0.25) -> Image.Image:
    rgb = img.convert("RGB")
    arr = np.asarray(rgb).astype(np.int32)
    gray = arr.mean(axis=2)
    nonbg = gray < 245

    row_frac = nonbg.mean(axis=1)
    height = row_frac.shape[0]
    min_run = max(1, int(height * min_run_frac))

    runs: list[tuple[int, int]] = []
    start = None
    for y, f in enumerate(row_frac):
        if f >= row_thresh:
            if start is None:
                start = y
        else:
            if start is not None:
                runs.append((start, y))
                start = None
    if start is not None:
        runs.append((start, height))

    runs = [r for r in runs if (r[1] - r[0]) >= min_run]
    if not runs:
        return img

    y0, y1 = max(runs, key=lambda r: r[1] - r[0])

    sub = nonbg[y0:y1]
    col_frac = sub.mean(axis=0)
    xs = np.where(col_frac >= col_thresh)[0]
    if len(xs) == 0:
        x0, x1 = 0, arr.shape[1]
    else:
        x0, x1 = int(xs.min()), int(xs.max()) + 1

    return img.crop((x0, y0, x1, y1))


def parse_nysdot_group_id(resolved_filename: str) -> str:
    import re
    m = re.match(r"^(.*)_\d{4}-\d{2}-\d{2}-\d{2}_\d{2}_\d{2}\.[A-Za-z0-9]+$", resolved_filename)
    if m:
        return m.group(1)
    return resolved_filename.rsplit(".", 1)[0]

