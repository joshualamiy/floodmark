"""Road-region priors and water_frac_road computation.

An image is labeled `flooded` only if water covers a meaningful part of the
ROAD region, not the whole frame. Where a dataset ships a road mask (FRED:
road label + water-hazard label), the road region is road union
water-hazard. Everywhere else there is no road mask, so we fall back to a
documented prior: the bottom band of the frame (a trapezoid narrowing toward
the horizon), see `common.road_bottom_trapezoid_mask`. The flooded threshold
on `water_frac_road` is chosen by spot-checking (see the phase report).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

from prep.common import road_bottom_trapezoid_mask

# Chosen by spot-checking >= 100 images (see docs/phase_reports/phase2_prep.md).
# Below WET_LOW_THRESHOLD, "some water" is excluded -- usually mask noise or
# a puddle too small to see. Between WET_LOW_THRESHOLD and FLOODED_THRESHOLD,
# a first pass that excluded everything below FLOODED_THRESHOLD was spot
# checked and revised: most images in that band are genuinely wet (rain
# sheen, standing puddles) but clearly not "flooded" -- exactly the
# wet-but-not-flooded data PLAN.md section 7 flags as scarce -- so they are
# labeled `wet` instead of dropped.
WET_LOW_THRESHOLD = 0.04
FLOODED_THRESHOLD = 0.10

# ---------------------------------------------------------------------------
# Orchestrator ruling (Phase 2 resume): the "wet band" (wet_low <= f <
# threshold -> "wet") and "dry at zero water" (f <= 0 -> "dry") rules above
# are only trusted, per source, once >= 20 images of that (source, label)
# bucket have actually been looked at and >= 80% visibly match the label.
# Where a source isn't in the matching set below, `compute_labels` drops
# those rows instead of labeling them. See the per-(source, label) spot-check
# table in docs/phase_reports/phase2_prep.md for the counts and agreement
# that earned each source its place here.
#
# - fred: wet band verified (164 images, 20 viewed, ~85% clearly show some
#   water on the road corridor -- typically a dry near-field with a flooded
#   stretch visible ahead under a "FLOODING" warning sign); dry-at-zero
#   verified (199 images, 20 viewed, 100% clean dry road, zero water anywhere
#   in frame).
# - roadway_flooding: wet band has only 8 images total (fails the n>=20 bar
#   before agreement is even considered) and dry-at-zero has exactly 1 (which,
#   on inspection, is actually a "ROAD CLOSED" sign on a road leading toward
#   standing water -- a flood-adjacent scene, not a dry road, exactly the
#   failure mode the ruling warns about). Both excluded for this source.
# - flood_area_segmentation: not gathered at all (excluded wholesale, see
#   build_manifest.py) so its buckets are moot.
# - flood_master_test (Greek video only): every frame is heavily flooded
#   (water_frac_road far above FLOODED_THRESHOLD); no wet-band or dry-at-zero
#   rows are currently produced, so nothing to verify, but the source is left
#   out of both sets below so any such row is excluded rather than silently
#   trusted if the video's mask ever produces one.
#
# orchestrator review: the fred wet band was dropped after all. most of those
# frames are a dry near-field with a flooded crossing further ahead, i.e. a
# distant flood, not a wet road surface. training them as "not flooded" would
# teach stage b to ignore floods that aren't close to the camera.
DRY_AT_ZERO_VERIFIED_SOURCES = frozenset({"fred"})
WET_BAND_VERIFIED_SOURCES: frozenset[str] = frozenset()


def water_frac_from_binary_mask(mask: np.ndarray, road_region: np.ndarray | None = None) -> float:
    """`mask` is a 2D array where nonzero == water. If `road_region` is None,
    the whole frame is used as the road region (only appropriate when the
    dataset's images are themselves tight road/flood crops).
    """
    water = mask.astype(bool)
    if road_region is None:
        road_region = np.ones_like(water, dtype=bool)
    denom = int(road_region.sum())
    if denom == 0:
        return float("nan")
    return float((water & road_region).sum()) / denom


def water_frac_bottom_band(mask_path_or_array, top_frac: float = 0.40) -> float:
    """Water fraction inside the bottom-band road-region prior. Accepts a
    path to a mask PNG (decoded, nonzero == water) or an ndarray.
    """
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
    """For masks stored as continuous grayscale (e.g. Flood Area
    Segmentation's own JPEG-compressed Mask/*.png, values cluster near 0/255
    with some compression noise) rather than clean {0,1}.
    """
    arr = np.array(Image.open(mask_path).convert("L"))
    water = arr > threshold
    h, w = water.shape
    road_region = road_bottom_trapezoid_mask(h, w, top_frac=top_frac)
    return water_frac_from_binary_mask(water, road_region)


def water_frac_fred(label_path) -> float:
    """FRED label PNG: 0=other, 1=road, 2=water hazard. Road region = road
    union water-hazard, per the Phase 2 brief.
    """
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
    """`water_frac` is the fraction of the road region flagged as water.

    - water_frac <= 0: `dry`
    - 0 < water_frac < wet_low: excluded (None) -- too small to trust, most
      likely mask noise or a puddle too small to see
    - wet_low <= water_frac < threshold: `wet` (spot-checked: this band is
      genuinely wet, non-flooded road in FRED and Roadway Flooding)
    - water_frac >= threshold: `flooded`
    """
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
    """Same as `assign_label_from_water_frac`, but gates the two unverified
    rules (dry-at-zero, wet-band) per source per the orchestrator ruling:
    a source not in `DRY_AT_ZERO_VERIFIED_SOURCES`/`WET_BAND_VERIFIED_SOURCES`
    has those outcomes excluded (returns None) rather than labeled. `flooded`
    is never gated this way (a large, unambiguous water fraction doesn't rest
    on an unverified small-sample rule).
    """
    label = assign_label_from_water_frac(water_frac, wet_low=wet_low, threshold=threshold)
    if label == "dry" and source not in DRY_AT_ZERO_VERIFIED_SOURCES:
        return None
    if label == "wet" and source not in WET_BAND_VERIFIED_SOURCES:
        return None
    return label


# ---------------------------------------------------------------------------
# NYSDOT header crop
# ---------------------------------------------------------------------------

def nysdot_crop(img: Image.Image, row_thresh: float = 0.45, col_thresh: float = 0.5,
                 min_run_frac: float = 0.25) -> Image.Image:
    """Crop an NYSDOT image down to the embedded camera frame, removing the
    rendered red-text weather header above it (a label leak: it states
    precipitation, snow depth, and time-since-precip directly).

    The header sits on a white/light background; the camera frame itself is
    a large, visually busy region that is non-white across nearly its full
    width for many consecutive rows. We find the tallest contiguous run of
    rows where a high fraction of pixels are non-background, then trim left
    and right margins within that row range the same way. Falls back to the
    original image if no such run is found (already-cropped input, or an
    unexpected layout) so this never raises on odd inputs.
    """
    rgb = img.convert("RGB")
    arr = np.asarray(rgb).astype(np.int32)
    gray = arr.mean(axis=2)
    nonbg = gray < 245  # not near-white

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
    """Camera name = text before the trailing _YYYY-MM-DD-HH_MM_SS timestamp."""
    import re
    m = re.match(r"^(.*)_\d{4}-\d{2}-\d{2}-\d{2}_\d{2}_\d{2}\.[A-Za-z0-9]+$", resolved_filename)
    if m:
        return m.group(1)
    # fallback: strip extension only
    return resolved_filename.rsplit(".", 1)[0]
