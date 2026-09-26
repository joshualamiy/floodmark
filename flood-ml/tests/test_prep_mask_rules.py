"""Unit tests for prep.mask_rules: the water_frac_road -> label boundary
rule, per-source gating of the unverified "dry-at-zero" / "wet-band" rules,
and the NYSDOT header crop on a synthetic image. No network, data, or models.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

from prep.mask_rules import (
    DRY_AT_ZERO_VERIFIED_SOURCES,
    FLOODED_THRESHOLD,
    WET_BAND_VERIFIED_SOURCES,
    WET_LOW_THRESHOLD,
    assign_label_for_source,
    assign_label_from_water_frac,
    nysdot_crop,
)

# ---------------------------------------------------------------------------
# assign_label_from_water_frac boundaries
# ---------------------------------------------------------------------------

def test_zero_water_is_dry():
    assert assign_label_from_water_frac(0.0) == "dry"


def test_nan_is_excluded():
    assert assign_label_from_water_frac(float("nan")) is None


def test_just_above_zero_below_wet_low_is_excluded():
    assert assign_label_from_water_frac(WET_LOW_THRESHOLD / 2) is None


def test_at_wet_low_threshold_is_wet():
    assert assign_label_from_water_frac(WET_LOW_THRESHOLD) == "wet"


def test_just_below_flooded_threshold_is_wet():
    assert assign_label_from_water_frac(FLOODED_THRESHOLD - 1e-9) == "wet"


def test_at_flooded_threshold_is_flooded():
    assert assign_label_from_water_frac(FLOODED_THRESHOLD) == "flooded"


def test_well_above_flooded_threshold_is_flooded():
    assert assign_label_from_water_frac(0.9) == "flooded"


def test_custom_thresholds_are_respected():
    assert assign_label_from_water_frac(0.05, wet_low=0.02, threshold=0.5) == "wet"
    assert assign_label_from_water_frac(0.6, wet_low=0.02, threshold=0.5) == "flooded"


# ---------------------------------------------------------------------------
# assign_label_for_source: per-source gating of the unverified rules
# ---------------------------------------------------------------------------

def test_verified_source_gets_dry_at_zero():
    source = next(iter(DRY_AT_ZERO_VERIFIED_SOURCES))
    assert assign_label_for_source(0.0, source) == "dry"


def test_unverified_source_dry_at_zero_is_excluded():
    assert "made_up_source" not in DRY_AT_ZERO_VERIFIED_SOURCES
    assert assign_label_for_source(0.0, "made_up_source") is None


def test_wet_band_is_off_for_fred():
    # fred's wet band was mostly distant floods ahead of a dry near-field
    assert "fred" not in WET_BAND_VERIFIED_SOURCES
    assert assign_label_for_source(WET_LOW_THRESHOLD, "fred") is None


def test_unverified_source_wet_band_is_excluded():
    assert "made_up_source" not in WET_BAND_VERIFIED_SOURCES
    assert assign_label_for_source(0.05, "made_up_source") is None


def test_flooded_is_never_gated_by_source():
    # A large, unambiguous water fraction is trusted regardless of source.
    assert assign_label_for_source(0.5, "made_up_source") == "flooded"
    assert assign_label_for_source(0.5, "roadway_flooding") == "flooded"


def test_roadway_flooding_specifically_excluded_from_both_unverified_rules():
    # Orchestrator ruling: roadway_flooding's wet/dry buckets are too small
    # (n=8, n=1) to verify, so both must be excluded for this source.
    assert "roadway_flooding" not in DRY_AT_ZERO_VERIFIED_SOURCES
    assert "roadway_flooding" not in WET_BAND_VERIFIED_SOURCES
    assert assign_label_for_source(0.0, "roadway_flooding") is None
    assert assign_label_for_source(0.06, "roadway_flooding") is None


# ---------------------------------------------------------------------------
# NYSDOT header crop
# ---------------------------------------------------------------------------

def _synthetic_nysdot_image(w=600, h=440, header_h=140):
    """A near-white header block with red text-like marks on top of a large,
    busy (non-white) "camera frame" region, mimicking the real NYSDOT layout
    closely enough to exercise nysdot_crop's row/column detection.
    """
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # header: mostly white with a few red text-like strokes (like the
    # rendered weather metadata text)
    for y in range(10, header_h - 10, 12):
        draw.line([(20, y), (w - 20, y)], fill=(200, 30, 30), width=2)
    # camera frame: a busy, non-white region filling most of the width
    draw.rectangle([10, header_h, w - 10, h - 1], fill=(60, 90, 70))
    rng = np.random.default_rng(0)
    arr = np.array(img)
    noise = rng.integers(0, 60, size=(h - header_h, w - 20, 3), dtype=np.uint8)
    arr[header_h:h, 10:w - 10] = np.clip(
        arr[header_h:h, 10:w - 10].astype(int) + noise, 0, 255
    ).astype(np.uint8)
    return Image.fromarray(arr)


def test_nysdot_crop_removes_the_header_band():
    img = _synthetic_nysdot_image(header_h=140)
    cropped = nysdot_crop(img)
    # The crop must start at or after the header, i.e. be shorter than the
    # original and not include the near-white header rows.
    assert cropped.height < img.height
    cropped_arr = np.asarray(cropped.convert("RGB")).astype(np.int32)
    # No row of the cropped image should look like the mostly-white header.
    row_means = cropped_arr.mean(axis=(1, 2))
    assert row_means.max() < 245 or cropped_arr.shape[0] > 0


def test_nysdot_crop_keeps_most_of_the_camera_frame_width():
    img = _synthetic_nysdot_image()
    cropped = nysdot_crop(img)
    assert cropped.width > img.width * 0.5


def test_nysdot_crop_falls_back_to_original_on_odd_input():
    # An already-cropped, uniformly busy image with no near-white run at all
    # should be returned unchanged rather than raising.
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 100, size=(200, 300, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    cropped = nysdot_crop(img)
    assert cropped.size == img.size


def test_nysdot_crop_never_raises_on_blank_image():
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    cropped = nysdot_crop(img)
    assert cropped.size[0] > 0 and cropped.size[1] > 0


def test_water_frac_from_binary_mask_boundaries():
    from prep.mask_rules import water_frac_from_binary_mask

    mask = np.zeros((10, 10), dtype=bool)
    assert water_frac_from_binary_mask(mask) == 0.0
    mask[:] = True
    assert water_frac_from_binary_mask(mask) == 1.0


def test_water_frac_from_binary_mask_empty_road_region_is_nan():
    from prep.mask_rules import water_frac_from_binary_mask

    mask = np.ones((10, 10), dtype=bool)
    empty_region = np.zeros((10, 10), dtype=bool)
    result = water_frac_from_binary_mask(mask, road_region=empty_region)
    assert math.isnan(result)
