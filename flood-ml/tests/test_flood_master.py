from __future__ import annotations

import numpy as np

from acquire import flood_master as fm


def test_hamming_distance():
    assert fm._hamming(0b0000, 0b0000) == 0
    assert fm._hamming(0b0000, 0b1111) == 4
    assert fm._hamming(0b1010, 0b0101) == 4


def test_band_keys_length_and_values():
    value = 0x0102030405060708
    bands = fm._band_keys(value, n_bands=8, band_bits=8)
    assert len(bands) == 8
    assert bands[0] == (0, 0x08)


def test_bucketed_overlap_search_finds_near_duplicate():
    base = 0x0F0F0F0F0F0F0F0F
    near = base ^ 0b111
    far = base ^ 0xFFFFFFFF

    corpus = {"near.jpg": near, "far.jpg": far}
    query = {"query.jpg": base}

    hits = fm.bucketed_overlap_search(query, corpus, threshold=6)
    assert "query.jpg" in hits
    matched_paths = {p for p, _dist in hits["query.jpg"]}
    assert "near.jpg" in matched_paths
    assert "far.jpg" not in matched_paths


def test_bucketed_overlap_search_empty_when_no_match():
    corpus = {"far.jpg": 0}
    query = {"query.jpg": 0xFFFFFFFFFFFFFFFF}
    hits = fm.bucketed_overlap_search(query, corpus, threshold=6)
    assert hits == {}


def test_iou_identical_masks_is_one():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    assert fm._iou(mask, mask) == 1.0


def test_iou_disjoint_masks_is_zero():
    a = np.zeros((10, 10), dtype=bool)
    a[0:5, 0:5] = True
    b = np.zeros((10, 10), dtype=bool)
    b[5:10, 5:10] = True
    assert fm._iou(a, b) == 0.0


def test_iou_resizes_mismatched_shapes():
    a = np.ones((4, 4), dtype=bool)
    b = np.ones((8, 8), dtype=bool)
    assert fm._iou(a, b) == 1.0


def test_resolve_train_val_annotation_roadway_uses_raw_labels():
    row = {"Annotation path": "Dataset/labels/label_1.png", "Source": "Roadway Flooding Image Dataset"}
    resolved = fm.resolve_train_val_annotation("train", row)
    assert resolved == fm.ROADWAY_ROOT / "Dataset/labels/label_1.png"


def test_resolve_train_val_annotation_default_uses_fmd_root():
    row = {"Annotation path": "annotations/1Ids.png", "Source": "Flood Area Segmentation"}
    resolved = fm.resolve_train_val_annotation("train", row)
    assert resolved == fm.FMD_ROOT / "train" / "annotations/1Ids.png"


def test_resolve_test_rgb_picks_subfolder_by_source():
    greek_row = {"Source": "greek video", "Image path": "frame1.jpg"}
    italian_row = {"Source": "italian video", "Image path": "frame1.jpg"}
    assert fm.resolve_test_rgb(greek_row) == fm.FMD_ROOT / "test/greek_test/rgb/frame1.jpg"
    assert fm.resolve_test_rgb(italian_row) == fm.FMD_ROOT / "test/italian_test/rgb/frame1.jpg"


def test_test_group_id_one_group_per_video():
    assert fm.test_group_id({"Source": "greek video"}) == "fmd_greek_video"
    assert fm.test_group_id({"Source": "italian video"}) == "fmd_italian_video"

