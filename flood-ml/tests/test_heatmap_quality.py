import csv

import numpy as np
import pytest
from PIL import Image

from eval import heatmap_quality as h
from inference.preprocess import preprocess


@pytest.mark.parametrize("mode", ["crop", "squash", "letterbox"])
def test_mask_follows_input_geometry(mode):
    mask = np.zeros((40, 80), dtype=bool)
    mask[:, 40:] = True
    _, geom = preprocess(Image.new("RGB", (80, 40)), mode=mode, size=32,
                         do_jpeg_roundtrip=False)
    water, valid = h.align_mask(mask, geom)
    assert water.shape == valid.shape == (32, 32)
    assert water[valid].mean() == pytest.approx(.5)
    assert not water[~valid].any()
    assert valid.sum() == (512 if mode == "letterbox" else 1024)


def test_crop_padding_is_excluded():
    _, geom = preprocess(Image.new("RGB", (8, 4)), size=16, do_jpeg_roundtrip=False)
    water, valid = h.align_mask(np.ones((4, 8)), geom)
    assert valid.sum() == water.sum() == 32
    with pytest.raises(ValueError, match="aspect mismatch"):
        h.align_mask(np.ones((4, 4)), geom)


def test_flat_cam_gets_area_baseline_and_zero_cam_is_unavailable():
    water = np.zeros((4, 4), dtype=bool)
    water[:1] = True
    value = h.cam_metrics(np.ones((4, 4)), water)
    assert value["water_energy_fraction"] == value["peak_in_water"] == .25
    assert value["energy_minus_area"] == 0
    assert h.cam_metrics(np.zeros((4, 4)), water)["reason"] == "zero_cam"
    valid = ~water
    assert h.cam_metrics(water.astype(float), water, valid)["reason"] == "no_content_energy"
    for bad in (np.full((4, 4), np.nan), -np.ones((4, 4))):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            h.cam_metrics(bad, water)


def test_occlusion_control_rectangles_have_same_area_and_avoid_padding():
    valid = np.zeros((20, 20), dtype=bool)
    valid[5:15] = True
    cam = np.zeros((20, 20), dtype=float)
    cam[5:9, :4] = 1
    boxes = h.occlusion_boxes(cam, valid, .1, np.random.default_rng(7), 5)
    assert len(boxes) == 7
    assert len({(r - l) * (b - t) for l, t, r, b in boxes}) == 1
    for left, top, right, bottom in boxes:
        assert valid[top:bottom, left:right].all()
    assert boxes == h.occlusion_boxes(cam, valid, .1, np.random.default_rng(7), 5)


def test_occlusion_detects_known_patch_dependence():
    image = np.zeros((20, 20, 3), dtype=np.float32)
    image[:6, :6] = 255
    cam = image[..., 0]

    def score(batch):
        return .1 + .8 * batch[:, :6, :6].mean(axis=(1, 2, 3)) / 255

    result = h.occlusion_diagnostic(score, image, cam, np.ones((20, 20), bool), .9,
                                    seed=7, fraction=.09)
    assert result["high_minus_random_logit_drop"] > 0
    assert result["high_minus_low_logit_drop"] > 0
    assert h.occlusion_diagnostic(score, image, np.ones((2, 2)),
                                  np.ones((20, 20), bool), .9, seed=7)["reason"] == "flat_cam"


def test_empty_segmentation_not_reported_as_perfect_overlap():
    empty = np.zeros((3, 3), bool)
    result = h.segmentation_metrics(empty, empty)
    assert result["iou"] is None and result["dice"] is None
    assert result["both_empty"]
    result = h.segmentation_metrics(~empty, empty)
    assert result["iou"] == result["dice"] == 0
    assert result["fp"] == 9


def test_manifest_excludes_train_test_and_unmasked_rows(tmp_path):
    path = tmp_path / "manifest.csv"
    fields = ["split", "source", "orig_path", "mask_path", "label"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for split, mask in [("train", "m"), ("test", "m"), ("val", "m"), ("val", "")]:
            writer.writerow(dict(zip(fields, [split, "fred", "image", mask, "dry"])))
    rows, coverage = h.read_validation_rows(path)
    assert len(rows) == 1 and rows[0]["split"] == "val"
    assert coverage["validation_without_mask"] == 1


def test_mask_classes_are_source_specific(tmp_path):
    path = tmp_path / "mask.png"
    Image.fromarray(np.array([[0, 1, 2]], dtype=np.uint8)).save(path)
    assert h.load_water_mask(path, "fred").tolist() == [[False, False, True]]
    with pytest.raises(ValueError, match="binary"):
        h.load_water_mask(path, "roadway_flooding")

