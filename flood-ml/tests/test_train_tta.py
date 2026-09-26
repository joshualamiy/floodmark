"""Unit tests for train.tta: crop-box geometry and the max-over-crops logic,
using the same tiny fake ONNX builder inference tests use. No real data.
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest
from _fake_onnx import build_stage_onnx
from PIL import Image

pytest.importorskip("tensorflow")  # keep this file gated with the rest of test_train_*.py

from train.tta import predict_tta_max, three_crop_boxes, three_crops


def test_three_crop_boxes_dedup_for_a_narrow_frame():
    # width already <= size on both axes after the short-side resize -> 1 box
    boxes = three_crop_boxes(224, 224, size=224)
    assert len(boxes) == 1


def test_three_crop_boxes_gives_left_center_right_for_a_wide_frame():
    boxes = three_crop_boxes(455, 256, size=224)
    assert len(boxes) == 3
    xs = [b[0] for b in boxes]
    assert xs == sorted(xs)
    assert xs[0] == 0
    assert xs[-1] == 455 - 224


def test_three_crops_returns_correctly_shaped_arrays():
    img = Image.fromarray(np.full((256, 455, 3), 100, dtype=np.uint8))
    crops = three_crops(img)
    assert len(crops) == 3
    for c in crops:
        assert c.shape == (224, 224, 3)


def test_predict_tta_max_takes_the_max_over_crops(tmp_path):
    # a wide gradient image: only the right crop is bright -> max should pick it up
    arr = np.zeros((256, 455, 3), dtype=np.uint8)
    arr[:, 300:] = 255
    p = tmp_path / "wide.jpg"
    Image.fromarray(arr).save(p, quality=95)

    a_path = tmp_path / "stage_a.onnx"
    b_path = tmp_path / "stage_b.onnx"
    build_stage_onnx(a_path, prob_bias=0.0, prob_slope=20.0, cam_mode="constant")  # prob tracks mean brightness
    build_stage_onnx(b_path, prob_bias=0.0, prob_slope=20.0, cam_mode="constant")
    sess_a = ort.InferenceSession(str(a_path), providers=["CPUExecutionProvider"])
    sess_b = ort.InferenceSession(str(b_path), providers=["CPUExecutionProvider"])

    r = predict_tta_max(sess_a, sess_b, [str(p)])
    assert r["pA"].shape == (1,)
    assert r["pB"][0] > 0.5  # the bright right crop pushes the max up
