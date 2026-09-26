from __future__ import annotations

import numpy as np
import pytest
from _fake_onnx import uniform_image
from PIL import Image

from inference.preprocess import (
    CROP_SIZE,
    MODE_LETTERBOX,
    MODE_SQUASH,
    preprocess,
    resize_short_side,
    to_pil,
)


def test_resize_short_side_no_op_when_small():
    img = Image.fromarray(uniform_image(100, size=(200, 240)))
    out = resize_short_side(img, short_side=256)
    assert out.size == (200, 240)


def test_resize_short_side_resizes_when_large():
    img = Image.fromarray(uniform_image(100, size=(1024, 768)))
    out = resize_short_side(img, short_side=256)
    assert min(out.size) == 256
    assert out.size[0] / out.size[1] == pytest.approx(1024 / 768, rel=1e-2)


def test_preprocess_output_shape():
    img = Image.fromarray(uniform_image(100, size=(500, 300)))
    arr, geom = preprocess(img, do_jpeg_roundtrip=False)
    assert arr.shape == (CROP_SIZE, CROP_SIZE, 3)
    assert arr.dtype == np.float32
    assert geom["orig_size"] == (500, 300)


def test_preprocess_pads_small_images():
    img = Image.fromarray(uniform_image(50, size=(100, 100)))
    arr, geom = preprocess(img, do_jpeg_roundtrip=False)
    assert arr.shape == (CROP_SIZE, CROP_SIZE, 3)
    left, top, _right, _bottom = geom["crop_box_resized"]
    assert left < 0 and top < 0


def test_preprocess_crop_box_orig_maps_back_correctly():
    img = Image.fromarray(uniform_image(50, size=(1024, 512)))
    _arr, geom = preprocess(img, do_jpeg_roundtrip=False)
    scale = geom["scale"]
    left, _top, right, _bottom = geom["crop_box_orig"]
    assert 0 <= left < right <= 1024 + 1
    assert (right - left) * scale == pytest.approx(CROP_SIZE, rel=1e-2)


def test_jpeg_roundtrip_changes_pixels_slightly_but_keeps_size():
    img = Image.fromarray(uniform_image(37, size=(300, 300)))
    arr_plain, _ = preprocess(img, do_jpeg_roundtrip=False)
    arr_rt, _ = preprocess(img, do_jpeg_roundtrip=True)
    assert arr_plain.shape == arr_rt.shape
    assert np.abs(arr_plain - arr_rt).max() < 5


def test_squash_ignores_aspect_and_fills_the_whole_canvas():
    img = Image.fromarray(uniform_image(100, size=(500, 200)))
    arr, geom = preprocess(img, mode=MODE_SQUASH, size=64, do_jpeg_roundtrip=False)
    assert arr.shape == (64, 64, 3)
    assert geom["mode"] == "squash"
    assert geom["orig_size"] == (500, 200)
    assert "crop_box_orig" not in geom


def test_letterbox_pads_a_wide_frame_top_and_bottom():
    img = Image.fromarray(uniform_image(200, size=(500, 200)))
    arr, geom = preprocess(img, mode=MODE_LETTERBOX, size=64, do_jpeg_roundtrip=False)
    assert arr.shape == (64, 64, 3)
    left, top, right, bottom = geom["content_box_canvas"]
    assert right - left == 64
    assert bottom - top < 64
    assert top > 0


def test_letterbox_pad_region_is_mid_gray():
    img = Image.fromarray(uniform_image(0, size=(500, 100)))
    arr, _geom = preprocess(img, mode=MODE_LETTERBOX, size=64, do_jpeg_roundtrip=False)
    assert arr[0, 0, 0] == pytest.approx(128.0, abs=1.0)


def test_unknown_mode_rejected():
    img = Image.fromarray(uniform_image(100, size=(100, 100)))
    with pytest.raises(ValueError):
        preprocess(img, mode="fisheye")


def test_to_pil_accepts_all_types(tmp_path):
    rgb = uniform_image(120, size=(64, 64))
    assert to_pil(Image.fromarray(rgb)).mode == "RGB"
    assert to_pil(rgb).mode == "RGB"
    rgba = np.dstack([rgb, np.full((64, 64), 255, dtype=np.uint8)])
    assert to_pil(rgba).mode == "RGB"
    gray = np.full((64, 64), 90, dtype=np.uint8)
    assert to_pil(gray).mode == "RGB"

    p = tmp_path / "x.jpg"
    Image.fromarray(rgb).save(p)
    assert to_pil(str(p)).mode == "RGB"
    assert to_pil(p).mode == "RGB"

    import io
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    assert to_pil(buf.getvalue()).mode == "RGB"

