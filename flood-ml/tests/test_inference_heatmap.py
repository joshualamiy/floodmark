"""Heatmap PNG: decodes, right size, aspect preserved, blank-safe."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
from _fake_onnx import build_stage_onnx, gradient_image, logit, uniform_image
from PIL import Image

from inference.predict import predict
from inference.preprocess import preprocess
from inference.session import clear_cache, load_models


@pytest.fixture
def models(tmp_path):
    build_stage_onnx(tmp_path / "stage_a.onnx", prob_bias=logit(0.9), cam_mode="constant")
    build_stage_onnx(tmp_path / "stage_b.onnx", prob_bias=logit(0.5), cam_mode="content", cam_scale=1.0)
    cfg = {
        "data_version": "vtest",
        "stage_a": {"run_id": "run_a", "onnx_path": "stage_a.onnx", "threshold_tA": 0.5},
        "stage_b": {"run_id": "run_b", "onnx_path": "stage_b.onnx", "threshold_tB": 0.5},
        "preprocess": {"jpeg_roundtrip": False},
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    yield load_models(tmp_path)
    clear_cache()


def test_heatmap_decodes_and_matches_frame_size(models):
    img = Image.fromarray(gradient_image((400, 300)))
    pred = predict(img, models=models, heatmap=True)
    assert pred.heatmap_png is not None
    out = Image.open(io.BytesIO(pred.heatmap_png))
    assert out.format == "PNG"
    assert out.size == (400, 300)


def test_heatmap_caps_long_side_at_640_and_keeps_aspect(models):
    img = Image.fromarray(gradient_image((1600, 800)))
    pred = predict(img, models=models, heatmap=True)
    out = Image.open(io.BytesIO(pred.heatmap_png))
    assert max(out.size) == 640
    assert out.size[0] / out.size[1] == pytest.approx(1600 / 800, rel=1e-2)


def test_blank_cam_returns_valid_png_with_no_error(tmp_path):
    # cam_mode="constant" with bias 0 -> ReLU(0) everywhere -> "blank heat" branch
    build_stage_onnx(tmp_path / "stage_a.onnx", prob_bias=logit(0.9), cam_mode="constant", cam_bias=0.0)
    build_stage_onnx(tmp_path / "stage_b.onnx", prob_bias=logit(0.9), cam_mode="constant", cam_bias=0.0)
    cfg = {
        "data_version": "vtest",
        "stage_a": {"run_id": "a", "onnx_path": "stage_a.onnx", "threshold_tA": 0.5},
        "stage_b": {"run_id": "b", "onnx_path": "stage_b.onnx", "threshold_tB": 0.5},
        "preprocess": {"jpeg_roundtrip": False},
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    m = load_models(tmp_path)
    pred = predict(Image.fromarray(uniform_image(128)), models=m, heatmap=True)
    out = Image.open(io.BytesIO(pred.heatmap_png))
    assert out.format == "PNG"
    clear_cache()


def test_heatmap_none_when_disabled(models):
    pred = predict(Image.fromarray(gradient_image()), models=models, heatmap=False)
    assert pred.heatmap_png is None


def test_heatmap_squash_covers_the_whole_frame():
    from inference.heatmap import make_heatmap_png

    frame = Image.fromarray(gradient_image((400, 300)))
    cam = np.ones((7, 7), dtype=np.float32)
    _arr, geom = preprocess(frame, mode="squash", size=64, do_jpeg_roundtrip=False)
    png = make_heatmap_png(frame, cam, geom)
    out = Image.open(io.BytesIO(png))
    assert out.size == (400, 300)


def test_heatmap_letterbox_maps_content_box_back():
    from inference.heatmap import make_heatmap_png

    frame = Image.fromarray(gradient_image((500, 200)))  # wide -> padded top/bottom
    cam = np.ones((7, 7), dtype=np.float32)
    _arr, geom = preprocess(frame, mode="letterbox", size=64, do_jpeg_roundtrip=False)
    png = make_heatmap_png(frame, cam, geom)
    out = Image.open(io.BytesIO(png))
    assert out.size == (500, 200)


@pytest.mark.parametrize("pa,pb", [(0.1, 0.99), (0.99, 0.1), (0.9, 0.5)])
def test_weak_evidence_stays_plain_but_debug_cam_is_available(tmp_path, pa, pb):
    build_stage_onnx(tmp_path / "stage_a.onnx", prob_bias=logit(pa))
    build_stage_onnx(tmp_path / "stage_b.onnx", prob_bias=logit(pb), cam_mode="content")
    cfg = {"stage_a": {"run_id": "test_a", "onnx_path": "stage_a.onnx", "threshold_tA": 0.5},
           "stage_b": {"run_id": "test_b", "onnx_path": "stage_b.onnx", "threshold_tB": 0.8},
           "preprocess": {"jpeg_roundtrip": False}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    img = Image.fromarray(gradient_image((450, 253)))
    pred = predict(img, models=load_models(tmp_path), raw_heatmap=True)
    assert pred.heatmap_status == "no_strong_evidence"
    assert np.array_equal(np.array(img), np.array(Image.open(io.BytesIO(pred.heatmap_png))))
    assert not np.array_equal(np.array(img), np.array(Image.open(io.BytesIO(pred.raw_heatmap_png))))
    assert pred.heatmap_score == pytest.approx(pa * pb, abs=1e-6)
    assert "does not rule out" in pred.heatmap_note
    clear_cache()


def test_display_strength_changes_overlay_not_predictions(models):
    from inference.heatmap import make_heatmap_png

    img = Image.new("RGB", (224, 224), (70, 70, 70))
    _, geom = preprocess(img, do_jpeg_roundtrip=False)
    cam = np.tile(np.linspace(0, 1, 7, dtype=np.float32), (7, 1))
    faint = np.asarray(Image.open(io.BytesIO(make_heatmap_png(img, cam, geom, strength=0.2))))
    strong = np.asarray(Image.open(io.BytesIO(make_heatmap_png(img, cam, geom, strength=0.9))))
    assert (faint.astype(float) - 70).std() < (strong.astype(float) - 70).std()
    a = predict(img, models=models, heatmap=False)
    b = predict(img, models=models, raw_heatmap=True)
    assert (a.status, a.confidence, a.stage_probabilities) == (b.status, b.confidence, b.stage_probabilities)


def test_no_spatial_evidence_is_not_presented_as_localized():
    from inference.heatmap import heatmap_display

    assert heatmap_display(0.99, 0.99, 0.5, np.ones((7, 7)))["state"] == "unlocalized"
    assert heatmap_display(0.99, 0.99, 0.5, np.full((7, 7), np.nan))["state"] == "unlocalized"
    assert heatmap_display(0.99, 0.99, 0.5, np.arange(49).reshape(7, 7))["state"] == "shown"


def test_heatmap_outside_center_crop_is_unchanged():
    from inference.heatmap import make_heatmap_png

    frame = Image.new("RGB", (450, 253), (90, 90, 90))
    _, geom = preprocess(frame, do_jpeg_roundtrip=False)
    cam = np.arange(49, dtype=np.float32).reshape(7, 7)
    out = np.asarray(Image.open(io.BytesIO(make_heatmap_png(frame, cam, geom))))
    assert np.all(out[:, :100] == 90)
    assert np.all(out[:, -100:] == 90)
    assert np.any(out[:, 150:300] != 90)


def test_letterbox_discards_attribution_in_padding():
    from inference.heatmap import make_heatmap_png

    frame = Image.new("RGB", (500, 200), (80, 80, 80))
    _, geom = preprocess(frame, mode="letterbox", size=64, do_jpeg_roundtrip=False)
    cam = np.zeros((64, 64), dtype=np.float32)
    cam[:8] = 1.0
    out = np.asarray(Image.open(io.BytesIO(make_heatmap_png(frame, cam, geom))))
    assert np.all(out == 80)
