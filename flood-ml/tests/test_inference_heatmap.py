"""Heatmap PNG: decodes, right size, aspect preserved, blank-safe."""
from __future__ import annotations

import io
import json

import pytest
from _fake_onnx import build_stage_onnx, gradient_image, logit, uniform_image
from PIL import Image

from inference.predict import predict
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
