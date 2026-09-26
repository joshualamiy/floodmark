from __future__ import annotations

import json

import numpy as np
import pytest
from _fake_onnx import build_stage_onnx, logit, uniform_image
from PIL import Image

from inference.predict import predict, predict_batch
from inference.session import clear_cache, load_models

TA = 0.5
TB = 0.5


@pytest.fixture
def model_dir(tmp_path):
    build_stage_onnx(tmp_path / "stage_a.onnx", prob_bias=0.0, cam_mode="constant")
    build_stage_onnx(tmp_path / "stage_b.onnx", prob_bias=0.0, cam_mode="constant")
    cfg = {
        "data_version": "vtest",
        "stage_a": {"run_id": "run_a", "onnx_path": "stage_a.onnx", "threshold_tA": TA},
        "stage_b": {"run_id": "run_b", "onnx_path": "stage_b.onnx", "threshold_tB": TB},
        "preprocess": {"jpeg_roundtrip": False},
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    yield tmp_path
    clear_cache()


def _set_probs(model_dir, pa, pb):
    build_stage_onnx(model_dir / "stage_a.onnx", prob_bias=logit(pa), cam_mode="constant")
    build_stage_onnx(model_dir / "stage_b.onnx", prob_bias=logit(pb), cam_mode="constant")
    clear_cache()
    return load_models(model_dir)


@pytest.mark.parametrize(
    "pa, pb, expected",
    [
        (TA - 0.1, 0.0, "dry"),
        (TA + 0.1, TB + 0.1, "flooded"),
        (TA + 0.1, TB - 0.1, "wet"),
        (TA, 0.0, "wet"),
    ],
)
def test_status_boundaries(model_dir, pa, pb, expected):
    models = _set_probs(model_dir, pa, pb)
    img = Image.fromarray(uniform_image(128))
    pred = predict(img, models=models, heatmap=False)
    assert pred.status == expected


def test_pb_exactly_at_threshold_is_flooded(model_dir):
    models = _set_probs(model_dir, TA + 0.1, TB)
    pred = predict(Image.fromarray(uniform_image(128)), models=models, heatmap=False)
    assert pred.status == "flooded"


@pytest.mark.parametrize("pa,pb", [(0.1, 0.2), (0.9, 0.1), (0.9, 0.99), (0.6, 0.6)])
def test_stage_probabilities_sum_to_one_and_match_confidence(model_dir, pa, pb):
    models = _set_probs(model_dir, pa, pb)
    pred = predict(Image.fromarray(uniform_image(100)), models=models, heatmap=False)
    total = sum(pred.stage_probabilities.values())
    assert total == pytest.approx(1.0, abs=1e-6)
    assert pred.confidence == pytest.approx(pred.stage_probabilities[pred.status], abs=1e-6)
    assert pred.stage_a_probs["wet"] == pytest.approx(pa, abs=1e-6)
    assert pred.stage_b_probs["flooded"] == pytest.approx(pb, abs=1e-6)


def test_wet_status_carries_note(model_dir):
    models = _set_probs(model_dir, TA + 0.1, TB - 0.2)
    pred = predict(Image.fromarray(uniform_image(128)), models=models, heatmap=False)
    assert pred.status == "wet"
    assert pred.note == "water detected below flood alert threshold"


def test_dry_and_flooded_have_no_note(model_dir):
    models = _set_probs(model_dir, TA - 0.1, 0.0)
    assert predict(Image.fromarray(uniform_image(128)), models=models, heatmap=False).note is None
    models = _set_probs(model_dir, TA + 0.1, TB + 0.1)
    assert predict(Image.fromarray(uniform_image(128)), models=models, heatmap=False).note is None


def test_thresholds_and_model_version_reported(model_dir):
    models = _set_probs(model_dir, 0.1, 0.1)
    pred = predict(Image.fromarray(uniform_image(128)), models=models, heatmap=False)
    assert pred.thresholds == {"tA": TA, "tB": TB}
    assert pred.model_version["stage_a_run_id"] == "run_a"
    assert pred.model_version["stage_b_run_id"] == "run_b"
    assert pred.model_version["data_version"] == "vtest"


def test_predict_batch_matches_single_predict(model_dir):
    models = _set_probs(model_dir, 0.7, 0.3)
    images = [Image.fromarray(uniform_image(v)) for v in (10, 100, 200)]
    batch_out = predict_batch(images, models=models, heatmap=False)
    single_out = [predict(im, models=models, heatmap=False) for im in images]
    for b, s in zip(batch_out, single_out):
        assert b.status == s.status
        assert b.confidence == pytest.approx(s.confidence, abs=1e-6)
        assert b.stage_probabilities == pytest.approx(s.stage_probabilities, abs=1e-6)


def test_input_type_pil(model_dir):
    models = _set_probs(model_dir, 0.6, 0.6)
    pred = predict(Image.fromarray(uniform_image(128)), models=models, heatmap=False)
    assert pred.status in ("dry", "wet", "flooded")


def test_input_type_ndarray_rgb(model_dir):
    models = _set_probs(model_dir, 0.6, 0.6)
    arr = uniform_image(128)
    pred = predict(arr, models=models, heatmap=False)
    assert pred.status in ("dry", "wet", "flooded")


def test_input_type_ndarray_rgba(model_dir):
    models = _set_probs(model_dir, 0.6, 0.6)
    rgb = uniform_image(128)
    rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
    pred = predict(rgba, models=models, heatmap=False)
    assert pred.status in ("dry", "wet", "flooded")


def test_input_type_ndarray_grayscale(model_dir):
    models = _set_probs(model_dir, 0.6, 0.6)
    gray = np.full((240, 240), 128, dtype=np.uint8)
    pred = predict(gray, models=models, heatmap=False)
    assert pred.status in ("dry", "wet", "flooded")


def test_input_type_bytes(model_dir):
    models = _set_probs(model_dir, 0.6, 0.6)
    import io
    buf = io.BytesIO()
    Image.fromarray(uniform_image(128)).save(buf, format="PNG")
    pred = predict(buf.getvalue(), models=models, heatmap=False)
    assert pred.status in ("dry", "wet", "flooded")


def test_input_type_path(model_dir, tmp_path):
    models = _set_probs(model_dir, 0.6, 0.6)
    p = tmp_path / "frame.jpg"
    Image.fromarray(uniform_image(128)).save(p, format="JPEG")
    pred = predict(str(p), models=models, heatmap=False)
    assert pred.status in ("dry", "wet", "flooded")
    pred2 = predict(p, models=models, heatmap=False)
    assert pred2.status == pred.status


def test_to_dict_excludes_heatmap_by_default(model_dir):
    models = _set_probs(model_dir, 0.9, 0.9)
    pred = predict(Image.fromarray(uniform_image(128)), models=models, heatmap=True)
    d = pred.to_dict()
    assert d["heatmap_png"] is None
    d_incl = pred.to_dict(include_heatmap=True)
    assert isinstance(d_incl["heatmap_png"], str) and len(d_incl["heatmap_png"]) > 0
    json.dumps(d)
    json.dumps(d_incl)

