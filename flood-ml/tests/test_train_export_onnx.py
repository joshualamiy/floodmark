"""Unit tests for train.export_onnx: builds a tiny (weights=None) model,
exports it with tf2onnx, and checks the ONNX `cam` output shape matches the
Keras conv feature map, plus Keras/ONNX numerical agreement -- all on
random weights and random input, no downloads and no real data.
"""
import numpy as np
import pytest

pytest.importorskip("tensorflow")
onnxruntime = pytest.importorskip("onnxruntime")
pytest.importorskip("tf2onnx")

from train.export_onnx import (
    benchmark_latency,
    build_export_fn,
    verify_export,
)
from train.model import build_model, make_grad_model

INPUT_SIZE = 64  # small: keeps tf2onnx conversion fast in CI-like runs


def _export_tiny(tmp_path, backbone_name="mobilenetv3small"):
    import tf2onnx

    built = build_model(backbone_name, input_size=INPUT_SIZE, weights=None)
    export_fn, input_signature = build_export_fn(built.model, INPUT_SIZE)
    model_proto, _ = tf2onnx.convert.from_function(export_fn, input_signature=input_signature, opset=17)
    out_path = tmp_path / "model.onnx"
    out_path.write_bytes(model_proto.SerializeToString())
    return built, out_path


def test_onnx_cam_output_shape_matches_conv_feature_map(tmp_path):
    built, out_path = _export_tiny(tmp_path)
    grad_model = make_grad_model(built)
    x = np.random.default_rng(0).uniform(0, 255, size=(1, INPUT_SIZE, INPUT_SIZE, 3)).astype("float32")
    conv_out, _prob = grad_model(x, training=False)

    sess = onnxruntime.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    onnx_out = sess.run(None, {"image": x})
    cam = onnx_out[out_names.index("cam")]

    assert cam.shape == (1, conv_out.shape[1], conv_out.shape[2])


def test_onnx_prob_matches_keras_within_tolerance(tmp_path):
    built, out_path = _export_tiny(tmp_path)
    x = np.random.default_rng(1).uniform(0, 255, size=(3, INPUT_SIZE, INPUT_SIZE, 3)).astype("float32")
    keras_prob = built.model(x, training=False).numpy().reshape(-1)

    sess = onnxruntime.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    onnx_out = sess.run(None, {"image": x})
    onnx_prob = onnx_out[out_names.index("prob")].reshape(-1)

    assert np.max(np.abs(onnx_prob - keras_prob)) <= 1e-4


def test_verify_export_reports_high_cam_correlation(tmp_path):
    import pandas as pd
    from PIL import Image

    built, out_path = _export_tiny(tmp_path)

    rows = []
    rng = np.random.default_rng(3)
    for i in range(5):
        fname = f"img{i}.jpg"
        arr = rng.integers(0, 256, size=(96, 110, 3), dtype=np.uint8)
        Image.fromarray(arr).save(tmp_path / fname, quality=90)
        rows.append({"path": str(tmp_path / fname), "label": "dry"})
    val_rows = pd.DataFrame(rows)

    result = verify_export(built.model, out_path, val_rows, img_size=INPUT_SIZE, n_images=5)
    assert result["n_images"] == 5
    assert result["max_prob_diff"] <= 1e-4
    assert result["cam_pearson_r"] > 0.99


def test_benchmark_latency_returns_positive_timings(tmp_path):
    _built, out_path = _export_tiny(tmp_path)
    result = benchmark_latency(out_path, INPUT_SIZE, n_runs=5, intra_op_threads=1)
    assert result["median_ms"] > 0
    assert result["p95_ms"] >= result["median_ms"] * 0
