import io
import json

import numpy as np
import pytest
from PIL import Image

from inference.water import WaterModel, letterbox_image, load_water_model, predict_water


class FakeSession:
    def run(self, outputs, inputs):
        assert outputs == ["water_prob"]
        pixels = inputs["image"]
        return [(pixels[..., :1] > 200).astype(np.float32)]


def fake_model():
    return WaterModel(FakeSession(), {"input_size": 32, "threshold": 0.5,
                                      "model_version": "fake"})


@pytest.mark.parametrize("shape", [(19, 83), (83, 19), (1, 100), (31, 31)])
def test_original_geometry_fraction_and_transparent_overlay(shape):
    image = np.zeros((*shape, 3), np.uint8)
    image[:, :shape[1] // 2, 0] = 255
    result = predict_water(image, model=fake_model())
    assert isinstance(result, dict)
    mask = np.array(Image.open(io.BytesIO(result["water_mask_png"])))
    overlay = np.array(Image.open(io.BytesIO(result["water_overlay_png"])))
    assert mask.shape == shape
    assert overlay.shape == (*shape, 4)
    assert set(np.unique(mask)) <= {0, 255}
    assert result["water_fraction"] == pytest.approx((mask > 0).mean())
    assert abs(result["water_fraction"] - 0.5) < 0.10
    assert np.all(overlay[mask == 0, 3] == 0)
    assert np.all(overlay[mask != 0, 3] == 112)
    assert "road=false" in result["note"]


def test_padding_predictions_do_not_become_water():
    class PadSession:
        def run(self, _, inputs):
            gray = np.all(inputs["image"] == 128, axis=-1, keepdims=True)
            return [gray.astype(np.float32)]
    model = WaterModel(PadSession(), fake_model().config)
    result = predict_water(Image.new("RGB", (100, 10)), model)
    assert result["water_fraction"] == 0


def test_optional_missing_model_and_no_fake_negative(tmp_path, monkeypatch):
    from inference import water
    assert load_water_model(tmp_path) is None
    monkeypatch.setattr(water, "load_water_model", lambda: None)
    with pytest.raises(FileNotFoundError):
        predict_water(Image.new("RGB", (10, 10)))


def test_model_loader_caches_real_artifacts_and_not_absence(tmp_path, monkeypatch):
    import onnxruntime

    from inference import water
    water._CACHE.clear()
    assert load_water_model(tmp_path) is None
    (tmp_path / "water.onnx").write_bytes(b"fake")
    (tmp_path / "config.json").write_text(json.dumps({
        "task": "water_segmentation", "threshold": 0.5, "input_size": 32,
        "model_version": "fake"}))
    calls = []
    def make_session(*args, **kwargs):
        calls.append(args)
        return FakeSession()
    monkeypatch.setattr(onnxruntime, "InferenceSession", make_session)
    assert load_water_model(tmp_path) is load_water_model(tmp_path)
    assert len(calls) == 1


def test_letterbox_valid_area():
    pixels, valid, geometry = letterbox_image(Image.new("RGB", (100, 25)), 32)
    assert valid.sum() == 32 * 8
    assert geometry["box"] == (0, 12, 32, 20)
    assert np.all(pixels[valid == 0] == 128)


def test_nonfinite_model_output_is_rejected():
    class BadSession:
        def run(self, _, inputs):
            return [np.full((*inputs["image"].shape[:3], 1), np.nan, np.float32)]
    with pytest.raises(ValueError, match="finite"):
        predict_water(Image.new("RGB", (40, 20)), WaterModel(BadSession(), fake_model().config))


def test_import_has_no_training_dependencies():
    import os
    import subprocess
    import sys
    from pathlib import Path
    code = (
        "import sys; import inference.water; "
        "assert not any(m in sys.modules for m in "
        "('tensorflow', 'keras', 'torch', 'tf2onnx', 'pandas'))"
    )
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    subprocess.run([sys.executable, "-B", "-c", code], env=environment, check=True)

