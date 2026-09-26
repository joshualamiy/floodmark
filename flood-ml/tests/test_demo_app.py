import io
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

pytest.importorskip("gradio")
import demo_app


def test_empty_upload_clears_every_panel():
    assert demo_app.run(None) == ("-", "-", {}, None, "", None, None, "")


def test_debug_request_and_water_panel_are_separate(monkeypatch):
    b = io.BytesIO()
    Image.new("RGB", (20, 20)).save(b, format="PNG")
    calls = []

    def predict(image, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status="dry", note=None, confidence=0.99,
            stage_probabilities={"dry": 0.99, "wet": 0.005, "flooded": 0.005},
            heatmap_png=b.getvalue(), raw_heatmap_png=b.getvalue(),
            heatmap_note="No strong flood evidence to highlight.")

    monkeypatch.setattr(demo_app, "predict", predict)
    monkeypatch.setitem(sys.modules, "inference.water", SimpleNamespace(
        load_water_model=lambda: object(),
        predict_water=lambda image, model: {"water_overlay_png": b.getvalue(),
                                           "note": "Predicted water, not road flooding."}))
    result = demo_app.run(Image.new("RGB", (20, 20)), raw_debug=True, show_water=True)
    assert calls == [{"heatmap": True, "raw_heatmap": True}]
    assert result[0] == "dry"
    assert result[5].size == (20, 20)
    assert result[6].size == (20, 20)
    assert "not road flooding" in result[7]


def test_app_builds():
    assert demo_app.build_app() is not None

