from __future__ import annotations

from pathlib import Path

import pytest

from inference.predict import predict
from inference.session import load_models

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "stage_a.onnx").exists(),
    reason="models/stage_a.onnx not present (gitignored; local-only test)",
)


def test_real_models_load_and_predict_on_a_real_frame():
    frames = sorted((Path(__file__).resolve().parents[1] / "data/ga511/frames").glob("*/*.jpg"))
    if not frames:
        pytest.skip("no 511GA frames on disk")
    models = load_models(MODELS_DIR)
    pred = predict(frames[0], models=models, heatmap=True)
    assert pred.status in ("dry", "wet", "flooded")
    assert 0.0 <= pred.confidence <= 1.0
    total = sum(pred.stage_probabilities.values())
    assert total == pytest.approx(1.0, abs=1e-4)
    assert pred.heatmap_png is not None
    assert pred.heatmap_png[:8] == b"\x89PNG\r\n\x1a\n"

