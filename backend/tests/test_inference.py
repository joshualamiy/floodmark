import io
import os
import sys
from pathlib import Path

import pytest
from PIL import Image


MODEL_DIR = os.environ.get("FLOODML_MODEL_DIR")
pytestmark = pytest.mark.skipif(
    not MODEL_DIR,
    reason="FLOODML_MODEL_DIR is not set; real model files are local-only",
)


if MODEL_DIR:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "flood-ml" / "src"))


def jpeg_bytes() -> bytes:
    image = Image.new("RGB", (320, 240), (128, 128, 128))
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def test_real_model_prediction_is_stable_for_retries():
    from floodmark_pipeline.inference import model_run

    first = model_run(jpeg_bytes())
    second = model_run(jpeg_bytes())

    assert first == second
    assert first.status in {"dry", "wet", "flooded"}
    assert first.model_version["stage_a_run_id"] != "stub-a"
    assert first.model_version["stage_b_run_id"] != "stub-b"
    assert sum(first.stage_probabilities.values()) == pytest.approx(1.0)
    assert first.heatmap_bytes is not None
    assert first.heatmap_bytes.startswith(b"\x89PNG")
