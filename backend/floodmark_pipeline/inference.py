from __future__ import annotations

from inference import load_models, predict

from .contracts import Prediction


_MODELS = load_models()


def model_run(image_bytes: bytes) -> Prediction:
    """Run the deployed flood classifier on raw or normalized image bytes."""
    result = predict(image_bytes, _MODELS)
    return Prediction(
        model_version=result.model_version,
        status=result.status,
        confidence=result.confidence,
        stage_a_probabilities=result.stage_a_probs,
        stage_b_probabilities=result.stage_b_probs,
        stage_probabilities=result.stage_probabilities,
        thresholds=result.thresholds,
        note=result.note,
        heatmap_bytes=result.heatmap_png,
        heatmap_status=result.heatmap_status,
        heatmap_note=result.heatmap_note,
    )
