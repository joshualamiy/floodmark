from __future__ import annotations

import hashlib

from .contracts import Prediction


def model_run(image_bytes: bytes) -> Prediction:
    """Temporary deterministic stand-in for the production ONNX contract."""
    seed = int.from_bytes(hashlib.sha256(image_bytes).digest()[:8], "big") / 2**64
    wet_probability = round(seed, 4)
    flooded_probability = round((seed * 1.61803398875) % 1, 4)
    status = "dry" if wet_probability < 0.5 else "flooded" if flooded_probability >= 0.7 else "wet"
    probabilities = {
        "dry": round(1 - wet_probability, 4),
        "wet": round(wet_probability * (1 - flooded_probability), 4),
        "flooded": round(wet_probability * flooded_probability, 4),
    }
    return Prediction(
        model_version={"data_version": "mvp-1", "stage_a_run_id": "stub-a", "stage_b_run_id": "stub-b"},
        status=status,
        confidence=probabilities[status],
        stage_a_probabilities={"dry": round(1 - wet_probability, 4), "wet": wet_probability},
        stage_b_probabilities={"not_flooded": round(1 - flooded_probability, 4), "flooded": flooded_probability},
        stage_probabilities=probabilities,
        thresholds={"tA": 0.5, "tB": 0.7},
        note="Water detected below the flood alert threshold." if status == "wet" else None,
        heatmap_bytes=None,
    )
