"""Loads config.json + creates cached onnxruntime CPU sessions for both stages."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import NamedTuple

import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "models"


class Models(NamedTuple):
    config: dict
    sess_a: ort.InferenceSession
    sess_b: ort.InferenceSession
    model_dir: Path


_CACHE: dict[str, Models] = {}


def _resolve_model_dir(model_dir=None) -> Path:
    if model_dir is not None:
        return Path(model_dir)
    env = os.environ.get("FLOODML_MODEL_DIR")
    return Path(env) if env else DEFAULT_MODEL_DIR


def load_models(model_dir: str | Path | None = None) -> Models:
    """Cached by resolved model_dir so repeated calls are free."""
    d = _resolve_model_dir(model_dir)
    key = str(d.resolve())
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    cfg = json.loads((d / "config.json").read_text())
    so = ort.SessionOptions()
    providers = ["CPUExecutionProvider"]
    sess_a = ort.InferenceSession(str(d / cfg["stage_a"]["onnx_path"]), so, providers=providers)
    sess_b = ort.InferenceSession(str(d / cfg["stage_b"]["onnx_path"]), so, providers=providers)
    m = Models(config=cfg, sess_a=sess_a, sess_b=sess_b, model_dir=d)
    _CACHE[key] = m
    return m


def clear_cache() -> None:
    """Test helper: drop cached sessions."""
    _CACHE.clear()
