"""Optional water segmentation; independent of classifier CAM and road extent."""
from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

NOTE = (
    "Water-only segmentation; road=false. Not flooded-roadway extent or classifier CAM. "
    "River water may be highlighted; this model does not identify roads."
)
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "water"
_CACHE = {}


@dataclass
class WaterModel:
    session: object
    config: dict


def _rgb(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        image = io.BytesIO(image)
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    with Image.open(image) as im:
        return im.convert("RGB")


def letterbox_image(image, size=320):
    """Return RGB float32 [0,255], valid pixels, and reversible geometry."""
    original = _rgb(image)
    w, h = original.size
    if size < 1:
        raise ValueError("Input size must be positive")
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    left, top = (size - nw) // 2, (size - nh) // 2
    box = (left, top, left + nw, top + nh)
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(original.resize((nw, nh), Image.Resampling.BILINEAR), (left, top))
    valid = np.zeros((size, size), np.float32)
    valid[top:top + nh, left:left + nw] = 1
    return np.asarray(canvas, np.float32), valid, {"original_size": (w, h), "box": box}


def restore_probability(probability, geometry):
    """Remove padding before resizing probabilities back to the full frame."""
    probability = np.asarray(probability, np.float32)
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise ValueError("Expected a finite two-dimensional probability map")
    left, top, right, bottom = geometry["box"]
    content = probability[top:bottom, left:right]
    restored = Image.fromarray(content).resize(
        tuple(geometry["original_size"]), Image.Resampling.BILINEAR
    )
    return np.clip(np.asarray(restored), 0, 1)


def load_water_model(model_dir=None):
    """Cached optional loader. Missing artifacts return None; corrupt ones raise."""
    directory = Path(model_dir or os.environ.get("FLOODML_WATER_MODEL_DIR", DEFAULT_MODEL_DIR))
    config_path, onnx_path = directory / "config.json", directory / "water.onnx"
    if not config_path.is_file() or not onnx_path.is_file():
        return None
    key = (str(directory.resolve()), config_path.stat().st_mtime_ns,
           onnx_path.stat().st_mtime_ns, onnx_path.stat().st_size)
    if key in _CACHE:
        return _CACHE[key]
    config = json.loads(config_path.read_text())
    if (config.get("task") != "water_segmentation"
            or not 0 < float(config["threshold"]) < 1
            or int(config["input_size"]) < 32):
        raise ValueError("Invalid water model configuration")
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path), options, providers=["CPUExecutionProvider"]
    )
    loaded = WaterModel(session, config)
    # Remove superseded versions of this directory without caching missing files.
    for old in list(_CACHE):
        if old[0] == key[0]:
            del _CACHE[old]
    _CACHE[key] = loaded
    return loaded


def _png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def predict_water(image, model=None):
    """Return original-size binary mask and transparent cyan overlay PNG bytes.

    water_fraction is the fraction of ALL original-frame pixels predicted water.
    A missing model raises FileNotFoundError, never a fabricated all-zero mask.
    """
    model = model if model is not None else load_water_model()
    if model is None:
        raise FileNotFoundError("Optional water model is absent from models/water")
    size = int(model.config["input_size"])
    pixels, _, geometry = letterbox_image(image, size)
    output = model.session.run(["water_prob"], {"image": pixels[None]})[0]
    if output.shape != (1, size, size, 1):
        raise ValueError(f"Unexpected water output shape: {output.shape}")
    probability = restore_probability(output[0, ..., 0], geometry)
    threshold = float(model.config["threshold"])
    mask = probability >= threshold
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay[mask] = (0, 190, 255, 112)
    return {
        "water_overlay_png": _png(Image.fromarray(overlay)),
        "water_mask_png": _png(Image.fromarray(mask.astype(np.uint8) * 255)),
        "water_fraction": float(mask.mean()),
        "threshold": threshold,
        "model_version": str(model.config["model_version"]),
        "note": NOTE,
    }
