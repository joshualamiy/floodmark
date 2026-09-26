"""Matches the training-time path: resize short side 256 (LANCZOS, no-op if
already <= 256 -- mirrors src/prep/common.py::resize_short_side) -> optional
JPEG q95 round-trip -> center crop/pad to 224. See docs/INFERENCE_API.md.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

RESIZE_SHORT_SIDE = 256
CROP_SIZE = 224
JPEG_QUALITY = 95


def to_pil(image) -> Image.Image:
    """PIL Image, RGB/RGBA/gray ndarray, raw bytes, or path -> RGB PIL image."""
    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image))
    elif isinstance(image, (str, Path)):
        img = Image.open(image)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            img = Image.fromarray(image, mode="L")
        elif image.shape[-1] == 4:
            img = Image.fromarray(image, mode="RGBA")
        else:
            img = Image.fromarray(image, mode="RGB")
    else:
        raise TypeError(f"unsupported image type: {type(image)!r}")
    img.load()
    return img.convert("RGB")


def resize_short_side(img: Image.Image, short_side: int = RESIZE_SHORT_SIDE) -> Image.Image:
    w, h = img.size
    if min(w, h) <= short_side:
        return img
    if w < h:
        nw, nh = short_side, round(h * short_side / w)
    else:
        nh, nw = short_side, round(w * short_side / h)
    return img.resize((nw, nh), Image.LANCZOS)


def jpeg_roundtrip(img: Image.Image, quality: int = JPEG_QUALITY) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out.convert("RGB")


def _center_crop_pad(img: Image.Image, size: int = CROP_SIZE):
    """Crop or zero-pad to size x size, centered (matches tf.image.resize_with_crop_or_pad).
    Works for both cases via one paste: negative offsets pad, positive ones crop.
    """
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    canvas = Image.new("RGB", (size, size))
    canvas.paste(img, (-left, -top))
    return canvas, (left, top, left + size, top + size)


def preprocess(image, *, do_jpeg_roundtrip: bool = True) -> tuple[np.ndarray, dict]:
    """raw input -> ((224,224,3) float32 in [0,255], geometry for heatmap mapping)."""
    orig = to_pil(image)
    w0, h0 = orig.size
    resized = resize_short_side(orig)
    if do_jpeg_roundtrip:
        resized = jpeg_roundtrip(resized)
    w1, h1 = resized.size
    cropped, box = _center_crop_pad(resized)
    arr = np.asarray(cropped, dtype=np.float32)

    scale = w1 / w0  # resize_short_side is uniform, == h1 / h0
    crop_box_orig = tuple(c / scale for c in box)
    geom = {
        "orig_size": (w0, h0),
        "resized_size": (w1, h1),
        "crop_box_resized": box,
        "crop_box_orig": crop_box_orig,
        "scale": scale,
    }
    return arr, geom
