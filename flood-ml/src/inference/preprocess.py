# inference preprocessing, must match training exactly (crop/squash/letterbox)
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

RESIZE_SHORT_SIDE = 256
CROP_SIZE = 224
JPEG_QUALITY = 95
CROP_PRECROP_RATIO = 256 / 224
LETTERBOX_PAD_VALUE = (128, 128, 128)

MODE_CROP = "crop"
MODE_SQUASH = "squash"
MODE_LETTERBOX = "letterbox"
MODES = (MODE_CROP, MODE_SQUASH, MODE_LETTERBOX)


def to_pil(image) -> Image.Image:
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
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    canvas = Image.new("RGB", (size, size))
    canvas.paste(img, (-left, -top))
    return canvas, (left, top, left + size, top + size)


# numpy copy of tf.image.resize bilinear (half-pixel, no antialias) so inference = training
def _tf_bilinear(arr: np.ndarray, nh: int, nw: int) -> np.ndarray:
    h, w = arr.shape[:2]
    ys = np.clip((np.arange(nh) + 0.5) * (h / nh) - 0.5, 0, h - 1)
    xs = np.clip((np.arange(nw) + 0.5) * (w / nw) - 0.5, 0, w - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    wy, wx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
    top = arr[y0][:, x0] * (1 - wx) + arr[y0][:, x1] * wx
    bot = arr[y1][:, x0] * (1 - wx) + arr[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def _to_training_uint8(arr: np.ndarray) -> np.ndarray:
    return np.floor(np.clip(arr, 0.0, 255.0)).astype(np.float32)


def _squash_resize(img: Image.Image, size: int) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    return _to_training_uint8(_tf_bilinear(arr, size, size))


def _letterbox_resize(img: Image.Image, size: int, pad_value=LETTERBOX_PAD_VALUE):
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = _tf_bilinear(np.asarray(img, dtype=np.float32), nh, nw)
    canvas = np.empty((size, size, 3), dtype=np.float32)
    canvas[:] = np.asarray(pad_value, dtype=np.float32)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return _to_training_uint8(canvas), (left, top, left + nw, top + nh)


def preprocess(image, *, mode: str = MODE_CROP, size: int = CROP_SIZE, do_jpeg_roundtrip: bool = True):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    orig = to_pil(image)
    w0, h0 = orig.size

    if mode == MODE_CROP:
        precrop = max(size, round(size * CROP_PRECROP_RATIO))
        resized = resize_short_side(orig, precrop)
        if do_jpeg_roundtrip:
            resized = jpeg_roundtrip(resized)
        w1, h1 = resized.size
        cropped, box = _center_crop_pad(resized, size)
        arr = np.asarray(cropped, dtype=np.float32)
        scale = w1 / w0
        crop_box_orig = tuple(c / scale for c in box)
        geom = {
            "mode": mode, "size": size, "orig_size": (w0, h0),
            "resized_size": (w1, h1), "crop_box_resized": box,
            "crop_box_orig": crop_box_orig, "scale": scale,
        }
        return arr, geom

    # same as the saved training copies: short side 256 + jpeg q95
    src = resize_short_side(orig, RESIZE_SHORT_SIDE)
    if do_jpeg_roundtrip:
        src = jpeg_roundtrip(src)
    if mode == MODE_SQUASH:
        arr = _squash_resize(src, size)
        geom = {"mode": mode, "size": size, "orig_size": (w0, h0)}
        return arr, geom

    arr, content_box = _letterbox_resize(src, size)
    geom = {"mode": mode, "size": size, "orig_size": (w0, h0), "content_box_canvas": content_box}
    return arr, geom

