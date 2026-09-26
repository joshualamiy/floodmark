"""7x7 CAM -> upsampled, normalized, colorized, alpha-blended PNG overlay.
No cv2/matplotlib: jet colormap and bilinear resize are plain numpy/PIL.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

CROP_SIZE = 224
MAX_LONG_SIDE = 640
ALPHA = 0.45
BLANK_EPS = 1e-6


def _resize_float(arr: np.ndarray, size) -> np.ndarray:
    im = Image.fromarray(arr.astype(np.float32), mode="F")
    im = im.resize(size, Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def _normalize(cam: np.ndarray) -> np.ndarray:
    lo, hi = float(cam.min()), float(cam.max())
    if hi < BLANK_EPS or (hi - lo) < BLANK_EPS:
        return np.zeros_like(cam)  # ~all-zero CAM -> blank heat
    return (cam - lo) / (hi - lo)


def _jet(x: np.ndarray) -> np.ndarray:
    """Cheap analytic jet colormap, x in [0,1] -> (...,3) in [0,1]."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def make_heatmap_png(
    frame: Image.Image, cam: np.ndarray, geom: dict,
    alpha: float = ALPHA, max_long_side: int = MAX_LONG_SIDE,
) -> bytes:
    w0, h0 = geom["orig_size"]
    up = _resize_float(np.asarray(cam), (CROP_SIZE, CROP_SIZE))
    norm = _normalize(up)

    left, top, right, bottom = geom["crop_box_orig"]
    box_w = max(1, round(right - left))
    box_h = max(1, round(bottom - top))
    heat_box = _resize_float(norm, (box_w, box_h))

    heat_full = np.zeros((h0, w0), dtype=np.float32)
    li, ti = round(left), round(top)
    dst_x0, dst_y0 = max(0, li), max(0, ti)
    dst_x1, dst_y1 = min(w0, li + box_w), min(h0, ti + box_h)
    src_x0, src_y0 = dst_x0 - li, dst_y0 - ti
    src_x1, src_y1 = src_x0 + (dst_x1 - dst_x0), src_y0 + (dst_y1 - dst_y0)
    if dst_x1 > dst_x0 and dst_y1 > dst_y0:
        heat_full[dst_y0:dst_y1, dst_x0:dst_x1] = heat_box[src_y0:src_y1, src_x0:src_x1]

    base = np.asarray(frame.convert("RGB"), dtype=np.float32)
    color = _jet(heat_full) * 255.0
    a = (heat_full * alpha)[..., None]
    blended = np.clip(base * (1 - a) + color * a, 0, 255).astype(np.uint8)
    out = Image.fromarray(blended, mode="RGB")

    if max(out.size) > max_long_side:
        s = max_long_side / max(out.size)
        new_size = (max(1, round(out.width * s)), max(1, round(out.height * s)))
        out = out.resize(new_size, Image.BILINEAR)

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()
