"""Render relative flood attribution; colors are not water probabilities."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

CROP_SIZE = 224
MAX_LONG_SIDE = 640
ALPHA = 0.45
BLANK_EPS = 1e-6
MIN_DISPLAY_SCORE = 0.5


def _resize_float(arr: np.ndarray, size) -> np.ndarray:
    im = Image.fromarray(arr.astype(np.float32), mode="F")
    im = im.resize(size, Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def _normalize(cam: np.ndarray) -> np.ndarray:
    cam = np.maximum(np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    lo, hi = float(cam.min()), float(cam.max())
    if hi < BLANK_EPS or (hi - lo) < BLANK_EPS:
        return np.zeros_like(cam)  # ~all-zero CAM -> blank heat
    return (cam - lo) / (hi - lo)


def _warm(x: np.ndarray) -> np.ndarray:
    return np.stack([np.ones_like(x), 1.0 - 0.85 * x, 0.15 * (1.0 - x)], axis=-1)


def heatmap_display(pa: float, pb: float, ta: float, cam: np.ndarray) -> dict:
    score = float(np.clip(pa * pb, 0.0, 1.0))
    if pa < ta or score < MIN_DISPLAY_SCORE:
        state, strength = "no_strong_evidence", 0.0
        note = "No strong flood evidence to highlight. This does not rule out flooding."
    elif not _normalize(np.asarray(cam, dtype=np.float32)).any():
        state, strength = "unlocalized", 0.0
        note = "The flood score has no localized attribution map."
    else:
        state, strength = "shown", score
        note = "Stage B flood evidence; colors show relative contribution, not water boundaries or pixel probabilities."
    return {"state": state, "strength": strength, "score": score, "note": note}


def _heat_full_crop(norm: np.ndarray, geom: dict, w0: int, h0: int) -> np.ndarray:
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
    return heat_full


def _heat_full_letterbox(norm: np.ndarray, geom: dict, w0: int, h0: int) -> np.ndarray:
    # invert the pad: crop out the real-content sub-box, then resize that
    # 1:1 back onto the whole original frame (padding never gets any heat).
    left, top, right, bottom = geom["content_box_canvas"]
    cw, ch = max(1, right - left), max(1, bottom - top)
    content = norm[top:top + ch, left:left + cw]
    return _resize_float(content, (w0, h0))


def make_heatmap_png(
    frame: Image.Image, cam: np.ndarray, geom: dict,
    alpha: float = ALPHA, max_long_side: int = MAX_LONG_SIDE,
    *, strength: float = 1.0,
) -> bytes:
    w0, h0 = geom["orig_size"]
    size = geom.get("size", CROP_SIZE)
    mode = geom.get("mode", "crop")
    up = _resize_float(np.asarray(cam), (size, size))
    norm = _normalize(up)

    if mode == "crop":
        heat_full = _heat_full_crop(norm, geom, w0, h0)
    elif mode == "squash":
        # the CAM covers the whole (aspect-ignored) frame directly.
        heat_full = _resize_float(norm, (w0, h0))
    else:  # letterbox
        heat_full = _heat_full_letterbox(norm, geom, w0, h0)

    base = np.asarray(frame.convert("RGB"), dtype=np.float32)
    heat_full = heat_full * float(np.clip(strength, 0.0, 1.0))
    color = _warm(heat_full) * 255.0
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
