# pr curves and eval charts
from __future__ import annotations

import base64
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from eval.common import load_image


def cam_up(cam: np.ndarray, size: int = 224) -> np.ndarray:
    c = cv2.resize(cam.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    lo, hi = c.min(), c.max()
    return (c - lo) / (hi - lo) if hi > lo else np.zeros_like(c)


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = cv2.applyColorMap((cam_up(cam, img.shape[0]) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.clip(img * (1 - alpha) + heat * alpha, 0, 255).astype(np.uint8)


def tile(path, cam=None, caption: str = "", size: int = 224, spec: dict | None = None) -> Image.Image:
    img = load_image(path, spec)
    pair = [img.astype(np.uint8)]
    if cam is not None:
        pair.append(overlay(img, cam))
    arr = np.concatenate(pair, axis=1)
    im = Image.fromarray(arr).resize((size * len(pair), size))
    canvas = Image.new("RGB", (im.width, size + 30), (250, 250, 250))
    canvas.paste(im, (0, 0))
    ImageDraw.Draw(canvas).text((4, size + 2), caption[:80], fill=(0, 0, 0))
    ImageDraw.Draw(canvas).text((4, size + 15), caption[80:160], fill=(0, 0, 0))
    return canvas


def sheet(tiles: list[Image.Image], out: Path, cols: int = 4) -> Path:
    if not tiles:
        return out
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    s = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for i, t in enumerate(tiles):
        s.paste(t, ((i % cols) * w, (i // cols) * h))
    out.parent.mkdir(parents=True, exist_ok=True)
    s.save(out, quality=88)
    return out


def b64_jpeg(img: Image.Image, q: int = 80) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    return base64.b64encode(buf.getvalue()).decode()

