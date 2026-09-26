# simple shortcut cues (size, jpeg quality, overlays, brightness, bars)
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from eval.common import ROOT

STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], float)


def jpeg_quality(path) -> float:
    try:
        with Image.open(ROOT / path) as im:
            q = getattr(im, "quantization", None)
            if not q:
                return float("nan")
            t = np.array(q[0], float)
    except (OSError, ValueError, KeyError):
        return float("nan")
    s = float(np.median(t * 100 / STD_LUMA))
    return float(np.clip((200 - s) / 2 if s <= 100 else 5000 / s, 1, 100))


def band_text_score(gray: np.ndarray, frac: float = 0.16) -> float:
    h = gray.shape[0]
    k = max(6, int(h * frac))
    best = 0.0
    for band in (gray[:k], gray[-k:]):
        b = band.astype(np.int16)
        flips = np.abs(np.diff((b > 170).astype(np.int8), axis=1)).sum(1)
        mid = ((b > 70) & (b < 170)).mean(1)
        dark = (b < 50).mean(1)
        rows = (flips >= 12) & (mid < 0.35) & (dark > 0.25)
        best = max(best, float(rows.mean()))
    return best


def letterbox(gray: np.ndarray) -> bool:
    k = max(3, gray.shape[0] // 40)
    top, bot = gray[:k], gray[-k:]
    return bool((top.std() < 4 and top.mean() < 25) or (bot.std() < 4 and bot.mean() < 25))


def image_cues(x: np.ndarray) -> dict:
    u8 = np.clip(x, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)
    return {
        "brightness": float(gray.mean()),
        "brightness_median": float(np.median(gray)),
        "top_median": float(np.median(gray[: gray.shape[0] // 4])),
        "dark_frac": float((gray < 40).mean()),
        "clip_frac": float((gray >= 250).mean()),
        "lap_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "saturation": float(hsv[..., 1].mean()),
        "green_frac": float(((u8[..., 1] > u8[..., 0] + 10) & (u8[..., 1] > u8[..., 2] + 10)).mean()),
        "text_score": band_text_score(gray),
        "letterbox": letterbox(gray),
    }


def tag_slices(c: dict) -> dict:
    return {
        "night": c["top_median"] < 50,
        "glare": c["clip_frac"] > 0.005 and c["top_median"] < 50,
        "blurry": c["lap_var"] < 60,
        "grayscale": c["saturation"] < 12,
        "overlay_text": c["text_score"] > 0.25,
    }

