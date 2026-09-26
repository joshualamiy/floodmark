# camera-style augmentation (blur, jpeg, noise, overlays, night)
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

P_DOWN_UPSCALE = 0.55
P_JPEG = 0.55
P_GAUSSIAN_NOISE = 0.30
P_POISSON_NOISE = 0.25
P_BLUR = 0.30
P_COLOR_SHIFT = 0.50
P_PERSPECTIVE_CROP = 0.25
P_TEXT_OVERLAY = 0.30
P_TEXT_OVERLAY_WET_FLOODED = 0.60
P_NIGHT = 0.35

DOWNSCALE_RANGE = (0.25, 0.6)
JPEG_QUALITY_RANGE = (15, 60)


def _downscale_upscale(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    scale = rng.uniform(*DOWNSCALE_RANGE)
    small_w, small_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _jpeg_recompress(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    quality = int(rng.integers(JPEG_QUALITY_RANGE[0], JPEG_QUALITY_RANGE[1] + 1))
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def _gaussian_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = rng.uniform(3.0, 15.0)
    noise = rng.normal(0.0, sigma, size=img.shape)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def _poisson_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    scale = rng.uniform(15.0, 60.0)
    vals = np.clip(img.astype(np.float32), 0, 255) / 255.0 * scale
    noisy = rng.poisson(vals).astype(np.float32) / scale * 255.0
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = int(rng.choice([3, 5]))
    return cv2.GaussianBlur(img, (k, k), sigmaX=0)


def _color_shift(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.astype(np.float32)
    gains = rng.uniform(0.85, 1.15, size=3)
    out = out * gains[np.newaxis, np.newaxis, :]
    out = np.clip(out, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat_scale = rng.uniform(0.7, 1.3)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    gamma = rng.uniform(0.7, 1.4)
    lut = (np.linspace(0, 1, 256) ** gamma) * 255.0
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    out = lut[out]
    return out


def _perspective_crop(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    jitter = 0.12
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dx_top = rng.uniform(0, jitter) * w
    dy_top = rng.uniform(-jitter, jitter) * h
    dst = np.float32([
        [dx_top, max(0.0, dy_top)],
        [w - 1 - dx_top, max(0.0, -dy_top)],
        [w - 1, h - 1],
        [0, h - 1],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return warped


def _night_style(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.astype(np.float32)

    gamma = rng.uniform(1.8, 3.2)
    factor = rng.uniform(0.15, 0.45)
    out = 255.0 * (out / 255.0) ** gamma * factor

    tint = np.array([1.25, 1.05, 0.65]) if rng.random() < 0.5 else np.array([0.85, 0.95, 1.25])
    out = np.clip(out * tint[np.newaxis, np.newaxis, :], 0, 255)

    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(int(rng.integers(0, 3))):
        cx, cy = rng.uniform(0, w), rng.uniform(h * 0.3, h)
        r = rng.uniform(min(w, h) * 0.05, min(w, h) * 0.25)
        bloom = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)) * rng.uniform(120, 255)
        out += bloom[..., np.newaxis]

    for _ in range(int(rng.integers(0, 2))):
        y0 = int(rng.integers(int(h * 0.4), h))
        y1 = max(0, y0 - int(rng.integers(1, 3)))
        out[y1:y0, :] += rng.uniform(100, 220)

    out = np.clip(out, 0, 255)
    out += rng.normal(0.0, rng.uniform(5.0, 20.0), size=out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _timestamp_overlay(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    day = int(rng.integers(1, 29))
    month = _MONTHS[int(rng.integers(0, 12))]
    year = int(rng.integers(2019, 2027))
    hh = int(rng.integers(0, 24))
    mm = int(rng.integers(0, 60))
    ss = int(rng.integers(0, 60))
    text = f"{day:02d}.{month}.{year} {hh:02d}:{mm:02d}:{ss:02d}"
    cam_no = int(rng.integers(1, 99))
    cam_text = f"Camera {cam_no}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.35, min(w, h) / 480.0)
    thickness = 1 if scale < 0.9 else 2
    color = (255, 255, 255) if rng.random() < 0.5 else (255, 255, 0)

    corner = rng.integers(0, 2)
    y = int(h * 0.05) if corner == 0 else int(h * 0.95)

    for txt, x_frac in ((cam_text, 0.02), (text, 0.60)):
        (tw, th), _ = cv2.getTextSize(txt, font, scale, thickness)
        x = int(w * x_frac)
        y_baseline = y if corner == 1 else y + th
        if rng.random() < 0.5:
            box_pad = 2
            cv2.rectangle(
                out,
                (x - box_pad, y_baseline - th - box_pad),
                (x + tw + box_pad, y_baseline + box_pad),
                (0, 0, 0),
                thickness=-1,
            )
        cv2.putText(out, txt, (x, y_baseline), font, scale, color, thickness, cv2.LINE_AA)
    return out


def camera_style(
    img: np.ndarray, rng: np.random.Generator, *, label: str | None = None, night_aug: bool = True,
) -> np.ndarray:
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"camera_style expects uint8 HxWx3, got shape={img.shape} dtype={img.dtype}")

    out = img
    if rng.random() < P_PERSPECTIVE_CROP:
        out = _perspective_crop(out, rng)
    if rng.random() < P_DOWN_UPSCALE:
        out = _downscale_upscale(out, rng)
    if rng.random() < P_COLOR_SHIFT:
        out = _color_shift(out, rng)
    if rng.random() < P_BLUR:
        out = _blur(out, rng)
    if night_aug and rng.random() < P_NIGHT:
        out = _night_style(out, rng)
    if rng.random() < P_GAUSSIAN_NOISE:
        out = _gaussian_noise(out, rng)
    if rng.random() < P_POISSON_NOISE:
        out = _poisson_noise(out, rng)
    text_p = P_TEXT_OVERLAY_WET_FLOODED if label in ("wet", "flooded") else P_TEXT_OVERLAY
    if rng.random() < text_p:
        out = _timestamp_overlay(out, rng)
    if rng.random() < P_JPEG:
        out = _jpeg_recompress(out, rng)

    out = np.ascontiguousarray(out)
    if out.shape != img.shape:
        out = cv2.resize(out, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return out.astype(np.uint8)


def tf_camera_style(image, label=None, night_aug: bool = True):
    import tensorflow as tf

    def _apply(np_img, np_label):
        rng = np.random.default_rng()
        lbl = np_label.decode("utf-8") if isinstance(np_label, (bytes, bytearray)) else np_label
        return camera_style(np_img, rng, label=lbl or None, night_aug=night_aug)

    label_tensor = label if label is not None else tf.constant(b"")
    out = tf.numpy_function(func=_apply, inp=[image, label_tensor], Tout=tf.uint8)
    out.set_shape(image.shape)
    return out

