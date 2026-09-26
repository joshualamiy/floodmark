# flags re-aimed cameras using edges only, not color
from __future__ import annotations

import numpy as np
from PIL import Image

from .preprocess import to_pil

REF_SIZE = (64, 64)
DEFAULT_K_REF = 3
DEFAULT_THRESHOLD = 0.35
MAX_SHIFT = 2


def _gray_downscaled(image, size=REF_SIZE) -> np.ndarray:
    img = to_pil(image).convert("L").resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def _sobel(gray: np.ndarray) -> np.ndarray:
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    padded = np.pad(gray, 1, mode="edge")
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    for i in range(3):
        for j in range(3):
            window = padded[i:i + gray.shape[0], j:j + gray.shape[1]]
            gx += kx[i, j] * window
            gy += kx[j, i] * window
    return np.hypot(gx, gy)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 1e-6 else 0.0


def _best_shift_ncc(ref: np.ndarray, cur: np.ndarray, max_shift: int = MAX_SHIFT) -> float:
    h, w = ref.shape
    m = max_shift
    best = -1.0
    for dy in range(-m, m + 1):
        for dx in range(-m, m + 1):
            shifted = np.roll(np.roll(cur, dy, axis=0), dx, axis=1)
            score = _ncc(ref[m:h - m, m:w - m], shifted[m:h - m, m:w - m])
            best = max(best, score)
    return best


class CameraMoveDetector:
    def __init__(self, k_ref: int = DEFAULT_K_REF, threshold: float = DEFAULT_THRESHOLD):
        self.k_ref = k_ref
        self.threshold = threshold
        self._building: dict[str, list] = {}
        self._refs: dict[str, np.ndarray] = {}

    def set_reference(self, camera_id: str, frame) -> None:
        self._refs[camera_id] = _sobel(_gray_downscaled(frame))
        self._building.pop(camera_id, None)

    def update(self, camera_id: str, frame) -> tuple[bool, float]:
        edges = _sobel(_gray_downscaled(frame))
        if camera_id not in self._refs:
            bucket = self._building.setdefault(camera_id, [])
            bucket.append(edges)
            if len(bucket) >= self.k_ref:
                self._refs[camera_id] = np.median(np.stack(bucket), axis=0)
                del self._building[camera_id]
            return False, 1.0
        score = _best_shift_ncc(self._refs[camera_id], edges)
        return score < self.threshold, score

    def to_dict(self) -> dict:
        return {
            "k_ref": self.k_ref,
            "threshold": self.threshold,
            "refs": {k: v.tolist() for k, v in self._refs.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> CameraMoveDetector:
        obj = cls(k_ref=d.get("k_ref", DEFAULT_K_REF), threshold=d.get("threshold", DEFAULT_THRESHOLD))
        obj._refs = {k: np.asarray(v, dtype=np.float32) for k, v in d.get("refs", {}).items()}
        return obj

