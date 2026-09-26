"""Dead-frame detection: HTTP/decoding failures are handled upstream in
snapshot.py; this module classifies frames that decoded fine but are still
not usable:

- tiny_file: suspiciously small response body
- low_variance: near-uniform image (blank/gray/black frame)
- placeholder: matches a known "camera offline" placeholder image (pHash)
- frozen_repeat: near-identical to the same view's previous frame

Dead frames are never written to frames/; a small sample of each observed
reason is kept under data/ga511/dead_samples/<reason>/ for inspection.
"""
from __future__ import annotations

import io
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

from ga511.paths import DEAD_SAMPLES_DIR, PLACEHOLDER_DIR, ensure_dirs, setup_logging

log = setup_logging("ga511_quality")

MIN_BYTES = 2000
MIN_STD = 4.0
PHASH_FROZEN_HAMMING = 2
MAE_FROZEN_THRESH = 1.0
PHASH_PLACEHOLDER_HAMMING = 6
DEAD_SAMPLE_CAP_PER_REASON = 8

DEAD_REASONS = (
    "http_error",
    "non_image",
    "tiny_file",
    "low_variance",
    "placeholder",
    "frozen_repeat",
)


def compute_phash(img: Image.Image) -> imagehash.ImageHash:
    return imagehash.phash(img)


def is_tiny(nbytes: int, min_bytes: int = MIN_BYTES) -> bool:
    return nbytes < min_bytes


def is_low_variance(img: Image.Image, min_std: float = MIN_STD) -> bool:
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    return float(arr.std()) < min_std


def load_placeholder_hashes(dirpath=PLACEHOLDER_DIR) -> list[imagehash.ImageHash]:
    hashes = []
    d = Path(dirpath)
    if not d.exists():
        return hashes
    for p in sorted(d.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        try:
            hashes.append(imagehash.phash(Image.open(p)))
        except Exception as e:  # noqa: BLE001
            log.warning("could not hash placeholder %s: %s", p.name, e)
    return hashes


def matches_placeholder(
    phash: imagehash.ImageHash,
    placeholder_hashes: list,
    thresh: int = PHASH_PLACEHOLDER_HAMMING,
) -> bool:
    return any((phash - ph) <= thresh for ph in placeholder_hashes)


def is_frozen_repeat(
    img: Image.Image,
    phash: imagehash.ImageHash,
    prev_phash: imagehash.ImageHash | None,
    prev_img: Image.Image | None = None,
    hamming_thresh: int = PHASH_FROZEN_HAMMING,
    mae_thresh: float = MAE_FROZEN_THRESH,
) -> bool:
    if prev_phash is None:
        return False
    if (phash - prev_phash) <= hamming_thresh:
        return True
    if prev_img is not None:
        try:
            a = np.asarray(img.convert("L"), dtype=np.float32)
            b = np.asarray(prev_img.convert("L"), dtype=np.float32)
            if a.shape == b.shape:
                mae = float(np.abs(a - b).mean())
                if mae <= mae_thresh:
                    return True
        except Exception as e:  # noqa: BLE001
            log.warning("frozen-repeat MAE check failed: %s", e)
    return False


def classify(
    img: Image.Image,
    nbytes: int,
    prev_phash: imagehash.ImageHash | None = None,
    prev_img: Image.Image | None = None,
    placeholder_hashes: list | None = None,
) -> tuple[str | None, imagehash.ImageHash | None]:
    """Classify a decoded frame. Returns (dead_reason or None, phash or None).

    `phash` is None only when the frame was rejected before hashing made
    sense (tiny_file), so callers should not update frozen-repeat state on
    that reason.
    """
    if is_tiny(nbytes):
        return "tiny_file", None
    phash = compute_phash(img)
    if is_low_variance(img):
        return "low_variance", phash
    if placeholder_hashes and matches_placeholder(phash, placeholder_hashes):
        return "placeholder", phash
    if is_frozen_repeat(img, phash, prev_phash, prev_img):
        return "frozen_repeat", phash
    return None, phash


def save_dead_sample(
    reason: str, view_id, ts: int, img: Image.Image, cap: int = DEAD_SAMPLE_CAP_PER_REASON
) -> Path | None:
    """Keep at most `cap` sample frames per dead reason for inspection."""
    ensure_dirs()
    out_dir = DEAD_SAMPLES_DIR / reason
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.jpg"))
    if len(existing) >= cap:
        return None
    out_path = out_dir / f"{view_id}_{ts}.jpg"
    try:
        img.convert("RGB").save(out_path, "JPEG", quality=85)
    except Exception as e:  # noqa: BLE001
        log.warning("could not save dead sample for %s: %s", reason, e)
        return None
    return out_path


def save_placeholder_candidate(img: Image.Image, name: str, dirpath=PLACEHOLDER_DIR) -> Path:
    """Manually invoked helper: save an observed offline/placeholder image
    into the pHash catalog directory.
    """
    ensure_dirs()
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    out_path = d / f"{name}.jpg"
    img.convert("RGB").save(out_path, "JPEG", quality=90)
    return out_path


def bytes_to_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()
    return img
