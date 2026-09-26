from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError


MIN_WIDTH = 16
MIN_HEIGHT = 16
FLAT_FRAME_STDDEV = 3.0
DARK_PIXEL_LIMIT = 15
MOSTLY_DARK_RATIO = 0.95
HASH_SIZE = 8
ANALYSIS_SIZE = 64
HASH_DISTANCE_LIMIT = 6

# These are fingerprints of camera-generated error frames, not image samples
# loaded at runtime. Add hashes here only after checking them against valid
# nighttime and weather frames from the same camera network.
KNOWN_ERROR_HASHES = frozenset({0xFFFFF98081F9FFFF})


@dataclass(frozen=True, slots=True)
class ImageQuality:
    valid: bool
    reason: str | None = None


def _average_hash(image: Image.Image) -> int:
    grayscale = image.convert("L").resize(
        (HASH_SIZE, HASH_SIZE), Image.Resampling.LANCZOS
    )
    pixels = list(grayscale.getdata())
    average = sum(pixels) / len(pixels)
    return sum((pixel >= average) << index for index, pixel in enumerate(pixels))


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _has_known_error_hash(image: Image.Image) -> bool:
    image_hash = _average_hash(image)
    return any(
        _hamming_distance(image_hash, known_hash) <= HASH_DISTANCE_LIMIT
        for known_hash in KNOWN_ERROR_HASHES
    )


def inspect_image(image_bytes: bytes) -> ImageQuality:
    """Validate that bytes contain a useful camera frame.

    This intentionally detects only high-confidence failures. A dark but
    detailed nighttime frame should remain eligible for inference.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            if image.width < MIN_WIDTH or image.height < MIN_HEIGHT:
                return ImageQuality(False, "image dimensions are too small")

            grayscale = image.convert("L").resize(
                (ANALYSIS_SIZE, ANALYSIS_SIZE), Image.Resampling.BILINEAR
            )
            pixels = list(grayscale.getdata())
            mean = sum(pixels) / len(pixels)
            variance = sum((pixel - mean) ** 2 for pixel in pixels) / len(pixels)

            if variance**0.5 < FLAT_FRAME_STDDEV:
                return ImageQuality(False, "flat frame")
            if (
                sum(pixel < DARK_PIXEL_LIMIT for pixel in pixels) / len(pixels)
                > MOSTLY_DARK_RATIO
            ):
                return ImageQuality(False, "mostly-black frame")
            if _has_known_error_hash(image):
                return ImageQuality(False, "known camera error image")
    except (UnidentifiedImageError, OSError, ValueError):
        return ImageQuality(False, "image cannot be decoded")

    return ImageQuality(True)


def ensure_usable_image(image_bytes: bytes) -> None:
    """Raise a non-retryable error when a camera response is not usable."""
    quality = inspect_image(image_bytes)
    if not quality.valid:
        raise InvalidImageError(quality.reason or "image failed quality checks")


class InvalidImageError(ValueError):
    """A deterministic camera-frame rejection that should not be retried."""
