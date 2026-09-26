from __future__ import annotations

import numpy as np
from PIL import Image

from inference.camera_move import CameraMoveDetector


def _textured_scene(seed=0, size=(320, 240)):
    rng = np.random.default_rng(seed)
    w, h = size
    arr = np.full((h, w, 3), 60, dtype=np.uint8)
    for _ in range(12):
        x0, y0 = rng.integers(0, w - 40), rng.integers(0, h - 30)
        x1, y1 = x0 + rng.integers(10, 40), y0 + rng.integers(10, 30)
        arr[y0:y1, x0:x1] = rng.integers(80, 230)
    return arr


def _build_ref(det, camera_id, scene):
    for _ in range(det.k_ref):
        det.update(camera_id, Image.fromarray(scene))


def test_reference_building_phase_reports_not_moved():
    det = CameraMoveDetector(k_ref=3)
    scene = _textured_scene()
    for _ in range(3):
        moved, _score = det.update("cam1", Image.fromarray(scene))
        assert moved is False


def test_brightness_change_not_moved():
    det = CameraMoveDetector(k_ref=1)
    scene = _textured_scene()
    _build_ref(det, "cam1", scene)
    brighter = np.clip(scene.astype(np.float32) * 1.4 + 30, 0, 255).astype(np.uint8)
    moved, _score = det.update("cam1", Image.fromarray(brighter))
    assert moved is False


def test_gamma_change_not_moved():
    det = CameraMoveDetector(k_ref=1)
    scene = _textured_scene()
    _build_ref(det, "cam1", scene)
    gamma = np.clip(255 * (scene.astype(np.float32) / 255) ** 0.5, 0, 255).astype(np.uint8)
    moved, _score = det.update("cam1", Image.fromarray(gamma))
    assert moved is False


def test_night_style_darkening_not_moved():
    det = CameraMoveDetector(k_ref=1)
    scene = _textured_scene()
    _build_ref(det, "cam1", scene)
    dark = (scene.astype(np.float32) * 0.25).astype(np.uint8)
    moved, _score = det.update("cam1", Image.fromarray(dark))
    assert moved is False


def test_heavy_crop_zoom_is_moved():
    det = CameraMoveDetector(k_ref=1)
    scene = _textured_scene()
    _build_ref(det, "cam1", scene)
    h, w = scene.shape[:2]
    cropped = scene[h // 4: h - h // 4, w // 4: w - w // 4]
    zoomed = np.asarray(Image.fromarray(cropped).resize((w, h)))
    moved, _score = det.update("cam1", Image.fromarray(zoomed))
    assert moved is True


def test_different_scene_is_moved():
    det = CameraMoveDetector(k_ref=1)
    _build_ref(det, "cam1", _textured_scene(seed=1))
    moved, _score = det.update("cam1", Image.fromarray(_textured_scene(seed=2)))
    assert moved is True


def test_cameras_are_independent():
    det = CameraMoveDetector(k_ref=1)
    _build_ref(det, "cam1", _textured_scene(seed=1))
    moved, _ = det.update("cam2", Image.fromarray(_textured_scene(seed=1)))
    assert moved is False


def test_set_reference_explicitly():
    det = CameraMoveDetector(k_ref=3)
    scene = _textured_scene(seed=5)
    det.set_reference("cam1", Image.fromarray(scene))
    moved, score = det.update("cam1", Image.fromarray(scene))
    assert moved is False
    assert score > 0.9


def test_serialization_round_trip():
    det = CameraMoveDetector(k_ref=1, threshold=0.4)
    scene = _textured_scene(seed=7)
    det.set_reference("cam1", Image.fromarray(scene))
    d = det.to_dict()
    det2 = CameraMoveDetector.from_dict(d)
    assert det2.threshold == 0.4
    moved, _score = det2.update("cam1", Image.fromarray(scene))
    assert moved is False

