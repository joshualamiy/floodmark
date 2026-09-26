import numpy as np
import pytest
from PIL import Image

pytest.importorskip("pandas")
pytest.importorskip("cv2")

from eval.cues import band_text_score, image_cues, jpeg_quality, tag_slices
from eval.gradcam_eval import energy_in, peak_in, to_crop


def test_jpeg_quality_estimate(tmp_path):
    rng = np.random.default_rng(0)
    im = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    for q in (50, 75, 95):
        p = tmp_path / f"q{q}.jpg"
        im.save(p, quality=q)
        assert abs(jpeg_quality(p) - q) <= 3
    im.save(tmp_path / "x.png")
    assert np.isnan(jpeg_quality(tmp_path / "x.png"))


def test_text_score_on_synthetic_overlay():
    g = np.full((224, 224), 120, np.uint8)
    assert band_text_score(g) == 0
    g[:30, :200] = 5
    g[5:25, 10:190:6] = 255
    assert band_text_score(g) > 0.25


def test_cues_and_tags_dark_image():
    x = np.zeros((224, 224, 3), np.float32)
    x[100:120, 100:120] = 255
    c = image_cues(x)
    t = tag_slices(c)
    assert t["night"] and t["glare"] and t["grayscale"] and not t["overlay_text"]


def test_to_crop_and_energy():
    m = np.zeros((256, 512), bool)
    m[:, 256:] = True
    r = to_crop(m)
    assert r.shape == (224, 224) and r[:, :100].sum() == 0 and r[:, 130:].all()
    cam = np.zeros((7, 7))
    cam[:, 5:] = 1
    assert energy_in(cam, r) > 0.9 and peak_in(cam, r)
