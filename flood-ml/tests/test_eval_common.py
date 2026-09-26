import numpy as np
import pytest
from PIL import Image

pd = pytest.importorskip("pandas")

from eval.common import (
    boot_groups,
    confidence,
    preprocess,
    rate,
    ratio_boot,
    stage_probs,
    status_of,
    wilson,
)


def test_status_boundaries():
    ta, tb = 0.8, 0.9
    pa = np.array([0.1, 0.8, 0.8, 0.95, 0.95])
    pb = np.array([0.99, 0.5, 0.9, 0.89, 0.91])
    assert status_of(pa, pb, ta, tb).tolist() == ["dry", "wet", "flooded", "wet", "flooded"]


def test_stage_probs_and_confidence():
    pa, pb = np.array([0.2, 0.9, 0.9]), np.array([0.3, 0.1, 0.95])
    sp = stage_probs(pa, pb)
    assert np.allclose(sp["dry"] + sp["wet"] + sp["flooded"], 1)
    st = status_of(pa, pb, 0.5, 0.5)
    assert np.allclose(confidence(pa, pb, st), [0.8, 0.9 * 0.9, 0.9 * 0.95])


@pytest.mark.parametrize("size", [(450, 253), (352, 240), (300, 600), (1920, 1200)])
def test_preprocess_shape(size):
    x = preprocess(Image.new("RGB", size, (10, 200, 30)))
    assert x.shape == (224, 224, 3) and x.dtype == np.float32
    assert np.allclose(x[112, 112], [10, 200, 30], atol=1)


def test_preprocess_center_crop():
    a = np.zeros((256, 512, 3), np.uint8)
    a[:, 256:] = 255
    x = preprocess(Image.fromarray(a))
    assert x[:, :100].mean() < 5 and x[:, 124:].mean() > 250


def test_wilson():
    lo, hi = wilson(0, 10)
    assert lo == 0 and 0.2 < hi < 0.35
    assert np.isnan(wilson(0, 0)[0])


def test_ratio_boot_single_group_is_degenerate():
    lo, hi, k = ratio_boot(np.array([1, 0, 1, 0]), np.ones(4), ["g"] * 4)
    assert k == 1 and lo == hi == 0.5


def test_rate_flags():
    r = rate(np.array([1, 0, 0]), np.array([1, 1, 1]), ["a", "b", "c"])
    assert r["k"] == 1 and r["n"] == 3 and "n<30" in r["flags"] and "groups<5" in r["flags"]
    many = rate(np.zeros(100), np.ones(100), [str(i % 20) for i in range(100)])
    assert many["flags"] == [] and many["value"] == 0


def test_boot_groups_fred_uses_sequence():
    df = pd.DataFrame({"source": ["fred", "fred", "ga511"],
                       "orig_path": ["data/raw/fred/dry/KITTI-style/SeqA/front-imgs/1.png",
                                     "data/raw/fred/dry/KITTI-style/SeqB/front-imgs/2.png",
                                     "data/ga511/frames/1/2.jpg"],
                       "group_id": ["loc", "loc", "123"]})
    assert boot_groups(df).tolist() == ["fred:SeqA", "fred:SeqB", "ga511:123"]

