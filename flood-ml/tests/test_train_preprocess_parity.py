# inference letterbox/squash must match the tf training resize
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from PIL import Image

from inference.preprocess import _letterbox_resize, _tf_bilinear
from train.data import _letterbox_resize as tf_letterbox


@pytest.mark.parametrize("shape", [(256, 455, 180, 320), (253, 450, 180, 320), (480, 640, 240, 320)])
def test_numpy_bilinear_matches_tf(shape):
    h, w, nh, nw = shape
    a = np.random.default_rng(0).uniform(0, 255, (h, w, 3)).astype(np.float32)
    ref = tf.image.resize(a, [nh, nw], method="bilinear").numpy()
    assert np.abs(ref - _tf_bilinear(a, nh, nw)).max() < 0.01


def test_letterbox_matches_training():
    a = np.random.default_rng(1).integers(0, 256, (256, 455, 3)).astype(np.uint8)
    ours, _ = _letterbox_resize(Image.fromarray(a), 320)
    ref = tf.cast(tf.clip_by_value(tf_letterbox(a, 320), 0.0, 255.0), tf.uint8).numpy().astype(np.float32)
    # float order can flip a truncation by one level
    assert np.abs(ours - ref).max() <= 1.0
