import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
pytest.importorskip("keras")

from train.water_model import build_water_model, masked_loss


def test_loss_and_gradient_ignore_padding():
    target = tf.constant([[[[1., 1.], [0., 0.]]]])
    prediction = tf.Variable([[[[0.8], [0.9]]]])
    with tf.GradientTape() as tape:
        first = masked_loss(target, prediction)
    gradient = tape.gradient(first, prediction).numpy()
    second = masked_loss(target, tf.constant([[[[0.8], [0.1]]]]))
    assert float(first) == pytest.approx(float(second))
    assert gradient[0, 0, 1, 0] == 0
    assert gradient[0, 0, 0, 0] != 0


def test_small_full_frame_model_no_download_and_frozen_encoder():
    model = build_water_model(size=32, weights_path=None)
    output = model(np.zeros((1, 32, 32, 3), np.float32))
    assert tuple(output.shape) == (1, 32, 32, 1)
    assert np.isfinite(output.numpy()).all()
    assert model.count_params() < 1_100_000
    assert not model.get_layer("water_encoder").trainable
