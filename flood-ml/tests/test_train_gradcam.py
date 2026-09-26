import keras
import numpy as np
import pytest

pytest.importorskip("tensorflow")

from train.gradcam import (
    cam_from_dense_weights,
    gradcam_batch,
    normalize_cam,
    overlay_heatmap,
)
from train.model import build_model, make_grad_model

INPUT_SIZE = 96


def _toy_head_model(seed=0):
    inputs = keras.Input(shape=(8, 8, 3))
    conv = keras.layers.Conv2D(
        6, 3, padding="same", activation="relu",
        kernel_initializer=keras.initializers.GlorotUniform(seed=seed),
        name="backbone",
    )(inputs)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(conv)
    x = keras.layers.Dropout(0.3, name="head_dropout")(x)
    outputs = keras.layers.Dense(
        1, activation="sigmoid", name="prob",
        kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 1),
    )(x)
    model = keras.Model(inputs, outputs)
    return model, conv


def test_cam_from_dense_weights_matches_gradcam_up_to_positive_scale():
    model, conv = _toy_head_model(seed=0)
    grad_model = keras.Model(inputs=model.input, outputs=[conv, model.output])
    x = np.random.default_rng(0).uniform(0, 1, size=(4, 8, 8, 3)).astype("float32")

    conv_out, _prob = grad_model(x, training=False)
    grad_cam, _ = gradcam_batch(grad_model, x)
    kernel, _bias = model.get_layer("prob").get_weights()
    dense_cam = cam_from_dense_weights(conv_out.numpy(), kernel)

    assert grad_cam.shape == dense_cam.shape

    grad_norm = normalize_cam(grad_cam)
    dense_norm = normalize_cam(dense_cam)
    for i in range(x.shape[0]):
        a, b = grad_norm[i].reshape(-1), dense_norm[i].reshape(-1)
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            continue
        r = np.corrcoef(a, b)[0, 1]
        assert r > 0.99, f"image {i}: pearson r={r}"
    assert np.allclose(grad_norm, dense_norm, atol=1e-3)


def test_cam_is_nonnegative_after_relu():
    built = build_model("mobilenetv3small", input_size=INPUT_SIZE, weights=None)
    grad_model = make_grad_model(built)
    x = np.random.default_rng(1).uniform(0, 255, size=(2, INPUT_SIZE, INPUT_SIZE, 3)).astype("float32")
    cam, _prob = gradcam_batch(grad_model, x)
    assert np.all(cam >= 0.0)


def test_normalize_cam_maps_to_unit_range():
    cam = np.array([[[0.0, 2.0], [4.0, 1.0]]], dtype=np.float32)
    norm = normalize_cam(cam)
    assert norm.min() == pytest.approx(0.0)
    assert norm.max() == pytest.approx(1.0)


def test_overlay_heatmap_shape_and_dtype():
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, size=(64, 80, 3), dtype=np.uint8)
    cam = rng.uniform(0, 1, size=(3, 3)).astype(np.float32)
    overlay = overlay_heatmap(image, cam)
    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8

