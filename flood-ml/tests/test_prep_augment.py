import numpy as np
import pytest

from prep.augment import camera_style


def _rng(seed=0):
    return np.random.default_rng(seed)


def _img(h=64, w=96):
    rng = np.random.default_rng(123)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_output_shape_and_dtype_match_input():
    img = _img(64, 96)
    out = camera_style(img, _rng(1))
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_output_value_range_is_valid_uint8():
    img = _img()
    out = camera_style(img, _rng(2))
    assert out.min() >= 0
    assert out.max() <= 255


def test_deterministic_given_same_seed():
    img = _img()
    out1 = camera_style(img, np.random.default_rng(42))
    out2 = camera_style(img, np.random.default_rng(42))
    assert np.array_equal(out1, out2)


def test_different_seeds_usually_differ():
    img = _img()
    out1 = camera_style(img, np.random.default_rng(1))
    out2 = camera_style(img, np.random.default_rng(2))
    assert not np.array_equal(out1, out2)


def test_rejects_wrong_dtype():
    bad = np.zeros((10, 10, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        camera_style(bad, _rng())


def test_rejects_wrong_ndim():
    bad = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        camera_style(bad, _rng())


def test_rejects_wrong_channel_count():
    bad = np.zeros((10, 10, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        camera_style(bad, _rng())


def test_many_seeds_never_crash_or_change_shape():
    img = _img(48, 64)
    for seed in range(30):
        out = camera_style(img, np.random.default_rng(seed))
        assert out.shape == img.shape
        assert out.dtype == np.uint8


def test_tf_camera_style_importable_without_tensorflow_installed():
    import prep.augment as augment_mod
    assert hasattr(augment_mod, "tf_camera_style")


def test_tf_camera_style_wraps_numpy_function():
    tf = pytest.importorskip("tensorflow")
    from prep.augment import tf_camera_style

    img = tf.constant(_img(32, 32))
    out = tf_camera_style(img)
    out_np = out.numpy()
    assert out_np.shape == (32, 32, 3)
    assert out_np.dtype == np.uint8


def test_tf_camera_style_accepts_a_label_tensor():
    tf = pytest.importorskip("tensorflow")
    from prep.augment import tf_camera_style

    img = tf.constant(_img(32, 32))
    out = tf_camera_style(img, tf.constant(b"flooded"))
    assert out.numpy().shape == (32, 32, 3)


def test_wet_flooded_labels_get_overlays_more_often():
    from prep.augment import P_TEXT_OVERLAY, P_TEXT_OVERLAY_WET_FLOODED
    assert P_TEXT_OVERLAY_WET_FLOODED > P_TEXT_OVERLAY


def test_label_kwarg_is_optional_and_backward_compatible():
    img = _img()
    out = camera_style(img, _rng(3))
    assert out.shape == img.shape


def test_night_style_reachable_and_shape_preserving():
    from prep.augment import _night_style

    img = _img(64, 96)
    out = _night_style(img, _rng(7))
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255

