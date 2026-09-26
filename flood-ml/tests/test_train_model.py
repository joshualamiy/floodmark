import keras
import numpy as np
import pytest

pytest.importorskip("tensorflow")

from train.model import (
    BACKBONES,
    DENSE_LAYER_NAME,
    DROPOUT_LAYER_NAME,
    GAP_LAYER_NAME,
    build_model,
    dense_kernel,
    make_grad_model,
    set_backbone_trainable,
)

INPUT_SIZE = 96


@pytest.mark.parametrize("backbone_name", BACKBONES)
def test_build_model_head_is_gap_dropout_dense(backbone_name):
    built = build_model(backbone_name, input_size=INPUT_SIZE, weights=None)
    layer_names = [layer.name for layer in built.model.layers]
    assert GAP_LAYER_NAME in layer_names
    assert DROPOUT_LAYER_NAME in layer_names
    assert DENSE_LAYER_NAME in layer_names
    assert layer_names.index(GAP_LAYER_NAME) < layer_names.index(DROPOUT_LAYER_NAME) < layer_names.index(DENSE_LAYER_NAME)

    dense = built.model.get_layer(DENSE_LAYER_NAME)
    assert isinstance(dense, keras.layers.Dense)
    assert dense.units == 1
    assert dense.activation.__name__ == "sigmoid"

    assert built.model.output_shape == (None, 1)


@pytest.mark.parametrize("backbone_name", BACKBONES)
def test_forward_pass_runs_and_is_finite(backbone_name):
    built = build_model(backbone_name, input_size=INPUT_SIZE, weights=None)
    x = np.random.default_rng(0).uniform(0, 255, size=(2, INPUT_SIZE, INPUT_SIZE, 3)).astype("float32")
    out = built.model(x, training=False).numpy()
    assert out.shape == (2, 1)
    assert np.all(np.isfinite(out))
    assert np.all((out >= 0) & (out <= 1))


def test_dense_kernel_shape_matches_feature_channels():
    built = build_model("mobilenetv3small", input_size=INPUT_SIZE, weights=None)
    kernel, bias = dense_kernel(built)
    n_channels = built.model.get_layer("gap").output.shape[-1]
    assert kernel.shape == (n_channels, 1)
    assert bias.shape == (1,)


def test_set_backbone_trainable_freezes_everything_when_falsy():
    built = build_model("mobilenetv3small", input_size=INPUT_SIZE, weights=None)
    set_backbone_trainable(built, 0)
    assert built.backbone.trainable is False


def test_set_backbone_trainable_keeps_batchnorm_frozen_when_finetuning():
    built = build_model("mobilenetv3small", input_size=INPUT_SIZE, weights=None)
    set_backbone_trainable(built, n_top_layers=10)
    assert built.backbone.trainable is True
    bn_layers = [l for l in built.backbone.layers if isinstance(l, keras.layers.BatchNormalization)]
    assert len(bn_layers) > 0
    assert all(l.trainable is False for l in bn_layers)
    non_bn_top = [l for l in built.backbone.layers[-10:] if not isinstance(l, keras.layers.BatchNormalization)]
    assert any(l.trainable for l in non_bn_top)


def test_make_grad_model_outputs_conv_map_and_prob_connected():
    built = build_model("mobilenetv3small", input_size=INPUT_SIZE, weights=None)
    grad_model = make_grad_model(built)
    x = np.random.default_rng(1).uniform(0, 255, size=(2, INPUT_SIZE, INPUT_SIZE, 3)).astype("float32")
    conv_out, prob = grad_model(x, training=False)
    assert conv_out.shape[0] == 2
    assert conv_out.shape[-1] == dense_kernel(built)[0].shape[0]
    assert prob.shape == (2, 1)
    direct = built.model(x, training=False).numpy()
    assert np.allclose(prob.numpy(), direct, atol=1e-5)


def test_make_grad_model_survives_save_and_reload(tmp_path):
    built = build_model("mobilenetv3small", input_size=INPUT_SIZE, weights=None)
    path = tmp_path / "model.keras"
    built.model.save(path)
    reloaded = keras.models.load_model(path)

    from train.model import FEATURE_LAYER_NAME, BuiltModel

    backbone = reloaded.get_layer(FEATURE_LAYER_NAME)
    reloaded_built = BuiltModel(model=reloaded, backbone=backbone, features=None,
                                 backbone_name="mobilenetv3small", input_size=INPUT_SIZE)
    grad_model = make_grad_model(reloaded_built)
    x = np.random.default_rng(2).uniform(0, 255, size=(1, INPUT_SIZE, INPUT_SIZE, 3)).astype("float32")
    conv_out, prob = grad_model(x, training=False)
    assert np.all(np.isfinite(conv_out.numpy()))
    assert np.all(np.isfinite(prob.numpy()))

