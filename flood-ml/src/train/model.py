"""Stage A/B backbone + head builder.

Backbone in {mobilenetv3small, efficientnetb0}, ImageNet weights, input
224x224x3 float in [0,255]. Both Keras applications ship a built-in
Rescaling (MobileNetV3, `include_preprocessing=True`) or
Rescaling+Normalization (EfficientNet, always present) as their first
layer(s), so raw 0..255 pixels are the correct input and no separate
`preprocess_input` call is needed -- one graph, in and out.

Head is fixed EXACTLY as GlobalAveragePooling2D -> Dropout -> Dense(1,
sigmoid). This matters: with a linear head after GAP, Grad-CAM's per-channel
weight (mean spatial gradient of the logit w.r.t. that channel) is the
Dense kernel weight times one positive scalar shared by every channel
(d sigmoid/d logit > 0, same for the whole image). So
`ReLU(sum_k w_k * A_k)` computed directly from the Dense kernel -- no
gradient tape needed -- equals real Grad-CAM up to that positive scale, and
after per-image max-normalization the two are identical. That is what lets
`export_onnx.py` compute the heatmap in-graph with plain ops (see
`gradcam.py` for both versions and the test that checks they agree).
"""
from __future__ import annotations

from dataclasses import dataclass

import keras

BACKBONE_MOBILENETV3SMALL = "mobilenetv3small"
BACKBONE_EFFICIENTNETB0 = "efficientnetb0"
BACKBONES = (BACKBONE_MOBILENETV3SMALL, BACKBONE_EFFICIENTNETB0)

DEFAULT_INPUT_SIZE = 224
DEFAULT_DROPOUT = 0.3

FEATURE_LAYER_NAME = "backbone"
GAP_LAYER_NAME = "gap"
DROPOUT_LAYER_NAME = "head_dropout"
DENSE_LAYER_NAME = "prob"


@dataclass
class BuiltModel:
    model: keras.Model
    backbone: keras.Model
    features: object  # KerasTensor; the last conv feature map, backbone's output
    backbone_name: str
    input_size: int


def _build_backbone(name: str, input_size: int, weights: str | None) -> keras.Model:
    # Deliberately built WITHOUT `name=FEATURE_LAYER_NAME` here, then renamed
    # below. Keras's EfficientNet applications build their ImageNet weights
    # download URL as `name + "_notop.h5"` (see
    # keras/src/applications/efficientnet.py) -- passing a custom `name` at
    # construction time silently breaks the download (a real 403 hit while
    # building this: `backbone_notop.h5` doesn't exist upstream).
    # MobileNetV3Small's weight file naming doesn't depend on `name`, but we
    # rename both the same way for consistency and to not depend on that
    # staying true in a future Keras version.
    if name == BACKBONE_MOBILENETV3SMALL:
        from keras.applications import MobileNetV3Small

        backbone = MobileNetV3Small(
            input_shape=(input_size, input_size, 3),
            include_top=False,
            weights=weights,
            pooling=None,
            include_preprocessing=True,
        )
    elif name == BACKBONE_EFFICIENTNETB0:
        from keras.applications import EfficientNetB0

        backbone = EfficientNetB0(
            input_shape=(input_size, input_size, 3),
            include_top=False,
            weights=weights,
            pooling=None,
        )
    else:
        raise ValueError(f"unknown backbone {name!r}, expected one of {BACKBONES}")
    backbone.name = FEATURE_LAYER_NAME  # plain attribute (see Operation.__init__), not a property
    return backbone


def build_model(
    backbone_name: str,
    input_size: int = DEFAULT_INPUT_SIZE,
    dropout: float = DEFAULT_DROPOUT,
    weights: str | None = "imagenet",
) -> BuiltModel:
    """`weights=None` builds a randomly-initialized backbone (no download);
    used by tests and anywhere ImageNet weights aren't wanted or reachable.
    """
    if backbone_name not in BACKBONES:
        raise ValueError(f"backbone must be one of {BACKBONES}, got {backbone_name!r}")

    inputs = keras.Input(shape=(input_size, input_size, 3), name="image")
    backbone = _build_backbone(backbone_name, input_size, weights)
    features = backbone(inputs)
    x = keras.layers.GlobalAveragePooling2D(name=GAP_LAYER_NAME)(features)
    x = keras.layers.Dropout(dropout, name=DROPOUT_LAYER_NAME)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name=DENSE_LAYER_NAME)(x)
    model = keras.Model(inputs, outputs, name=f"{backbone_name}_head")

    return BuiltModel(
        model=model,
        backbone=backbone,
        features=features,
        backbone_name=backbone_name,
        input_size=input_size,
    )


def set_backbone_trainable(built: BuiltModel, n_top_layers: int | None) -> None:
    """Phase 1 (n_top_layers falsy): freeze the whole backbone -- only the
    head trains.
    Phase 2 (n_top_layers > 0): unfreeze the top `n_top_layers` layers of the
    backbone (by position in `backbone.layers`, input-to-output order), but
    every BatchNormalization layer is always kept non-trainable. In Keras,
    `layer.trainable = False` on a BatchNormalization layer also forces it to
    run in inference mode (moving mean/variance) regardless of the outer
    model's `training` flag, which is exactly "BatchNorm kept frozen /
    inference mode" during fine-tuning at a low LR.
    """
    backbone = built.backbone
    if not n_top_layers:
        backbone.trainable = False
        return

    backbone.trainable = True
    n = len(backbone.layers)
    freeze_until = max(0, n - n_top_layers)
    for i, layer in enumerate(backbone.layers):
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = i >= freeze_until


def dense_kernel(built: BuiltModel):
    """Returns (kernel, bias) numpy arrays from the Dense(1, sigmoid) head.
    `kernel` has shape (C, 1) where C is the feature map's channel count.
    """
    dense = built.model.get_layer(DENSE_LAYER_NAME)
    kernel, bias = dense.get_weights()
    return kernel, bias


def make_grad_model(built: BuiltModel) -> keras.Model:
    """A two-output model (conv feature map, prob) sharing `built.model`'s
    weights, for GradientTape Grad-CAM (see gradcam.py) and for building the
    export graph in export_onnx.py.

    Deliberately rebuilds the whole forward chain -- `backbone -> gap ->
    dropout -> dense` -- as ONE fresh call on `built.model.input`, reusing
    the original (shared-weight) layer objects, rather than pairing
    `built.features` with `built.model.output`. Two reasons:
    1. After a model is saved and reloaded (`keras.models.load_model`), a
       stored KerasTensor like `built.features` can come back detached from
       the reloaded graph ("Output with path `0` is not connected to
       `inputs`"), even though the layers and weights reload fine.
    2. `tf.GradientTape` needs `prob` and the conv feature map it
       differentiates against to be part of the *same* call chain -- pairing
       a freshly-recomputed `features` with the model's *original* `prob`
       tensor (from a separate internal call to the same backbone) gives a
       disconnected graph and `tape.gradient(...)` silently returns `None`.
    Rebuilding one connected chain (dropout runs in inference/identity mode)
    fixes both, and is numerically identical to the original forward pass.
    """
    inputs = built.model.input
    features = built.backbone(inputs, training=False)
    x = built.model.get_layer(GAP_LAYER_NAME)(features)
    x = built.model.get_layer(DROPOUT_LAYER_NAME)(x, training=False)
    prob = built.model.get_layer(DENSE_LAYER_NAME)(x)
    return keras.Model(inputs=inputs, outputs=[features, prob])
