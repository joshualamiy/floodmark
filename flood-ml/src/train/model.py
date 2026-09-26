# mobilenetv3 / efficientnet backbone + gap -> dropout -> dense head
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
    features: object
    backbone_name: str
    input_size: int


def _build_backbone(name: str, input_size: int, weights: str | None) -> keras.Model:
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
    backbone.name = FEATURE_LAYER_NAME
    return backbone


def build_model(
    backbone_name: str,
    input_size: int = DEFAULT_INPUT_SIZE,
    dropout: float = DEFAULT_DROPOUT,
    weights: str | None = "imagenet",
) -> BuiltModel:
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
    dense = built.model.get_layer(DENSE_LAYER_NAME)
    kernel, bias = dense.get_weights()
    return kernel, bias


def make_grad_model(built: BuiltModel) -> keras.Model:
    inputs = built.model.input
    features = built.backbone(inputs, training=False)
    x = built.model.get_layer(GAP_LAYER_NAME)(features)
    x = built.model.get_layer(DROPOUT_LAYER_NAME)(x, training=False)
    prob = built.model.get_layer(DENSE_LAYER_NAME)(x)
    return keras.Model(inputs=inputs, outputs=[features, prob])

