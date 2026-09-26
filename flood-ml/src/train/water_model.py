"""Small frozen ImageNet encoder and a shallow trainable skip decoder."""
from pathlib import Path

import keras
import tensorflow as tf


def build_water_model(size=320, weights_path=None):
    if size % 32:
        raise ValueError("Input size must be a multiple of 32")
    backbone = keras.applications.MobileNetV3Small(
        input_shape=(size, size, 3), include_top=False, weights=None,
        include_preprocessing=True,
    )
    if weights_path is not None:
        if not Path(weights_path).is_file():
            raise FileNotFoundError("Local pretrained weights required; no download fallback")
        backbone.load_weights(weights_path)
    names = ("expanded_conv_project_bn", "expanded_conv_2_add",
             "expanded_conv_7_add", "expanded_conv_10_add")
    encoder = keras.Model(backbone.input, [backbone.get_layer(n).output for n in names],
                          name="water_encoder")
    encoder.trainable = False
    inputs = keras.Input((size, size, 3), name="image")
    features = encoder(inputs, training=False)
    x = keras.layers.Conv2D(48, 1, activation="relu")(features[-1])
    for skip, channels in zip(reversed(features[:-1]), (48, 32, 24)):
        x = keras.layers.UpSampling2D(2, interpolation="bilinear")(x)
        x = keras.layers.Concatenate()([x, skip])
        x = keras.layers.SeparableConv2D(channels, 3, padding="same", activation="relu")(x)
        x = keras.layers.SeparableConv2D(channels, 3, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(1, 1)(x)
    x = keras.layers.UpSampling2D(4, interpolation="bilinear")(x)
    outputs = keras.layers.Activation("sigmoid", name="water_prob")(x)
    return keras.Model(inputs, outputs, name="water_mobilenetv3small")


def masked_loss(target, probability):
    """Per-image BCE + soft Dice, ignoring padding in both terms."""
    truth, valid = target[..., :1], target[..., 1:]
    probability = tf.clip_by_value(probability, 1e-6, 1 - 1e-6)
    axes = (1, 2, 3)
    bce = -(truth * tf.math.log(probability)
            + (1 - truth) * tf.math.log(1 - probability))
    bce = tf.reduce_sum(bce * valid, axes) / tf.maximum(tf.reduce_sum(valid, axes), 1)
    intersection = tf.reduce_sum(truth * probability * valid, axes)
    denom = tf.reduce_sum((truth + probability) * valid, axes)
    dice_loss = 1 - (2 * intersection + 1) / (denom + 1)
    return tf.reduce_mean(bce + dice_loss)
