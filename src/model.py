from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models

from .config import CONFIG


def _build_backbone(input_shape: Tuple[int, int, int], trainable: bool):
    if CONFIG.cnn_backbone.lower() == "mobilenetv2":
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
        backbone.trainable = trainable
        return backbone
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    backbone = models.Model(inputs, x, name="CustomCNN")
    return backbone


def build_model(
    num_classes: int,
    seq_len: int = CONFIG.seq_len,
    input_shape: Tuple[int, int, int] = (CONFIG.image_size, CONFIG.image_size, 3),
    cnn_trainable: bool = CONFIG.cnn_trainable,
) -> tf.keras.Model:
    backbone = _build_backbone(input_shape, cnn_trainable)
    inputs = layers.Input(shape=(seq_len, *input_shape))
    x = layers.TimeDistributed(backbone)(inputs)
    if len(backbone.output_shape) == 4:
        x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=False, dropout=0.3))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="SignLang_CNN_LSTM")
    optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
