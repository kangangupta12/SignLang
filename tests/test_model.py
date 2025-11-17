import numpy as np
import tensorflow as tf
from src.model import build_model
from src.config import CONFIG


def test_model_output_shape():
    model = build_model(num_classes=5, seq_len=CONFIG.seq_len, input_shape=(CONFIG.image_size, CONFIG.image_size, 3))
    dummy = tf.random.uniform((1, CONFIG.seq_len, CONFIG.image_size, CONFIG.image_size, 3))
    output = model(dummy)
    assert output.shape == (1, 5)
