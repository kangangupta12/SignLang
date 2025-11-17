import argparse
import time

import numpy as np
import tensorflow as tf

from .config import CONFIG


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark inference FPS.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--seq_len", type=int, default=CONFIG.seq_len)
    parser.add_argument("--num_runs", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    model = tf.keras.models.load_model(args.model_path)
    dummy = np.random.rand(args.seq_len, CONFIG.image_size, CONFIG.image_size, 3).astype("float32")
    start = time.time()
    for _ in range(args.num_runs):
        _ = model.predict(dummy[None, ...], verbose=0)
    elapsed = time.time() - start
    fps = args.num_runs / elapsed
    print(f"Inference FPS: {fps:.2f} (sequence len {args.seq_len})")


if __name__ == "__main__":
    main()
