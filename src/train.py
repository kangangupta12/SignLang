import argparse
import json
import os
from datetime import datetime

import tensorflow as tf

from .config import CONFIG
from .data import make_dataset
from .model import build_model
from .utils import save_label_encoder


def parse_args():
    parser = argparse.ArgumentParser(description="Train Sign Language Recognition model.")
    parser.add_argument("--dataset_dir", required=True, help="Path to dataset root.")
    parser.add_argument("--use_frames", action="store_true", help="Use pre-extracted frames.")
    parser.add_argument("--epochs", type=int, default=CONFIG.epochs)
    parser.add_argument("--batch_size", type=int, default=CONFIG.batch_size)
    parser.add_argument("--seq_len", type=int, default=CONFIG.seq_len)
    parser.add_argument("--learning_rate", type=float, default=CONFIG.learning_rate)
    parser.add_argument("--checkpoint_dir", default=CONFIG.checkpoint_dir)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--export_saved_model", default=CONFIG.export_saved_model)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    CONFIG.ensure_dirs()
    train_ds, val_ds, test_ds, label_map = make_dataset(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        image_size=(CONFIG.image_size, CONFIG.image_size),
        use_frames=args.use_frames,
        cache=CONFIG.cache_data,
    )

    model = build_model(
        num_classes=len(label_map),
        seq_len=args.seq_len,
        input_shape=(CONFIG.image_size, CONFIG.image_size, 3),
        cnn_trainable=CONFIG.cnn_trainable,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, "best.ckpt")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_accuracy", mode="max"),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(CONFIG.artifacts_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    history_path = os.path.join(CONFIG.artifacts_dir, "history.json")
    serializable_history = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(serializable_history, f, indent=2)

    label_encoder_path = os.path.join(CONFIG.artifacts_dir, "label_encoder.json")
    save_label_encoder(label_map, label_encoder_path)

    if args.export_saved_model:
        model.save(args.export_saved_model)

    print("Training complete. Evaluating on test set...")
    test_metrics = model.evaluate(test_ds)
    print("Test metrics:", dict(zip(model.metrics_names, test_metrics)))


if __name__ == "__main__":
    main()
