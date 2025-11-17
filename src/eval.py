import argparse
import os
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from .config import CONFIG
from .data import make_dataset
from .utils import load_label_encoder


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on test set.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--use_frames", action="store_true")
    parser.add_argument("--seq_len", type=int, default=CONFIG.seq_len)
    parser.add_argument("--checkpoint_path", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    _, _, test_ds, label_map = make_dataset(
        dataset_dir=args.dataset_dir,
        batch_size=CONFIG.batch_size,
        seq_len=args.seq_len,
        image_size=(CONFIG.image_size, CONFIG.image_size),
        use_frames=args.use_frames,
        cache=False,
        augment=False,
    )
    model = tf.keras.models.load_model(args.checkpoint_path)
    y_true, y_pred = [], []
    for sequences, labels in test_ds:
        preds = model.predict(sequences, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    report = classification_report(
        y_true, y_pred, target_names=sorted(label_map, key=label_map.get), output_dict=True
    )
    report_path = os.path.join(CONFIG.artifacts_dir, "metrics_report.csv")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("class,precision,recall,f1,support\n")
        for cls, stats in report.items():
            if cls in {"accuracy", "macro avg", "weighted avg"}:
                continue
            f.write(f"{cls},{stats['precision']:.4f},{stats['recall']:.4f},{stats['f1-score']:.4f},{stats['support']}\n")

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=label_map.keys(), yticklabels=label_map.keys())
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    cm_path = os.path.join(CONFIG.artifacts_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Saved report to {report_path} and confusion matrix to {cm_path}")


if __name__ == "__main__":
    main()
