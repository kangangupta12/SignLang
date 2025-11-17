import argparse
import os
import cv2
import numpy as np


def generate_sequence(output_path, seq_len=30, image_size=224):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, 10, (image_size, image_size))
    x, y = np.random.randint(20, image_size - 60, size=2)
    dx, dy = np.random.randint(-5, 5, size=2)
    for _ in range(seq_len):
        frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        color = tuple(int(c) for c in np.random.randint(50, 255, size=3))
        cv2.rectangle(frame, (x, y), (x + 40, y + 40), color, -1)
        writer.write(frame)
        x = np.clip(x + dx, 0, image_size - 40)
        y = np.clip(y + dy, 0, image_size - 40)
    writer.release()


def main():
    parser = argparse.ArgumentParser(description="Synthetic sequence generator.")
    parser.add_argument("--output_dir", default="dataset/videos")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--samples_per_class", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    for cls_idx in range(args.num_classes):
        cls_name = f"class_{cls_idx}"
        cls_dir = os.path.join(args.output_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        for sample_idx in range(args.samples_per_class):
            output_path = os.path.join(cls_dir, f"sample_{sample_idx}.mp4")
            generate_sequence(output_path, seq_len=args.seq_len, image_size=args.image_size)


if __name__ == "__main__":
    main()
