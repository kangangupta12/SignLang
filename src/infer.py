import argparse
import os
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

from .config import CONFIG
from .utils import load_label_encoder, extract_frames_from_video, overlay_text


def predict_sequence(model, frames: np.ndarray, label_map: dict) -> tuple[str, np.ndarray]:
    if frames.ndim != 4:
        raise ValueError("frames must be rank-4 (seq_len, H, W, 3)")
    preds = model.predict(frames[None, ...], verbose=0)[0]
    class_idx = int(np.argmax(preds))
    inv_map = {idx: cls for cls, idx in label_map.items()}
    return inv_map[class_idx], preds


def predict_video_file(model, video_path: str, label_map: dict, seq_len: int) -> tuple[str, np.ndarray]:
    frames = extract_frames_from_video(video_path, seq_len, (CONFIG.image_size, CONFIG.image_size))
    return predict_sequence(model, frames, label_map)


def predict_from_webcam(model, label_map: dict, capture_device: int = 0, seq_len: int = CONFIG.seq_len):
    cap = cv2.VideoCapture(capture_device)
    buffer = deque(maxlen=seq_len)
    inv_map = {idx: cls for cls, idx in label_map.items()}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (CONFIG.image_size, CONFIG.image_size))
        buffer.append(resized / 255.0)
        display = frame.copy()
        if len(buffer) == seq_len:
            frames = np.array(buffer)
            preds = model.predict(frames[None, ...], verbose=0)[0]
            top_class = inv_map[int(np.argmax(preds))]
            display = overlay_text(display, f"{top_class}: {preds.max():.2f}")
        cv2.imshow("Sign Language Recognition", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Inference utilities.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--label_encoder", default=os.path.join(CONFIG.artifacts_dir, "label_encoder.json"))
    parser.add_argument("--video_path")
    parser.add_argument("--webcam", action="store_true")
    parser.add_argument("--seq_len", type=int, default=CONFIG.seq_len)
    return parser.parse_args()


def main():
    args = parse_args()
    model = tf.keras.models.load_model(args.model_path)
    label_map = load_label_encoder(args.label_encoder)
    if args.webcam:
        predict_from_webcam(model, label_map, seq_len=args.seq_len)
    elif args.video_path:
        cls, probs = predict_video_file(model, args.video_path, label_map, args.seq_len)
        print(f"Predicted: {cls}")
        for label, idx in label_map.items():
            print(f"{label}: {probs[idx]:.4f}")
    else:
        raise ValueError("Provide --video_path or --webcam flag")


if __name__ == "__main__":
    main()
