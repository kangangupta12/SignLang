import json
import os
import cv2
import numpy as np
from typing import Dict, List, Tuple
from tensorflow.keras.models import load_model


def save_label_encoder(mapping: Dict[str, int], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)


def load_label_encoder(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_labels(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    classes = sorted(set(labels))
    mapping = {cls: idx for idx, cls in enumerate(classes)}
    inv_mapping = {idx: cls for cls, idx in mapping.items()}
    return mapping, inv_mapping


def extract_frames_from_video(
    video_path: str,
    seq_len: int,
    image_size: Tuple[int, int],
    frame_rate: int = 6,
) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    success = True
    sampled = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(fps // frame_rate))
    while success and sampled < seq_len:
        success, frame = cap.read()
        if not success:
            break
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_id % step != 0:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, image_size)
        frames.append(frame)
        sampled += 1
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from {video_path}")
    frames = np.array(frames)
    if frames.shape[0] < seq_len:
        pad = np.repeat(frames[-1][None, ...], seq_len - frames.shape[0], axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    return frames.astype("float32") / 255.0


def overlay_text(frame: np.ndarray, text: str, color=(0, 255, 0)) -> np.ndarray:
    output = frame.copy()
    cv2.putText(
        output,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
        cv2.LINE_AA,
    )
    return output


def load_keras_model(path: str):
    if os.path.isdir(path):
        return load_model(path)
    return load_model(path, compile=False)
