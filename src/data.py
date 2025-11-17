import glob
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import cv2

from .config import CONFIG
from .utils import encode_labels


@dataclass
class DatasetPaths:
    dataset_dir: str
    use_frames: bool = False


def list_samples(dataset_dir: str, use_frames: bool) -> Tuple[List[str], List[str]]:
    classes = sorted(
        d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))
    )
    video_paths, labels = [], []
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        if use_frames:
            seq_dirs = sorted(
                d for d in glob.glob(os.path.join(cls_dir, "*")) if os.path.isdir(d)
            )
            for seq in seq_dirs:
                video_paths.append(seq)
                labels.append(cls)
        else:
            files = glob.glob(os.path.join(cls_dir, "*.mp4")) + glob.glob(
                os.path.join(cls_dir, "*.avi")
            )
            for file in files:
                video_paths.append(file)
                labels.append(cls)
    return video_paths, labels


def extract_frames_from_dir(seq_dir: str, seq_len: int, image_size: Tuple[int, int]) -> np.ndarray:
    frame_files = sorted(glob.glob(os.path.join(seq_dir, "*.jpg")))
    frames = []
    for file in frame_files[:seq_len]:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, image_size)
        frames.append(img)
    if not frames:
        raise ValueError(f"No frames in {seq_dir}")
    frames = np.array(frames)
    if len(frames) < seq_len:
        pad = np.repeat(frames[-1][None, ...], seq_len - len(frames), axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    return frames.astype("float32") / 255.0


def augment_sequence(sequence: tf.Tensor) -> tf.Tensor:
    seq = tf.image.random_flip_left_right(sequence)
    seq = tf.image.random_brightness(seq, max_delta=0.1)
    seq = tfa.image.rotate(seq, tf.random.uniform([], -0.08, 0.08))
    seq = tf.image.resize_with_crop_or_pad(seq, CONFIG.image_size + 16, CONFIG.image_size + 16)
    seq = tf.image.random_crop(seq, size=[CONFIG.seq_len, CONFIG.image_size, CONFIG.image_size, 3])
    return seq


def make_dataset(
    dataset_dir: str,
    batch_size: int,
    seq_len: int,
    image_size: Tuple[int, int],
    use_frames: bool,
    split=(0.8, 0.1, 0.1),
    cache: bool = True,
    augment: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[str, int]]:
    paths, labels = list_samples(dataset_dir, use_frames)
    label_map, _ = encode_labels(labels)
    label_indices = np.array([label_map[l] for l in labels])
    data = list(zip(paths, label_indices))
    rng = np.random.default_rng(42)
    rng.shuffle(data)
    n = len(data)
    train_end = int(split[0] * n)
    val_end = train_end + int(split[1] * n)
    train_data, val_data, test_data = data[:train_end], data[train_end:val_end], data[val_end:]

    def load_sample(path, label):
        if use_frames:
            seq = tf.numpy_function(
                extract_frames_from_dir,
                [path, seq_len, image_size],
                Tout=tf.float32,
            )
        else:
            seq = tf.numpy_function(
                lambda p: _extract_video_frames(p.decode(), seq_len, image_size),
                [path],
                Tout=tf.float32,
            )
        seq.set_shape((seq_len, image_size[0], image_size[1], 3))
        return seq, tf.one_hot(label, depth=len(label_map))

    def apply_aug(seq, label):
        return augment_sequence(seq), label

    def build_tf_dataset(items, training=False):
        paths = [p for p, _ in items]
        labels = [l for _, l in items]
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(
            lambda p, l: (tf.numpy_function(lambda x: x, [p], tf.string), l),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.map(load_sample, num_parallel_calls=tf.data.AUTOTUNE)
        if training and augment:
            ds = ds.map(apply_aug, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        if cache:
            ds = ds.cache()
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    return (
        build_tf_dataset(train_data, training=True),
        build_tf_dataset(val_data),
        build_tf_dataset(test_data),
        label_map,
    )


def _extract_video_frames(path: str, seq_len: int, image_size: Tuple[int, int]) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frames, count = [], 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(fps // CONFIG.frame_rate))
    while count < seq_len:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_id % step != 0:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, image_size)
        frames.append(frame)
        count += 1
    cap.release()
    if not frames:
        raise ValueError(f"No frames extracted from {path}")
    frames = np.array(frames, dtype=np.float32) / 255.0
    if len(frames) < seq_len:
        pad = np.repeat(frames[-1][None, ...], seq_len - len(frames), axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    return frames
