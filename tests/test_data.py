import numpy as np
import tensorflow as tf
from src.data import make_dataset


def test_dataset_shapes(monkeypatch, tmp_path):
    class_names = ["a", "b"]
    sample_paths = []
    for cls in class_names:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        file_path = cls_dir / "sample.mp4"
        file_path.write_text("placeholder")
        sample_paths.append(str(file_path))

    def mock_list_samples(dataset_dir, use_frames):
        labels = ["a", "b"]
        return sample_paths, labels

    def mock_extract(path, seq_len, image_size):
        return np.ones((seq_len, image_size[0], image_size[1], 3), dtype=np.float32)

    monkeypatch.setattr("src.data.list_samples", mock_list_samples)
    monkeypatch.setattr("src.data._extract_video_frames", mock_extract)

    train_ds, val_ds, test_ds, label_map = make_dataset(
        dataset_dir=str(tmp_path),
        batch_size=2,
        seq_len=8,
        image_size=(224, 224),
        use_frames=False,
        cache=False,
        augment=False,
    )

    seqs, labels = next(iter(train_ds))
    assert seqs.shape == (2, 8, 224, 224, 3)
    assert labels.shape[-1] == len(label_map)
