import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    image_size: int = int(os.getenv("IMAGE_SIZE", 224))
    seq_len: int = int(os.getenv("SEQ_LEN", 30))
    batch_size: int = int(os.getenv("BATCH_SIZE", 8))
    epochs: int = int(os.getenv("EPOCHS", 30))
    learning_rate: float = float(os.getenv("LEARNING_RATE", 1e-4))
    cache_data: bool = os.getenv("CACHE_DATA", "true").lower() == "true"
    mixed_precision: bool = os.getenv("USE_MIXED_PRECISION", "false").lower() == "true"
    cnn_backbone: str = os.getenv("CNN_BACKBONE", "MobileNetV2")
    cnn_trainable: bool = os.getenv("CNN_TRAINABLE", "false").lower() == "true"
    checkpoint_dir: str = os.getenv("CHECKPOINT_DIR", "models/checkpoints")
    artifacts_dir: str = os.getenv("ARTIFACTS_DIR", "artifacts")
    frame_rate: int = int(os.getenv("FRAME_RATE", 6))
    model_name: str = os.getenv("MODEL_NAME", "signlang_cnn_lstm")
    export_saved_model: str | None = os.getenv("EXPORT_SAVED_MODEL")

    def ensure_dirs(self) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs("models", exist_ok=True)


CONFIG = Config()
