from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


EMOTION_MAPPING = {0: 0, 3: 1, 4: 2, 6: 3}
EMOTION_LABELS = ["Angry", "Happy", "Sad", "Neutral"]
AFFECTNET_LABEL_ALIASES = {
    "angry": 0,
    "anger": 0,
    "happy": 1,
    "sad": 2,
    "neutral": 3,
}
DATASET_KINDS = ("fer2013", "affectnet")
EVALUATION_MODES = ("same_dataset", "cross_dataset")


@dataclass(slots=True)
class DatasetConfig:
    name: str
    kind: str
    path: Path
    validation_split: float = 0.2
    balance_target_count: int = 7000


@dataclass(slots=True)
class ExperimentConfig:
    train_dataset: DatasetConfig
    test_mode: str = "same_dataset"
    test_dataset: DatasetConfig | None = None

    @property
    def resolved_test_dataset(self) -> DatasetConfig:
        if self.test_mode == "cross_dataset" and self.test_dataset is not None:
            return self.test_dataset
        return self.train_dataset


@dataclass(slots=True)
class EmotionConfig:
    output_dir: Path = Path("artifacts")
    results_dir: Path = Path("results")
    target_image_size: int = 224
    emotions_to_keep: tuple[int, ...] = (0, 3, 4, 6)
    train_split_seed: int = 42
    use_face_crop: bool = True


@dataclass(slots=True)
class TrainConfig:
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    dropout: float = 0.3
    pin_memory: bool = True
    persistent_workers: bool = True
    pretrained_backbone: bool = True
    save_name: str = "best_emotion_model.pt"
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    device: str = "auto"
    use_landmarks: bool = False
    landmark_dim: int = 936
    landmark_hidden_dim: int = 128
    fusion_hidden_dim: int = 256
    class_names: list[str] = field(default_factory=lambda: EMOTION_LABELS.copy())
