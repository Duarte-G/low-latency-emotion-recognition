from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


EMOTION_MAPPING = {0: 0, 3: 1, 4: 2, 6: 3}
EMOTION_LABELS = ["Angry", "Happy", "Sad", "Neutral"]


@dataclass(slots=True)
class EmotionConfig:
    fer_csv: Path
    output_dir: Path = Path("artifacts")
    results_dir: Path = Path("results")
    target_image_size: int = 224
    emotions_to_keep: tuple[int, ...] = (0, 3, 4, 6)
    balance_target_count: int = 7000
    train_split_seed: int = 42
    use_face_crop: bool = True


@dataclass(slots=True)
class TrainConfig:
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    dropout: float = 0.3
    pin_memory: bool = True
    persistent_workers: bool = True
    pretrained_backbone: bool = True
    save_name: str = "best_emotion_model.pt"
    use_amp: bool = True
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    device: str = "auto"
    use_landmarks: bool = False
    landmark_dim: int = 936
    landmark_hidden_dim: int = 128
    fusion_hidden_dim: int = 256
    class_names: list[str] = field(default_factory=lambda: EMOTION_LABELS.copy())
