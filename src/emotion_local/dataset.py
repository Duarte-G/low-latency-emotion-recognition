from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .data import row_to_rgb
from .face_detection import FaceCropper


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(training: bool) -> transforms.Compose:
    items: list[object] = [transforms.Resize((224, 224))]
    if training:
        items.extend(
            [
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(0.5),
            ]
        )
    items.extend([transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return transforms.Compose(items)


class EmotionDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: transforms.Compose | None = None,
        target_size: tuple[int, int] = (224, 224),
        use_face_crop: bool = True,
        landmarks: np.ndarray | None = None,
    ) -> None:
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform or build_transforms(training=False)
        self.target_size = target_size
        self.use_face_crop = use_face_crop
        self._face_cropper = None
        self.landmarks = landmarks

    def __len__(self) -> int:
        return len(self.data)

    def _get_face_cropper(self):
        if self._face_cropper is None:
            self._face_cropper = FaceCropper(min_detection_confidence=0.5)
        return self._face_cropper

    def _extract_face_roi(self, image_rgb: np.ndarray) -> np.ndarray:
        if not self.use_face_crop:
            return image_rgb
        return self._get_face_cropper().crop_face(image_rgb)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        image_rgb = row_to_rgb(row)
        face_rgb = self._extract_face_roi(image_rgb)
        image = Image.fromarray(face_rgb).resize(self.target_size)
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(int(row["emotion"]), dtype=torch.long)

        if self.landmarks is None:
            return image_tensor, label_tensor

        landmark_tensor = torch.tensor(self.landmarks[idx], dtype=torch.float32)
        return image_tensor, landmark_tensor, label_tensor


def load_split_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)
