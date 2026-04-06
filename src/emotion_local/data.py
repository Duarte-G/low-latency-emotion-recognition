from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import EMOTION_LABELS, EMOTION_MAPPING, EmotionConfig


def load_fer2013(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"emotion", "pixels", "Usage"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV nao contem colunas obrigatorias: {sorted(missing)}")
    return df


def filter_and_remap_emotions(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[df["emotion"].isin(EMOTION_MAPPING)].copy()
    filtered["emotion"] = filtered["emotion"].map(EMOTION_MAPPING)
    return filtered.reset_index(drop=True)


def pixels_to_image(pixel_string: str, size: tuple[int, int] = (48, 48)) -> np.ndarray:
    pixels = np.fromstring(pixel_string, dtype=np.float32, sep=" ")
    return pixels.reshape(size)


def augment_image(
    img: np.ndarray,
    rotation_range: float = 10.0,
    width_shift_range: float = 0.1,
    height_shift_range: float = 0.1,
    brightness_range: float = 0.2,
    horizontal_flip: bool = True,
) -> np.ndarray:
    h, w = img.shape

    if rotation_range > 0:
        angle = np.random.uniform(-rotation_range, rotation_range)
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    if width_shift_range > 0 or height_shift_range > 0:
        tx = np.random.uniform(-width_shift_range, width_shift_range) * w
        ty = np.random.uniform(-height_shift_range, height_shift_range) * h
        matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    if horizontal_flip and np.random.random() > 0.5:
        img = cv2.flip(img, 1)

    if brightness_range > 0:
        brightness = np.random.uniform(1 - brightness_range, 1 + brightness_range)
        img = np.clip(img * brightness, 0, 255)

    return img.astype(np.float32)


def balance_dataset(df: pd.DataFrame, target_count: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for emotion_idx, emotion_name in enumerate(EMOTION_LABELS):
        emotion_data = df[df["emotion"] == emotion_idx]
        current_count = len(emotion_data)

        for _, row in emotion_data.iterrows():
            rows.append(
                {
                    "emotion": emotion_idx,
                    "pixels": row["pixels"],
                    "usage": row["Usage"],
                    "augmented": False,
                }
            )

        if current_count >= target_count:
            continue

        train_samples = emotion_data[emotion_data["Usage"] == "Training"]
        if train_samples.empty:
            raise ValueError(f"Classe {emotion_name} nao possui amostras de treino para augmentation.")

        needed = target_count - current_count
        for _ in range(needed):
            sample = train_samples.sample(1).iloc[0]
            image = pixels_to_image(sample["pixels"])
            augmented = augment_image(image)
            rows.append(
                {
                    "emotion": emotion_idx,
                    "pixels": " ".join(map(str, augmented.astype(np.uint8).reshape(-1))),
                    "usage": "Training",
                    "augmented": True,
                }
            )

    return pd.DataFrame(rows)


def split_dataset(df_balanced: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_data = df_balanced[df_balanced["usage"] == "Training"].copy()
    test_data = df_balanced[df_balanced["usage"] == "PublicTest"].copy()
    train_df, val_df = train_test_split(
        train_data,
        test_size=0.2,
        stratify=train_data["emotion"],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_data.reset_index(drop=True)


def summarize_split(df: pd.DataFrame) -> dict[str, int]:
    counts = df["emotion"].value_counts().sort_index()
    return {EMOTION_LABELS[idx]: int(counts.get(idx, 0)) for idx in range(len(EMOTION_LABELS))}


def prepare_datasets(config: EmotionConfig) -> dict[str, pd.DataFrame]:
    np.random.seed(config.train_split_seed)
    raw = load_fer2013(config.fer_csv)
    filtered = filter_and_remap_emotions(raw)
    balanced = balance_dataset(filtered, config.balance_target_count)
    train_df, val_df, test_df = split_dataset(balanced, config.train_split_seed)
    return {
        "raw": raw,
        "filtered": filtered,
        "balanced": balanced,
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }


def load_prepared_splits(output_dir: Path) -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(output_dir / f"{name}.csv") for name in ("train", "val", "test")}


def prepared_splits_exist(output_dir: Path) -> bool:
    return all((output_dir / f"{name}.csv").exists() for name in ("train", "val", "test"))


def save_prepared_splits(datasets: dict[str, pd.DataFrame], config: EmotionConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        datasets[name].to_csv(config.output_dir / f"{name}.csv", index=False)

    metadata = {
        "config": {
            **asdict(config),
            "fer_csv": str(config.fer_csv),
            "output_dir": str(config.output_dir),
        },
        "counts": {name: summarize_split(datasets[name]) for name in ("train", "val", "test")},
        "total_rows": {name: int(len(datasets[name])) for name in ("train", "val", "test")},
    }
    (config.output_dir / "data_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
