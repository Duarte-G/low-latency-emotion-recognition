"""Leitura, filtragem, balanceamento e divisao (splits) dos datasets.

Trata o FER-2013 (CSV de pixels) e o AffectNet (imagens .jpg em pastas por
classe), padronizando ambos para as 4 classes de emocao do projeto.
"""

from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import AFFECTNET_LABEL_ALIASES, DatasetConfig, EMOTION_LABELS, EMOTION_MAPPING, EmotionConfig, ExperimentConfig


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


def row_to_rgb(row: pd.Series) -> np.ndarray:
    image_path = row.get("image_path")
    if isinstance(image_path, str) and image_path.strip():
        image_path = image_path.replace("\\", "/")
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Nao foi possivel abrir a imagem: {image_path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    pixel_string = row.get("pixels")
    if isinstance(pixel_string, str) and pixel_string.strip():
        pixels = np.fromstring(pixel_string, dtype=np.float32, sep=" ")
        side = int(np.sqrt(len(pixels)))
        image = pixels.reshape(side, side).astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    raise ValueError("A amostra nao contem nem 'pixels' nem 'image_path'.")


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


def balance_training_split(df: pd.DataFrame, target_count: int, dataset_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for emotion_idx, emotion_name in enumerate(EMOTION_LABELS):
        emotion_data = df[df["emotion"] == emotion_idx]
        current_count = len(emotion_data)

        for _, row in emotion_data.iterrows():
            rows.append(
                {
                    "emotion": emotion_idx,
                    "pixels": row["pixels"],
                    "image_path": row.get("image_path"),
                    "usage": row["usage"],
                    "augmented": bool(row.get("augmented", False)),
                    "source_dataset": dataset_name,
                }
            )

        if current_count >= target_count:
            continue

        if emotion_data.empty:
            raise ValueError(f"Classe {emotion_name} nao possui amostras de treino para augmentation.")

        needed = target_count - current_count
        for _ in range(needed):
            sample = emotion_data.sample(1).iloc[0]
            image = pixels_to_image(sample["pixels"])
            augmented = augment_image(image)
            rows.append(
                {
                    "emotion": emotion_idx,
                    "pixels": " ".join(map(str, augmented.astype(np.uint8).reshape(-1))),
                    "image_path": None,
                    "usage": "Training",
                    "augmented": True,
                    "source_dataset": dataset_name,
                }
            )

    return pd.DataFrame(rows)


def split_training_validation(df: pd.DataFrame, seed: int, validation_split: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        df,
        test_size=validation_split,
        stratify=df["emotion"],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def split_fer_dataset(df_balanced: pd.DataFrame, seed: int, validation_split: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_data = df_balanced[df_balanced["usage"] == "Training"].copy()
    test_data = df_balanced[df_balanced["usage"] == "PublicTest"].copy()
    train_df, val_df = split_training_validation(train_data, seed=seed, validation_split=validation_split)
    return train_df, val_df, test_data.reset_index(drop=True)


def summarize_split(df: pd.DataFrame) -> dict[str, int]:
    counts = df["emotion"].value_counts().sort_index()
    return {EMOTION_LABELS[idx]: int(counts.get(idx, 0)) for idx in range(len(EMOTION_LABELS))}


def _normalize_fer_dataframe(filtered: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    return (
        filtered.rename(columns={"Usage": "usage"})
        .assign(
            image_path=None,
            augmented=False,
            source_dataset=dataset_name,
        )[["emotion", "pixels", "image_path", "usage", "augmented", "source_dataset"]]
        .reset_index(drop=True)
    )


def prepare_fer2013_datasets(dataset_config: DatasetConfig, emotion_config: EmotionConfig) -> dict[str, pd.DataFrame]:
    np.random.seed(emotion_config.train_split_seed)
    raw = load_fer2013(dataset_config.path)
    filtered = filter_and_remap_emotions(raw)
    normalized = _normalize_fer_dataframe(filtered, dataset_config.name)
    train_pool = normalized[normalized["usage"] == "Training"].reset_index(drop=True)
    test_df = normalized[normalized["usage"] == "PublicTest"].reset_index(drop=True)
    balanced_train = balance_training_split(train_pool, dataset_config.balance_target_count, dataset_name=dataset_config.name)
    train_df, val_df = split_training_validation(
        balanced_train,
        seed=emotion_config.train_split_seed,
        validation_split=dataset_config.validation_split,
    )
    return {
        "raw": raw,
        "filtered": filtered,
        "train_pool": train_pool,
        "balanced": balanced_train,
        "train": train_df,
        "val": val_df,
        "test": test_df.reset_index(drop=True),
    }


def _emotion_index_from_folder(folder_name: str) -> int | None:
    return AFFECTNET_LABEL_ALIASES.get(folder_name.strip().lower())


def _load_affectnet_split(split_dir: Path, usage: str, dataset_name: str) -> pd.DataFrame:
    if not split_dir.exists():
        raise FileNotFoundError(f"Diretorio do AffectNet nao encontrado: {split_dir}")

    rows: list[dict[str, object]] = []
    for emotion_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        emotion_idx = _emotion_index_from_folder(emotion_dir.name)
        if emotion_idx is None:
            continue

        for image_path in sorted(path for path in emotion_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS):
            rows.append(
                {
                    "emotion": emotion_idx,
                    "pixels": None,
                    "image_path": str(image_path),
                    "usage": usage,
                    "augmented": False,
                    "source_dataset": dataset_name,
                }
            )

    if not rows:
        raise ValueError(f"Nenhuma imagem valida foi encontrada em {split_dir}.")
    return pd.DataFrame(rows)


def prepare_affectnet_datasets(dataset_config: DatasetConfig, emotion_config: EmotionConfig) -> dict[str, pd.DataFrame]:
    train_root = dataset_config.path / "train"
    validation_root = dataset_config.path / "validation"

    train_pool = _load_affectnet_split(train_root, usage="Training", dataset_name=dataset_config.name)
    train_df, val_df = split_training_validation(
        train_pool,
        seed=emotion_config.train_split_seed,
        validation_split=dataset_config.validation_split,
    )
    test_df = _load_affectnet_split(validation_root, usage="PublicTest", dataset_name=dataset_config.name)
    return {
        "raw": pd.concat([train_pool, test_df], ignore_index=True),
        "train_pool": train_pool,
        "train": train_df,
        "val": val_df,
        "test": test_df.reset_index(drop=True),
    }


def prepare_dataset(config: DatasetConfig, emotion_config: EmotionConfig) -> dict[str, pd.DataFrame]:
    if config.kind == "fer2013":
        return prepare_fer2013_datasets(config, emotion_config)
    if config.kind == "affectnet":
        return prepare_affectnet_datasets(config, emotion_config)
    raise ValueError(f"Tipo de dataset nao suportado: {config.kind}")


def prepare_experiment_datasets(
    experiment_config: ExperimentConfig,
    emotion_config: EmotionConfig,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    train_source = prepare_dataset(experiment_config.train_dataset, emotion_config)
    if experiment_config.test_mode == "cross_dataset":
        test_source = prepare_dataset(experiment_config.resolved_test_dataset, emotion_config)
    else:
        test_source = train_source

    datasets = {
        "train": train_source["train"].reset_index(drop=True),
        "val": train_source["val"].reset_index(drop=True),
        "test": test_source["test"].reset_index(drop=True),
    }
    metadata = {
        "train_dataset": {
            "name": experiment_config.train_dataset.name,
            "kind": experiment_config.train_dataset.kind,
            "path": str(experiment_config.train_dataset.path),
        },
        "test_dataset": {
            "name": experiment_config.resolved_test_dataset.name,
            "kind": experiment_config.resolved_test_dataset.kind,
            "path": str(experiment_config.resolved_test_dataset.path),
        },
        "test_mode": experiment_config.test_mode,
        "counts": {name: summarize_split(datasets[name]) for name in ("train", "val", "test")},
    }
    return datasets, metadata


def load_prepared_splits(output_dir: Path) -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(output_dir / f"{name}.csv") for name in ("train", "val", "test")}


def prepared_splits_exist(output_dir: Path) -> bool:
    return all((output_dir / f"{name}.csv").exists() for name in ("train", "val", "test"))


def save_prepared_splits(datasets: dict[str, pd.DataFrame], output_dir: Path, metadata: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        datasets[name].to_csv(output_dir / f"{name}.csv", index=False)

    payload = {
        "metadata": metadata,
        "counts": {name: summarize_split(datasets[name]) for name in ("train", "val", "test")},
        "total_rows": {name: int(len(datasets[name])) for name in ("train", "val", "test")},
    }
    (output_dir / "data_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
