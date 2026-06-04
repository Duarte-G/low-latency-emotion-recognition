"""Geracao de artefatos de cada execucao.

Define os nomes de diretorios (artifact/run), salva os graficos de acuracia,
perda e F1, a matriz de confusao e os arquivos JSON/CSV de metricas e
metadados em results/<timestamp>_<descricao>/.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import EmotionConfig, ExperimentConfig, TrainConfig


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "run"


def build_artifact_name(experiment_config: ExperimentConfig, emotion_config: EmotionConfig) -> str:
    train_name = _slugify(experiment_config.train_dataset.name)
    test_name = _slugify(experiment_config.resolved_test_dataset.name)
    mode = "cross" if experiment_config.test_mode == "cross_dataset" else "self"
    seed = f"seed{emotion_config.train_split_seed}"
    val_split = f"val{int(experiment_config.train_dataset.validation_split * 100):02d}"
    balance = f"bal{experiment_config.train_dataset.balance_target_count}" if experiment_config.train_dataset.kind == "fer2013" else "balna"
    return "_".join([train_name, mode, test_name, seed, val_split, balance])


def build_run_name(experiment_config: ExperimentConfig, emotion_config: EmotionConfig, train_config: TrainConfig) -> str:
    dataset_name = _slugify(experiment_config.train_dataset.name)
    test_name = _slugify(experiment_config.resolved_test_dataset.name)
    test_mode = "cross" if experiment_config.test_mode == "cross_dataset" else "self"
    modality = "img-lm" if train_config.use_landmarks else "img-only"
    crop_mode = "facecrop" if emotion_config.use_face_crop else "nocrop"
    image_size = f"img{emotion_config.target_image_size}"
    batch = f"bs{train_config.batch_size}"
    epochs = f"ep{train_config.num_epochs}"
    lr = f"lr{format(train_config.learning_rate, '.0e').replace('+', '')}"
    device = _slugify(train_config.device)
    return "_".join([dataset_name, test_mode, test_name, modality, crop_mode, image_size, batch, epochs, lr, device])


def create_run_directory(results_dir: Path, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_history_csv(history: dict[str, list[float]], run_dir: Path) -> Path:
    frame = pd.DataFrame(history)
    frame.index = frame.index + 1
    frame.index.name = "epoch"
    output = run_dir / "history.csv"
    frame.to_csv(output)
    return output


def save_training_metadata(
    run_dir: Path,
    experiment_config: ExperimentConfig,
    emotion_config: EmotionConfig,
    train_config: TrainConfig,
    extra: dict[str, object],
) -> Path:
    payload = {
        "experiment_config": {
            "train_dataset": {
                "name": experiment_config.train_dataset.name,
                "kind": experiment_config.train_dataset.kind,
                "path": str(experiment_config.train_dataset.path),
                "validation_split": experiment_config.train_dataset.validation_split,
                "balance_target_count": experiment_config.train_dataset.balance_target_count,
            },
            "test_mode": experiment_config.test_mode,
            "test_dataset": {
                "name": experiment_config.resolved_test_dataset.name,
                "kind": experiment_config.resolved_test_dataset.kind,
                "path": str(experiment_config.resolved_test_dataset.path),
                "validation_split": experiment_config.resolved_test_dataset.validation_split,
                "balance_target_count": experiment_config.resolved_test_dataset.balance_target_count,
            },
        },
        "emotion_config": {
            **asdict(emotion_config),
            "output_dir": str(emotion_config.output_dir),
            "results_dir": str(emotion_config.results_dir),
        },
        "train_config": asdict(train_config),
        "extra": extra,
    }
    output = run_dir / "run_metadata.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def save_accuracy_loss_plots(history: dict[str, list[float]], run_dir: Path) -> None:
    epochs = range(1, len(history["train_accuracy"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_accuracy"], marker="o", label="Train Accuracy")
    plt.plot(epochs, history["val_accuracy"], marker="s", label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Evolution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "accuracy.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="s", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Evolution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "loss.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_f1"], marker="o", label="Train F1")
    plt.plot(epochs, history["val_f1"], marker="s", label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 Evolution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "f1.png", dpi=200)
    plt.close()


def save_confusion_matrix(confusion_matrix_data: list[list[int]], class_names: list[str], run_dir: Path) -> Path:
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_data, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    output = run_dir / "confusion_matrix.png"
    plt.savefig(output, dpi=200)
    plt.close()
    return output


def save_json_report(payload: dict[str, object], run_dir: Path, filename: str) -> Path:
    output = run_dir / filename
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output
