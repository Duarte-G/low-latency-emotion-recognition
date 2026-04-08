from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from .config import EmotionConfig, ExperimentConfig, TrainConfig
from .data import load_prepared_splits, prepare_experiment_datasets, prepared_splits_exist, save_prepared_splits
from .dataset import EmotionDataset, build_transforms
from .landmarks import compute_landmarks_for_dataframe
from .model import EmotionClassifier, resolve_device
from .results import (
    build_artifact_name,
    build_run_name,
    create_run_directory,
    save_accuracy_loss_plots,
    save_confusion_matrix,
    save_history_csv,
    save_json_report,
    save_training_metadata,
)


def _ensure_landmark_cache(
    datasets: dict[str, object],
    artifact_dir: Path,
    emotion_config: EmotionConfig,
    train_config: TrainConfig,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    landmark_dir = artifact_dir / "landmarks"
    landmark_arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, object] = {}

    for split in ("train", "val", "test"):
        cache_path = landmark_dir / f"{split}_landmarks.npy"
        if cache_path.exists():
            cached = np.load(cache_path)
            if cached.shape == (len(datasets[split]), train_config.landmark_dim):
                landmark_arrays[split] = cached
            else:
                metadata[split] = compute_landmarks_for_dataframe(
                    datasets[split],
                    cache_path=cache_path,
                    landmark_dim=train_config.landmark_dim,
                    use_face_crop=emotion_config.use_face_crop,
                )
                landmark_arrays[split] = np.load(cache_path)
            meta_path = cache_path.with_suffix(".json")
            if meta_path.exists():
                metadata[split] = json.loads(meta_path.read_text(encoding="utf-8"))
            continue

        metadata[split] = compute_landmarks_for_dataframe(
            datasets[split],
            cache_path=cache_path,
            landmark_dim=train_config.landmark_dim,
            use_face_crop=emotion_config.use_face_crop,
        )
        landmark_arrays[split] = np.load(cache_path)

    return landmark_arrays, metadata


def create_dataloaders(
    experiment_config: ExperimentConfig,
    emotion_config: EmotionConfig,
    train_config: TrainConfig,
) -> tuple[dict[str, object], dict[str, DataLoader], dict[str, object], Path]:
    artifact_dir = emotion_config.output_dir / build_artifact_name(experiment_config, emotion_config)
    summary_path = artifact_dir / "data_summary.json"

    if prepared_splits_exist(artifact_dir):
        datasets = load_prepared_splits(artifact_dir)
        if summary_path.exists():
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            dataset_metadata = summary_payload.get("metadata", summary_payload)
        else:
            dataset_metadata = {}
    else:
        datasets, dataset_metadata = prepare_experiment_datasets(experiment_config, emotion_config)
        save_prepared_splits(datasets, artifact_dir, dataset_metadata)

    landmark_arrays = None
    landmark_metadata: dict[str, object] = {}
    if train_config.use_landmarks:
        landmark_arrays, landmark_metadata = _ensure_landmark_cache(datasets, artifact_dir, emotion_config, train_config)

    train_dataset = EmotionDataset(
        datasets["train"],
        transform=build_transforms(training=True),
        target_size=(emotion_config.target_image_size, emotion_config.target_image_size),
        use_face_crop=emotion_config.use_face_crop,
        landmarks=None if landmark_arrays is None else landmark_arrays["train"],
    )
    val_dataset = EmotionDataset(
        datasets["val"],
        transform=build_transforms(training=False),
        target_size=(emotion_config.target_image_size, emotion_config.target_image_size),
        use_face_crop=emotion_config.use_face_crop,
        landmarks=None if landmark_arrays is None else landmark_arrays["val"],
    )
    test_dataset = EmotionDataset(
        datasets["test"],
        transform=build_transforms(training=False),
        target_size=(emotion_config.target_image_size, emotion_config.target_image_size),
        use_face_crop=emotion_config.use_face_crop,
        landmarks=None if landmark_arrays is None else landmark_arrays["test"],
    )

    common_loader_args = {
        "batch_size": train_config.batch_size,
        "num_workers": train_config.num_workers,
        "pin_memory": train_config.pin_memory,
    }
    if train_config.num_workers == 0:
        common_loader_args["persistent_workers"] = False
    else:
        common_loader_args["persistent_workers"] = train_config.persistent_workers

    loaders = {
        "train": DataLoader(train_dataset, shuffle=True, **common_loader_args),
        "val": DataLoader(val_dataset, shuffle=False, **common_loader_args),
        "test": DataLoader(test_dataset, shuffle=False, **common_loader_args),
    }
    runtime_info = {
        "landmarks": landmark_metadata,
        "prepared_data": dataset_metadata,
        "artifact_dir": str(artifact_dir),
    }
    return datasets, loaders, runtime_info, artifact_dir


def build_training_components(
    train_config: TrainConfig,
    train_df,
):
    device = resolve_device(train_config.device)
    model = EmotionClassifier(
        num_classes=len(train_config.class_names),
        dropout=train_config.dropout,
        pretrained=train_config.pretrained_backbone,
        use_landmarks=train_config.use_landmarks,
        landmark_dim=train_config.landmark_dim,
        landmark_hidden_dim=train_config.landmark_hidden_dim,
        fusion_hidden_dim=train_config.fusion_hidden_dim,
    ).to(device)

    classes = np.array(sorted(train_df["emotion"].unique()))
    class_weights = compute_class_weight("balanced", classes=classes, y=train_df["emotion"].values)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=train_config.scheduler_patience,
        factor=train_config.scheduler_factor,
    )
    return device, model, criterion, optimizer, scheduler


def _forward_step(model, batch, criterion, device, amp_enabled, scaler=None, optimizer=None):
    if len(batch) == 3:
        images, landmarks, labels = batch
        landmarks = landmarks.to(device, non_blocking=True)
    else:
        images, labels = batch
        landmarks = None
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
        logits = model(images, landmarks)
        loss = criterion(logits, labels)

    if optimizer is not None:
        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    predictions = logits.argmax(dim=1)
    return loss.detach(), predictions.detach(), labels.detach()


def run_epoch(model, loader, criterion, device, training, amp_enabled, optimizer=None, scaler=None):
    model.train(mode=training)
    total_loss = 0.0
    all_preds: list[int] = []
    all_targets: list[int] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            loss, predictions, labels = _forward_step(
                model,
                batch,
                criterion,
                device,
                amp_enabled=amp_enabled,
                scaler=scaler,
                optimizer=optimizer if training else None,
            )
            total_loss += loss.item()
            all_preds.extend(predictions.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    accuracy = 100.0 * sum(int(p == t) for p, t in zip(all_preds, all_targets)) / max(1, len(all_targets))
    f1 = f1_score(all_targets, all_preds, average="weighted")
    avg_loss = total_loss / max(1, len(loader))
    return {"loss": avg_loss, "accuracy": accuracy, "f1": f1, "preds": all_preds, "targets": all_targets}


def train_pipeline(
    experiment_config: ExperimentConfig,
    emotion_config: EmotionConfig,
    train_config: TrainConfig,
) -> dict[str, object]:
    emotion_config.output_dir.mkdir(parents=True, exist_ok=True)
    emotion_config.results_dir.mkdir(parents=True, exist_ok=True)
    datasets, loaders, runtime_info, artifact_dir = create_dataloaders(experiment_config, emotion_config, train_config)
    device, model, criterion, optimizer, scheduler = build_training_components(train_config, datasets["train"])
    run_dir = create_run_directory(emotion_config.results_dir, build_run_name(experiment_config, emotion_config, train_config))

    amp_enabled = train_config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_val_acc = 0.0
    best_metrics: dict[str, float] = {}
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_f1": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
    }

    checkpoint_path = run_dir / train_config.save_name

    if train_config.use_landmarks:
        nonzero_total = sum(int(runtime_info["landmarks"].get(split, {}).get("nonzero_landmarks", 0)) for split in ("train", "val", "test"))
        if nonzero_total == 0:
            print("Aviso: landmarks habilitados, mas nenhum vetor util foi extraido. O treino seguira com landmarks zerados.")

    for epoch in range(train_config.num_epochs):
        start = time.time()
        train_metrics = run_epoch(
            model, loaders["train"], criterion, device, training=True, amp_enabled=amp_enabled, optimizer=optimizer, scaler=scaler
        )
        val_metrics = run_epoch(model, loaders["val"], criterion, device, training=False, amp_enabled=amp_enabled)
        scheduler.step(val_metrics["accuracy"])

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_metrics = {
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "epoch": epoch + 1,
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "emotion_labels": train_config.class_names,
                    "config": {
                        "emotion": asdict(emotion_config)
                        | {
                            "output_dir": str(emotion_config.output_dir),
                            "results_dir": str(emotion_config.results_dir),
                        },
                        "train": asdict(train_config),
                        "experiment": {
                            "train_dataset": {
                                "name": experiment_config.train_dataset.name,
                                "kind": experiment_config.train_dataset.kind,
                                "path": str(experiment_config.train_dataset.path),
                            },
                            "test_mode": experiment_config.test_mode,
                            "test_dataset": {
                                "name": experiment_config.resolved_test_dataset.name,
                                "kind": experiment_config.resolved_test_dataset.kind,
                                "path": str(experiment_config.resolved_test_dataset.path),
                            },
                        },
                    },
                    "model_metadata": {
                        "use_landmarks": train_config.use_landmarks,
                        "landmark_dim": train_config.landmark_dim,
                        "landmark_hidden_dim": train_config.landmark_hidden_dim,
                        "fusion_hidden_dim": train_config.fusion_hidden_dim,
                    },
                },
                checkpoint_path,
            )

        duration = time.time() - start
        print(
            f"Epoch {epoch + 1}/{train_config.num_epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.2f}% | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.2f}% val_f1={val_metrics['f1']:.4f} | "
            f"{duration:.1f}s"
        )

    if checkpoint_path.exists():
        best_checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])

    test_metrics = run_epoch(model, loaders["test"], criterion, device, training=False, amp_enabled=amp_enabled)
    report = classification_report(
        test_metrics["targets"],
        test_metrics["preds"],
        target_names=train_config.class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(test_metrics["targets"], test_metrics["preds"]).tolist()

    result = {
        "device": str(device),
        "run_dir": str(run_dir),
        "best_checkpoint": str(checkpoint_path),
        "artifact_dir": str(artifact_dir),
        "experiment": {
            "train_dataset": experiment_config.train_dataset.name,
            "test_dataset": experiment_config.resolved_test_dataset.name,
            "test_mode": experiment_config.test_mode,
            "face_crop": emotion_config.use_face_crop,
            "landmarks": train_config.use_landmarks,
        },
        "best_metrics": best_metrics,
        "history": history,
        "runtime_info": runtime_info,
        "test": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "f1": test_metrics["f1"],
            "classification_report": report,
            "confusion_matrix": matrix,
        },
    }
    save_history_csv(history, run_dir)
    save_accuracy_loss_plots(history, run_dir)
    save_confusion_matrix(matrix, train_config.class_names, run_dir)
    save_json_report(result["test"]["classification_report"], run_dir, "classification_report.json")
    save_json_report(result, run_dir, "training_results.json")
    save_training_metadata(
        run_dir,
        experiment_config,
        emotion_config,
        train_config,
        {
            "device": str(device),
            "best_checkpoint": str(checkpoint_path),
            "artifact_dir": str(artifact_dir),
            "runtime_info": runtime_info,
            "train_rows": int(len(datasets["train"])),
            "val_rows": int(len(datasets["val"])),
            "test_rows": int(len(datasets["test"])),
        },
    )
    return result


def evaluate_checkpoint(
    checkpoint_path: Path,
    experiment_config: ExperimentConfig,
    emotion_config: EmotionConfig,
    train_config: TrainConfig,
) -> dict[str, object]:
    device = resolve_device(train_config.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata = checkpoint.get("model_metadata", {})
    if metadata.get("use_landmarks"):
        train_config.use_landmarks = True
        train_config.landmark_dim = metadata.get("landmark_dim", train_config.landmark_dim)
        train_config.landmark_hidden_dim = metadata.get("landmark_hidden_dim", train_config.landmark_hidden_dim)
        train_config.fusion_hidden_dim = metadata.get("fusion_hidden_dim", train_config.fusion_hidden_dim)

    _, loaders, runtime_info, artifact_dir = create_dataloaders(experiment_config, emotion_config, train_config)
    model = EmotionClassifier(
        num_classes=len(train_config.class_names),
        dropout=train_config.dropout,
        pretrained=False,
        use_landmarks=train_config.use_landmarks,
        landmark_dim=train_config.landmark_dim,
        landmark_hidden_dim=train_config.landmark_hidden_dim,
        fusion_hidden_dim=train_config.fusion_hidden_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    metrics = run_epoch(
        model,
        loaders["test"],
        criterion,
        device,
        training=False,
        amp_enabled=train_config.use_amp and device.type == "cuda",
    )
    return {
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "artifact_dir": str(artifact_dir),
        "runtime_info": runtime_info,
        "confusion_matrix": confusion_matrix(metrics["targets"], metrics["preds"]).tolist(),
    }
