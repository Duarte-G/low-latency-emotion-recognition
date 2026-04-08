from __future__ import annotations

from pathlib import Path

from .config import DatasetConfig, ExperimentConfig


DATASET_DISPLAY_NAMES = {
    "fer2013": "FER-2013",
    "affectnet": "AffectNet",
}


def default_dataset_root() -> Path:
    return Path("dataset")


def default_fer_csv(dataset_root: Path) -> Path:
    return dataset_root / "FER-2013" / "fer2013.csv"


def default_affectnet_dir(dataset_root: Path) -> Path:
    return dataset_root / "AffectNet"


def build_dataset_config(
    dataset_kind: str,
    dataset_root: Path,
    fer_csv: Path | None = None,
    affectnet_dir: Path | None = None,
    validation_split: float = 0.2,
    fer_balance_target_count: int = 7000,
) -> DatasetConfig:
    if dataset_kind == "fer2013":
        path = fer_csv or default_fer_csv(dataset_root)
        return DatasetConfig(
            name=DATASET_DISPLAY_NAMES[dataset_kind],
            kind=dataset_kind,
            path=path,
            validation_split=validation_split,
            balance_target_count=fer_balance_target_count,
        )

    if dataset_kind == "affectnet":
        path = affectnet_dir or default_affectnet_dir(dataset_root)
        return DatasetConfig(
            name=DATASET_DISPLAY_NAMES[dataset_kind],
            kind=dataset_kind,
            path=path,
            validation_split=validation_split,
            balance_target_count=0,
        )

    raise ValueError(f"Dataset nao suportado: {dataset_kind}")


def build_experiment_config(
    train_dataset_kind: str,
    test_mode: str,
    dataset_root: Path,
    fer_csv: Path | None = None,
    affectnet_dir: Path | None = None,
    test_dataset_kind: str | None = None,
    validation_split: float = 0.2,
    fer_balance_target_count: int = 7000,
) -> ExperimentConfig:
    train_dataset = build_dataset_config(
        train_dataset_kind,
        dataset_root=dataset_root,
        fer_csv=fer_csv,
        affectnet_dir=affectnet_dir,
        validation_split=validation_split,
        fer_balance_target_count=fer_balance_target_count,
    )

    if test_mode == "cross_dataset":
        if not test_dataset_kind:
            raise ValueError("Teste cruzado exige um dataset de teste.")
        test_dataset = build_dataset_config(
            test_dataset_kind,
            dataset_root=dataset_root,
            fer_csv=fer_csv,
            affectnet_dir=affectnet_dir,
            validation_split=validation_split,
            fer_balance_target_count=fer_balance_target_count,
        )
    else:
        test_dataset = None

    return ExperimentConfig(
        train_dataset=train_dataset,
        test_mode=test_mode,
        test_dataset=test_dataset,
    )


def describe_experiment(experiment_config: ExperimentConfig, use_face_crop: bool, use_landmarks: bool) -> dict[str, object]:
    return {
        "train_dataset": experiment_config.train_dataset.name,
        "test_mode": experiment_config.test_mode,
        "test_dataset": experiment_config.resolved_test_dataset.name,
        "face_crop": use_face_crop,
        "landmarks": use_landmarks,
    }
