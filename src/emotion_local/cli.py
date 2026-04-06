from __future__ import annotations

import argparse
from pathlib import Path
import json

from .config import EmotionConfig, TrainConfig
from .data import prepare_datasets, save_prepared_splits, summarize_split
from .inference import EmotionPredictor, run_webcam
from .training import evaluate_checkpoint, train_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline local para treino e teste FER2013 + MediaPipe.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepara e salva os splits train/val/test.")
    _add_common_data_args(prepare)

    train = subparsers.add_parser("train", help="Prepara os dados e treina o modelo.")
    _add_common_data_args(train)
    _add_train_args(train)

    evaluate = subparsers.add_parser("evaluate", help="Avalia um checkpoint salvo no split de teste.")
    _add_common_data_args(evaluate)
    _add_train_args(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)

    predict = subparsers.add_parser("predict", help="Prediz emocao para uma imagem local.")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--image", type=Path, required=True)
    predict.add_argument("--device", default="auto")

    webcam = subparsers.add_parser("webcam", help="Roda inferencia em tempo real pela webcam.")
    webcam.add_argument("--checkpoint", type=Path, required=True)
    webcam.add_argument("--camera-index", type=int, default=0)
    webcam.add_argument("--device", default="auto")

    return parser


def _add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fer-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--balance-target-count", type=int, default=7000)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-face-crop", action="store_true")


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--use-landmarks", action="store_true")
    parser.add_argument("--landmark-dim", type=int, default=936)
    parser.add_argument("--landmark-hidden-dim", type=int, default=128)
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)


def _emotion_config_from_args(args) -> EmotionConfig:
    return EmotionConfig(
        fer_csv=args.fer_csv,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        target_image_size=args.image_size,
        balance_target_count=args.balance_target_count,
        train_split_seed=args.seed,
        use_face_crop=not args.disable_face_crop,
    )


def _train_config_from_args(args) -> TrainConfig:
    return TrainConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        dropout=args.dropout,
        pretrained_backbone=not args.no_pretrained,
        use_amp=not args.disable_amp,
        device=args.device,
        use_landmarks=args.use_landmarks,
        landmark_dim=args.landmark_dim,
        landmark_hidden_dim=args.landmark_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        emotion_config = _emotion_config_from_args(args)
        datasets = prepare_datasets(emotion_config)
        save_prepared_splits(datasets, emotion_config)
        print(json.dumps({name: summarize_split(datasets[name]) for name in ("train", "val", "test")}, indent=2))
        return

    if args.command == "train":
        result = train_pipeline(_emotion_config_from_args(args), _train_config_from_args(args))
        print(json.dumps(result["best_metrics"], indent=2))
        print(json.dumps(result["test"], indent=2))
        return

    if args.command == "evaluate":
        metrics = evaluate_checkpoint(args.checkpoint, _emotion_config_from_args(args), _train_config_from_args(args))
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "predict":
        predictor = EmotionPredictor(args.checkpoint, device=args.device)
        print(json.dumps(predictor.predict_image(args.image), indent=2))
        return

    if args.command == "webcam":
        run_webcam(args.checkpoint, camera_index=args.camera_index, device=args.device)
        return


if __name__ == "__main__":
    main()
