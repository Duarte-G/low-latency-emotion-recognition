"""Interface de linha de comando do projeto.

Define os comandos: prepare, train, evaluate, predict, webcam, serve,
benchmark, wizard e compare.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from .comparison import export_comparison
from .config import EmotionConfig, TrainConfig
from .data import prepare_experiment_datasets, save_prepared_splits, summarize_split
from .experiments import build_experiment_config, default_dataset_root, describe_experiment
from .inference import EmotionPredictor, run_webcam
from .server import run_server
from .results import build_artifact_name
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

    serve = subparsers.add_parser("serve", help="Inicia um servidor HTTP para receber imagens da Unity.")
    serve.add_argument("--checkpoint", type=Path, required=True)
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=5000)
    serve.add_argument("--device", default="auto")
    serve.add_argument("--debug", action="store_true")

    benchmark = subparsers.add_parser("benchmark", help="Executa todas as combinacoes de experimento em lote.")
    _add_common_data_args(benchmark)
    _add_train_args(benchmark)

    wizard = subparsers.add_parser("wizard", help="Abre um menu interativo para configurar e rodar um experimento.")
    _add_common_data_args(wizard)
    _add_train_args(wizard)

    compare = subparsers.add_parser("compare", help="Compara resultados de duas ou mais execucoes salvas.")
    compare.add_argument("--results-dir", type=Path, default=Path("results"))
    compare.add_argument("--run-dir", type=Path, action="append", dest="run_dirs")
    compare.add_argument("--latest", type=int)
    compare.add_argument("--name", default="comparison")

    return parser


def _add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--train-dataset", choices=["fer2013", "affectnet"], default="fer2013")
    parser.add_argument("--test-mode", choices=["same_dataset", "cross_dataset"], default="same_dataset")
    parser.add_argument("--test-dataset", choices=["fer2013", "affectnet"])
    parser.add_argument("--fer-csv", type=Path)
    parser.add_argument("--affectnet-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--balance-target-count", type=int, default=7000)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-face-crop", action="store_true")


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--use-landmarks", action="store_true")
    parser.add_argument("--landmark-dim", type=int, default=936)
    parser.add_argument("--landmark-hidden-dim", type=int, default=128)
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16"], default="bfloat16")


def _emotion_config_from_args(args) -> EmotionConfig:
    return EmotionConfig(
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        target_image_size=args.image_size,
        train_split_seed=args.seed,
        use_face_crop=not args.disable_face_crop,
    )


def _experiment_config_from_args(args):
    return build_experiment_config(
        train_dataset_kind=args.train_dataset,
        test_mode=args.test_mode,
        test_dataset_kind=args.test_dataset,
        dataset_root=args.dataset_root,
        fer_csv=args.fer_csv,
        affectnet_dir=args.affectnet_dir,
        validation_split=args.validation_split,
        fer_balance_target_count=args.balance_target_count,
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
        amp_dtype=args.amp_dtype,
    )


def _prompt_choice(title: str, options: list[tuple[str, str]], default_index: int = 0) -> str:
    print(f"\n{title}")
    for idx, (_, label) in enumerate(options, start=1):
        default_marker = " [padrao]" if idx - 1 == default_index else ""
        print(f"{idx}. {label}{default_marker}")

    while True:
        raw = input("Escolha uma opcao: ").strip()
        if not raw:
            return options[default_index][0]
        if raw.isdigit():
            selection = int(raw) - 1
            if 0 <= selection < len(options):
                return options[selection][0]
        print("Opcao invalida. Tente novamente.")


def _prompt_yes_no(question: str, default: bool = False) -> bool:
    suffix = "[S/n]" if default else "[s/N]"
    while True:
        raw = input(f"{question} {suffix}: ").strip().lower()
        if not raw:
            return default
        if raw in {"s", "sim", "y", "yes"}:
            return True
        if raw in {"n", "nao", "não", "no"}:
            return False
        print("Resposta invalida. Digite s ou n.")


def _build_wizard_configs(args):
    execution_mode = _prompt_choice(
        "Modo de execucao:",
        [("single_experiment", "Rodar um experimento"), ("full_benchmark", "Rodar todas as combinacoes (overnight)")],
        default_index=0,
    )
    if execution_mode == "full_benchmark":
        return _build_benchmark_configs(args)

    train_dataset = _prompt_choice(
        "Dataset de treino:",
        [("fer2013", "FER-2013"), ("affectnet", "AffectNet")],
        default_index=0,
    )
    crop_mode = _prompt_choice(
        "Modo de entrada:",
        [("baseline", "Baseline (sem face crop)"), ("face_crop", "Face crop com MediaPipe/OpenCV")],
        default_index=1,
    )
    use_landmarks = _prompt_yes_no("Ativar landmarks do MediaPipe?", default=False)
    test_mode = _prompt_choice(
        "Modo de avaliacao final:",
        [("same_dataset", "Usar o teste do proprio dataset"), ("cross_dataset", "Usar teste cruzado em outro dataset")],
        default_index=0,
    )

    test_dataset = None
    if test_mode == "cross_dataset":
        test_options = [("fer2013", "FER-2013"), ("affectnet", "AffectNet")]
        test_dataset = _prompt_choice("Dataset de teste:", test_options, default_index=1 if train_dataset == "fer2013" else 0)
        if test_dataset == train_dataset:
            print("Teste cruzado exige um dataset diferente. O teste sera feito no outro dataset automaticamente.")
            test_dataset = "affectnet" if train_dataset == "fer2013" else "fer2013"

    experiment_config = build_experiment_config(
        train_dataset_kind=train_dataset,
        test_mode=test_mode,
        test_dataset_kind=test_dataset,
        dataset_root=args.dataset_root,
        fer_csv=args.fer_csv,
        affectnet_dir=args.affectnet_dir,
        validation_split=args.validation_split,
        fer_balance_target_count=args.balance_target_count,
    )
    emotion_config = EmotionConfig(
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        target_image_size=args.image_size,
        train_split_seed=args.seed,
        use_face_crop=(crop_mode == "face_crop"),
    )
    train_config = _train_config_from_args(args)
    train_config.use_landmarks = use_landmarks
    return experiment_config, emotion_config, train_config


def _build_benchmark_configs(args):
    jobs = []
    for train_dataset in ("fer2013", "affectnet"):
        opposite_dataset = "affectnet" if train_dataset == "fer2013" else "fer2013"
        for use_face_crop in (False, True):
            for use_landmarks in (False, True):
                for test_mode in ("same_dataset", "cross_dataset"):
                    experiment_config = build_experiment_config(
                        train_dataset_kind=train_dataset,
                        test_mode=test_mode,
                        test_dataset_kind=None if test_mode == "same_dataset" else opposite_dataset,
                        dataset_root=args.dataset_root,
                        fer_csv=args.fer_csv,
                        affectnet_dir=args.affectnet_dir,
                        validation_split=args.validation_split,
                        fer_balance_target_count=args.balance_target_count,
                    )
                    emotion_config = EmotionConfig(
                        output_dir=args.output_dir,
                        results_dir=args.results_dir,
                        target_image_size=args.image_size,
                        train_split_seed=args.seed,
                        use_face_crop=use_face_crop,
                    )
                    train_config = _train_config_from_args(args)
                    train_config.use_landmarks = use_landmarks
                    jobs.append(
                        (
                            experiment_config,
                            emotion_config,
                            train_config,
                            describe_experiment(experiment_config, emotion_config.use_face_crop, train_config.use_landmarks),
                        )
                    )
    return jobs


def _run_benchmark_jobs(args, jobs):
    benchmark_dir = args.results_dir / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "created_at": datetime.now().isoformat(),
        "num_jobs": len(jobs),
        "succeeded": 0,
        "failed": 0,
        "runs": [],
    }
    successful_run_dirs: list[Path] = []

    for index, (experiment_config, emotion_config, train_config, description) in enumerate(jobs, start=1):
        print(f"\n[{index}/{len(jobs)}] Iniciando experimento:")
        print(json.dumps(description, indent=2, ensure_ascii=False))

        try:
            result = train_pipeline(experiment_config, emotion_config, train_config)
            summary["succeeded"] += 1
            successful_run_dirs.append(Path(result["run_dir"]))
            summary["runs"].append(
                {
                    **description,
                    "status": "ok",
                    "run_dir": result["run_dir"],
                    "best_checkpoint": result["best_checkpoint"],
                    "test_accuracy": result["test"]["accuracy"],
                    "test_f1": result["test"]["f1"],
                }
            )
        except Exception as exc:
            summary["failed"] += 1
            summary["runs"].append(
                {
                    **description,
                    "status": "error",
                    "error": str(exc),
                }
            )
            print(f"Falha no experimento {index}: {exc}")

    summary_path = benchmark_dir / f"{timestamp}_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    comparison_result = None
    if len(successful_run_dirs) >= 2:
        comparison_result = export_comparison(
            results_dir=args.results_dir,
            run_dirs=successful_run_dirs,
            comparison_name=f"{timestamp}_benchmark",
        )
        summary["comparison"] = comparison_result
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "summary_path": str(summary_path),
        "benchmark": summary,
        "comparison": comparison_result,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        experiment_config = _experiment_config_from_args(args)
        emotion_config = _emotion_config_from_args(args)
        datasets, metadata = prepare_experiment_datasets(experiment_config, emotion_config)
        artifact_dir = emotion_config.output_dir / build_artifact_name(experiment_config, emotion_config)
        save_prepared_splits(datasets, artifact_dir, metadata)
        print(json.dumps({name: summarize_split(datasets[name]) for name in ("train", "val", "test")}, indent=2))
        return

    if args.command == "train":
        result = train_pipeline(_experiment_config_from_args(args), _emotion_config_from_args(args), _train_config_from_args(args))
        print(json.dumps(result["best_metrics"], indent=2))
        print(json.dumps(result["test"], indent=2))
        return

    if args.command == "evaluate":
        metrics = evaluate_checkpoint(
            args.checkpoint,
            _experiment_config_from_args(args),
            _emotion_config_from_args(args),
            _train_config_from_args(args),
        )
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "predict":
        predictor = EmotionPredictor(args.checkpoint, device=args.device)
        print(json.dumps(predictor.predict_image(args.image), indent=2))
        return

    if args.command == "webcam":
        run_webcam(args.checkpoint, camera_index=args.camera_index, device=args.device)
        return

    if args.command == "serve":
        run_server(
            checkpoint_path=args.checkpoint,
            host=args.host,
            port=args.port,
            device=args.device,
            debug=args.debug,
        )
        return

    if args.command == "benchmark":
        result = _run_benchmark_jobs(args, _build_benchmark_configs(args))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "wizard":
        wizard_selection = _build_wizard_configs(args)
        if isinstance(wizard_selection, list):
            print("\nResumo do benchmark:")
            print(f"Total de experimentos: {len(wizard_selection)}")
            if not _prompt_yes_no("Deseja iniciar o benchmark completo agora?", default=True):
                print("Benchmark cancelado.")
                return

            result = _run_benchmark_jobs(args, wizard_selection)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return

        experiment_config, emotion_config, train_config = wizard_selection
        summary = describe_experiment(experiment_config, emotion_config.use_face_crop, train_config.use_landmarks)
        print("\nResumo do experimento:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if not _prompt_yes_no("Deseja iniciar o treinamento com essa configuração?", default=True):
            print("Treinamento cancelado.")
            return

        result = train_pipeline(experiment_config, emotion_config, train_config)
        print(json.dumps(result["best_metrics"], indent=2))
        print(json.dumps(result["test"], indent=2))
        return

    if args.command == "compare":
        result = export_comparison(
            results_dir=args.results_dir,
            run_dirs=args.run_dirs,
            latest=args.latest,
            comparison_name=args.name,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return


if __name__ == "__main__":
    main()
