from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv
import json


def _parse_run_timestamp(run_dir: Path) -> str:
    prefix = run_dir.name[:15]
    try:
        dt = datetime.strptime(prefix, "%Y%m%d_%H%M%S")
        return dt.isoformat()
    except ValueError:
        return datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat()


def discover_completed_runs(results_dir: Path) -> list[Path]:
    if not results_dir.exists():
        return []

    run_dirs: list[Path] = []
    for child in results_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name == "comparisons":
            continue
        if (child / "training_results.json").exists() and (child / "run_metadata.json").exists():
            run_dirs.append(child)

    run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return run_dirs


def _extract_class_metrics(report: dict[str, object]) -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in report.items():
        if not isinstance(value, dict):
            continue
        prefix = key.lower().replace(" ", "_")
        for metric_name in ("precision", "recall", "f1-score", "support"):
            if metric_name in value:
                normalized_name = metric_name.replace("-", "_")
                flat[f"{prefix}_{normalized_name}"] = value[metric_name]
    return flat


def load_run_comparison_row(run_dir: Path) -> dict[str, object]:
    training_results = json.loads((run_dir / "training_results.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))

    experiment = training_results.get("experiment", {})
    train_cfg = metadata.get("train_config", {})
    emotion_cfg = metadata.get("emotion_config", {})
    test_metrics = training_results.get("test", {})
    report = test_metrics.get("classification_report", {})

    row = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "timestamp": _parse_run_timestamp(run_dir),
        "train_dataset": experiment.get("train_dataset"),
        "test_dataset": experiment.get("test_dataset"),
        "test_mode": experiment.get("test_mode"),
        "face_crop": experiment.get("face_crop"),
        "landmarks": experiment.get("landmarks"),
        "device": training_results.get("device"),
        "batch_size": train_cfg.get("batch_size"),
        "epochs": train_cfg.get("num_epochs"),
        "learning_rate": train_cfg.get("learning_rate"),
        "weight_decay": train_cfg.get("weight_decay"),
        "dropout": train_cfg.get("dropout"),
        "image_size": emotion_cfg.get("target_image_size"),
        "seed": emotion_cfg.get("train_split_seed"),
        "best_epoch": training_results.get("best_metrics", {}).get("epoch"),
        "val_accuracy": training_results.get("best_metrics", {}).get("val_accuracy"),
        "val_f1": training_results.get("best_metrics", {}).get("val_f1"),
        "test_loss": test_metrics.get("loss"),
        "test_accuracy": test_metrics.get("accuracy"),
        "test_f1": test_metrics.get("f1"),
        "best_checkpoint": training_results.get("best_checkpoint"),
        "artifact_dir": training_results.get("artifact_dir"),
    }
    row.update(_extract_class_metrics(report))
    return row


def select_runs(results_dir: Path, run_dirs: list[Path] | None = None, latest: int | None = None) -> list[Path]:
    available = discover_completed_runs(results_dir)
    if run_dirs:
        selected = [path.resolve() for path in run_dirs]
    elif latest is not None:
        selected = available[:latest]
    else:
        selected = available

    existing = [path for path in selected if (path / "training_results.json").exists() and (path / "run_metadata.json").exists()]
    existing.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return existing


def _write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_rankings(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    sortable = [row for row in rows if row.get("test_accuracy") is not None]
    sortable.sort(key=lambda row: (row.get("test_accuracy", -1), row.get("test_f1", -1)), reverse=True)
    rankings: list[dict[str, object]] = []
    for idx, row in enumerate(sortable, start=1):
        rankings.append(
            {
                "position": idx,
                "run_name": row.get("run_name"),
                "train_dataset": row.get("train_dataset"),
                "test_dataset": row.get("test_dataset"),
                "test_mode": row.get("test_mode"),
                "face_crop": row.get("face_crop"),
                "landmarks": row.get("landmarks"),
                "test_accuracy": row.get("test_accuracy"),
                "test_f1": row.get("test_f1"),
                "val_accuracy": row.get("val_accuracy"),
            }
        )
    return rankings


def _write_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    rankings = _build_rankings(rows)
    lines = [
        f"Total de execucoes comparadas: {len(rows)}",
        "",
        "Ranking por test_accuracy:",
    ]
    if not rankings:
        lines.append("Nenhuma execucao com metrica de teste disponivel.")
    else:
        for item in rankings:
            lines.append(
                f"{item['position']}. {item['run_name']} | train={item['train_dataset']} | "
                f"test={item['test_dataset']} | mode={item['test_mode']} | "
                f"face_crop={item['face_crop']} | landmarks={item['landmarks']} | "
                f"test_accuracy={item['test_accuracy']:.4f} | test_f1={item['test_f1']:.4f}"
            )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def export_comparison(
    results_dir: Path,
    run_dirs: list[Path] | None = None,
    latest: int | None = None,
    comparison_name: str | None = None,
) -> dict[str, object]:
    selected_runs = select_runs(results_dir=results_dir, run_dirs=run_dirs, latest=latest)
    if len(selected_runs) < 2:
        raise ValueError("Sao necessarias pelo menos duas execucoes completas para gerar a comparacao.")

    rows = [load_run_comparison_row(run_dir) for run_dir in selected_runs]
    comparison_root = results_dir / "comparisons"
    comparison_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = (comparison_name or "comparison").strip().replace(" ", "_")
    output_dir = comparison_root / f"{timestamp}_{safe_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "comparison.csv"
    json_path = output_dir / "comparison.json"
    summary_path = output_dir / "summary.txt"

    _write_csv(rows, csv_path)
    json_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(),
                "results_dir": str(results_dir),
                "runs": rows,
                "ranking": _build_rankings(rows),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_summary(rows, summary_path)

    return {
        "output_dir": str(output_dir),
        "comparison_csv": str(csv_path),
        "comparison_json": str(json_path),
        "summary_txt": str(summary_path),
        "num_runs": len(rows),
        "runs": [str(path) for path in selected_runs],
    }
