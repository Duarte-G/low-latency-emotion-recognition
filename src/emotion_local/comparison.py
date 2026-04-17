from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv
import json
import warnings


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


_EMOTION_LABELS = ["Angry", "Happy", "Sad", "Neutral"]


def _short_label(row: dict[str, object]) -> str:
    train_ds = str(row.get("train_dataset", "?"))[:3].upper()
    test_ds = str(row.get("test_dataset", "?"))[:3].upper()
    crop = "crop" if row.get("face_crop") else "nocrop"
    lm = "+lm" if row.get("landmarks") else ""
    mode = "cross" if row.get("test_mode") == "cross_dataset" else "same"
    return f"{train_ds}→{test_ds}\n{crop}{lm}\n({mode})"


def save_comparison_plots(rows: list[dict[str, object]], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
    except ImportError:
        warnings.warn("matplotlib/seaborn nao encontrado — plots nao gerados.")
        return

    if not rows:
        return

    labels = [_short_label(r) for r in rows]
    x = np.arange(len(rows))

    # --- Plot 1: Accuracy e F1 por configuracao ---
    acc_vals = [float(r.get("test_accuracy") or 0) for r in rows]
    f1_vals = [float(r.get("test_f1") or 0) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(rows) * 1.5), 6))
    fig.suptitle("Comparação de Configurações — Conjunto de Teste", fontsize=13, fontweight="bold")

    axes[0].bar(x, acc_vals, color="steelblue", width=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].set_ylabel("Acurácia (%)")
    axes[0].set_title("Test Accuracy")
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(acc_vals):
        axes[0].text(i, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    axes[1].bar(x, f1_vals, color="darkorange", width=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_ylabel("F1 Weighted")
    axes[1].set_title("Test F1 (Weighted)")
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(f1_vals):
        axes[1].text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(output_dir / "accuracy_f1_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: F1 por classe para cada configuracao ---
    per_class_keys = {lbl: f"{lbl.lower()}_f1_score" for lbl in _EMOTION_LABELS}
    available_classes = [lbl for lbl in _EMOTION_LABELS if any(per_class_keys[lbl] in r for r in rows)]

    if available_classes:
        n_classes = len(available_classes)
        bar_w = 0.8 / max(n_classes, 1)
        fig2, ax2 = plt.subplots(figsize=(max(12, len(rows) * 1.5), 6))
        colors = sns.color_palette("Set2", n_classes)

        for ci, cls in enumerate(available_classes):
            key = per_class_keys[cls]
            cls_vals = [float(r.get(key) or 0) for r in rows]
            offsets = x - 0.4 + bar_w * ci + bar_w / 2
            ax2.bar(offsets, cls_vals, width=bar_w * 0.9, label=cls, color=colors[ci])

        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=7)
        ax2.set_ylabel("F1-Score")
        ax2.set_title("F1 por Classe (per-class) — Comparação entre Configurações", fontweight="bold")
        ax2.set_ylim(0, 1)
        ax2.legend(loc="upper right")
        plt.tight_layout()
        fig2.savefig(output_dir / "per_class_f1_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)

    # --- Plot 3: Heatmap — face_crop × landmarks para cada dataset de treino ---
    for train_ds in ("fer2013", "affectnet"):
        ds_rows = [r for r in rows if r.get("train_dataset") == train_ds and r.get("test_mode") == "same_dataset"]
        if len(ds_rows) < 2:
            continue

        heatmap_data = np.full((2, 2), np.nan)
        for r in ds_rows:
            ci = 1 if r.get("face_crop") else 0
            ri = 1 if r.get("landmarks") else 0
            heatmap_data[ri, ci] = float(r.get("test_accuracy") or 0)

        fig3, ax3 = plt.subplots(figsize=(6, 5))
        mask = np.isnan(heatmap_data)
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            mask=mask,
            xticklabels=["Sem Face Crop", "Com Face Crop"],
            yticklabels=["Sem Landmarks", "Com Landmarks"],
            ax=ax3,
            vmin=0,
            vmax=100,
            linewidths=0.5,
        )
        ax3.set_title(f"Acurácia (%) — {train_ds.upper()} (same-dataset)", fontweight="bold")
        plt.tight_layout()
        fig3.savefig(output_dir / f"heatmap_{train_ds}.png", dpi=150, bbox_inches="tight")
        plt.close(fig3)


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
    save_comparison_plots(rows, output_dir)
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
