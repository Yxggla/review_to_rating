#!/usr/bin/env python3
"""Evaluate available prediction files and generate confusion matrices."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_to_rating.config import (
    CONFUSION_MATRIX_FIGURES_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    ensure_project_dirs,
)
from review_to_rating.evaluation import evaluate_predictions, load_predictions, save_evaluation_outputs
from review_to_rating.labels import RATING_LABELS, SENTIMENT_LABELS
from review_to_rating.visualization import save_confusion_matrix


EXPERIMENTS = {
    "baseline_sentiment": ("sentiment", "baseline", SENTIMENT_LABELS),
    "baseline_rating": ("rating", "baseline", RATING_LABELS),
    "distilbert_sentiment": ("sentiment", "distilbert", SENTIMENT_LABELS),
    "distilbert_rating": ("rating", "distilbert", RATING_LABELS),
}


def main() -> None:
    ensure_project_dirs()
    summaries = []
    reports = {}

    for experiment_name, (task, model_name, labels) in EXPERIMENTS.items():
        path = PREDICTIONS_DIR / f"{experiment_name}_predictions.csv"
        if not path.exists():
            print(f"Skipping missing prediction file: {path}")
            continue
        df = load_predictions(path)
        if task == "rating":
            df["true_label"] = df["true_label"].astype(int)
            df["pred_label"] = df["pred_label"].astype(int)
        summary, report = evaluate_predictions(df, task=task, model_name=model_name, labels=labels)
        summaries.append(summary)
        reports[experiment_name] = report
        save_confusion_matrix(
            df,
            labels=labels,
            title=f"{model_name.title()} {task.title()} Confusion Matrix",
            output_path=CONFUSION_MATRIX_FIGURES_DIR / f"{experiment_name}_confusion_matrix.png",
        )

    if not summaries:
        raise SystemExit("No prediction files found to evaluate.")

    summary_path = METRICS_DIR / "results_summary.csv"
    save_evaluation_outputs(summaries, reports, summary_path, REPORTS_DIR)
    print(f"Saved metrics summary: {summary_path}")
    print(f"Saved classification reports: {REPORTS_DIR}")
    print(f"Saved confusion matrices: {CONFUSION_MATRIX_FIGURES_DIR}")


if __name__ == "__main__":
    main()
