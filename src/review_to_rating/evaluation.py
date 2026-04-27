"""Evaluation helpers for prediction files."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)


REQUIRED_PREDICTION_COLUMNS = ["id", "text", "true_label", "pred_label"]


def load_predictions(path: Path) -> pd.DataFrame:
    """Load and validate a prediction CSV file."""
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def evaluate_predictions(
    df: pd.DataFrame,
    task: str,
    model_name: str,
    labels: Sequence[str] | Sequence[int] | None = None,
) -> tuple[dict[str, object], str]:
    """Compute summary metrics and a text classification report."""
    y_true = df["true_label"]
    y_pred = df["pred_label"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    summary = {
        "task": task,
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision,
        "recall_macro": recall,
        "macro_f1": f1,
        "samples": len(df),
    }
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    return summary, report


def save_evaluation_outputs(
    summaries: list[dict[str, object]],
    reports: dict[str, str],
    summary_path: Path,
    reports_dir: Path,
) -> None:
    """Save the aggregate metrics table and individual reports."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).sort_values(["task", "model"]).to_csv(summary_path, index=False)
    for experiment_name, report in reports.items():
        (reports_dir / f"{experiment_name}_classification_report.txt").write_text(report, encoding="utf-8")
