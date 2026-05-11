"""Streamlit dashboard helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import METRICS_DIR, PREDICTIONS_DIR, PROJECT_ROOT, SPLIT_FILES
from .data_loader import label_distribution, read_split, split_overview


def available_prediction_files() -> dict[str, Path]:
    """Return available prediction files keyed by experiment name."""
    files = {}
    for path in sorted(PREDICTIONS_DIR.glob("*_predictions.csv")):
        files[path.name.replace("_predictions.csv", "")] = path
    return files


def load_data_overview() -> pd.DataFrame:
    """Load cached data overview or compute it from local CSV files."""
    path = METRICS_DIR / "data_overview.csv"
    if path.exists():
        return pd.read_csv(path)
    splits = {split: read_split(split) for split in SPLIT_FILES}
    return split_overview(splits)


def load_label_distribution(split: str) -> pd.DataFrame:
    """Load label distribution for one split."""
    df = read_split(split)
    return label_distribution(df)


def load_results_summary() -> pd.DataFrame | None:
    """Load aggregate model metrics when available."""
    path = METRICS_DIR / "results_summary.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_all_results_summary() -> pd.DataFrame | None:
    """Load baseline and Kaggle DistilBERT metrics in one normalized table."""
    frames = []
    baseline_path = METRICS_DIR / "results_summary.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        frames.append(baseline)

    kaggle_path = (
        PROJECT_ROOT
        / "kaggle_outputs"
        / "distilbert"
        / "review_to_rating_distilbert"
        / "metrics"
        / "distilbert_results_summary.csv"
    )
    if kaggle_path.exists():
        distilbert = pd.read_csv(kaggle_path)
        distilbert = distilbert.assign(model="distilbert", samples=distilbert["test_samples"])
        keep_columns = ["task", "model", "accuracy", "precision_macro", "recall_macro", "macro_f1", "samples"]
        frames.append(distilbert[keep_columns])

    if not frames:
        return None
    results = pd.concat(frames, ignore_index=True)
    return results.sort_values(["task", "model"]).reset_index(drop=True)


def load_prediction_preview(experiment_name: str, nrows: int = 1000) -> pd.DataFrame:
    """Load a preview of one prediction file."""
    path = PREDICTIONS_DIR / f"{experiment_name}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, nrows=nrows)
