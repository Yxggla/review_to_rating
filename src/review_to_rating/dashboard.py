"""Streamlit dashboard helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from .config import (
    DATA_DISTRIBUTION_FIGURES_DIR,
    FIGURES_DIR,
    KAGGLE_DISTILBERT_METRICS_DIR,
    KAGGLE_DISTILBERT_PREDICTIONS_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    SPLIT_FILES,
)
from .data_loader import label_distribution, read_split, split_overview
from .labels import SENTIMENT_LABELS


OVERVIEW_COLUMNS = ["split", "rows", "mean_words", "median_words", "max_words", "min_words"]


def _prediction_file_registry() -> dict[str, Path]:
    """Return available prediction files from local and Kaggle output folders."""
    files: dict[str, Path] = {}
    for directory in [PREDICTIONS_DIR, KAGGLE_DISTILBERT_PREDICTIONS_DIR]:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*_predictions.csv")):
            files[path.name.replace("_predictions.csv", "")] = path
    return files


def _normalize_overview_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map legacy overview columns into one stable schema for the dashboard."""
    normalized = df.copy()
    if "avg_text_length" in normalized.columns and "mean_words" not in normalized.columns:
        normalized["mean_words"] = normalized["avg_text_length"]
    for column in OVERVIEW_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    return normalized[OVERVIEW_COLUMNS]


def available_prediction_files() -> dict[str, Path]:
    """Return available prediction files keyed by experiment name."""
    return _prediction_file_registry()


def load_data_overview() -> pd.DataFrame:
    """Load cached data overview or compute it from local CSV files."""
    path = METRICS_DIR / "data_overview.csv"
    if path.exists():
        return _normalize_overview_columns(pd.read_csv(path))
    splits = {split: read_split(split) for split in SPLIT_FILES}
    return _normalize_overview_columns(split_overview(splits))


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

    kaggle_path = KAGGLE_DISTILBERT_METRICS_DIR / "distilbert_results_summary.csv"
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
    files = _prediction_file_registry()
    if experiment_name not in files:
        raise FileNotFoundError(experiment_name)
    path = files[experiment_name]
    return pd.read_csv(path, nrows=nrows)


def get_wordcloud_paths(split: str) -> dict[str, Path]:
    """Return paths for sentiment wordcloud images if they exist."""
    wordcloud_dir = FIGURES_DIR / "wordclouds"
    return {
        "positive": wordcloud_dir / f"{split}_positive_wordcloud.png",
        "negative": wordcloud_dir / f"{split}_negative_wordcloud.png",
    }


def get_text_length_plot_path() -> Path:
    """Return path for text length distribution plot."""
    return DATA_DISTRIBUTION_FIGURES_DIR / "text_length_distribution.png"
