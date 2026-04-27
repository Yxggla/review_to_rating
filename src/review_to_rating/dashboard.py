"""Streamlit dashboard helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import METRICS_DIR, PREDICTIONS_DIR, SPLIT_FILES
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


def load_prediction_preview(experiment_name: str, nrows: int = 1000) -> pd.DataFrame:
    """Load a preview of one prediction file."""
    path = PREDICTIONS_DIR / f"{experiment_name}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, nrows=nrows)
