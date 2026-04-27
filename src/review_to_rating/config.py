"""Project paths and shared configuration."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "amazon_reviews_multi_en"
PROCESSED_DATA_DIR = DATA_DIR / "processed_3class"
RAW_PARQUET_DIR = DATA_DIR / "raw_parquet"
SUMMARY_PATH = DATA_DIR / "summary.json"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
METRICS_DIR = OUTPUTS_DIR / "metrics"
REPORTS_DIR = METRICS_DIR / "classification_reports"
ERROR_ANALYSIS_DIR = METRICS_DIR / "error_analysis"
FIGURES_DIR = OUTPUTS_DIR / "figures"
DATA_DISTRIBUTION_FIGURES_DIR = FIGURES_DIR / "data_distribution"
CONFUSION_MATRIX_FIGURES_DIR = FIGURES_DIR / "confusion_matrices"
LOGS_DIR = OUTPUTS_DIR / "logs"

MODELS_DIR = PROJECT_ROOT / "models"
DISTILBERT_SENTIMENT_DIR = MODELS_DIR / "distilbert_sentiment"
DISTILBERT_RATING_DIR = MODELS_DIR / "distilbert_rating"

RANDOM_SEED = 42
TEXT_COLUMN = "text"
ID_COLUMN = "id"
SENTIMENT_COLUMN = "label_3class"
RATING_COLUMN = "stars"
RAW_LABEL_COLUMN = "raw_label_5way"

SPLIT_FILES = {
    "train": PROCESSED_DATA_DIR / "train_3class.csv",
    "validation": PROCESSED_DATA_DIR / "validation_3class.csv",
    "test": PROCESSED_DATA_DIR / "test_3class.csv",
}


def ensure_project_dirs() -> None:
    """Create generated-output directories if they do not exist."""
    for path in [
        PREDICTIONS_DIR,
        REPORTS_DIR,
        ERROR_ANALYSIS_DIR,
        DATA_DISTRIBUTION_FIGURES_DIR,
        CONFUSION_MATRIX_FIGURES_DIR,
        LOGS_DIR,
        DISTILBERT_SENTIMENT_DIR,
        DISTILBERT_RATING_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
