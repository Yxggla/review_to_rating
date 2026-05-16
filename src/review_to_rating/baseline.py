"""TF-IDF + Logistic Regression baseline training and prediction."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import ID_COLUMN, TEXT_COLUMN
from .labels import get_target_column


def build_baseline_pipeline(max_features: int = 50000) -> Pipeline:
    """Create the TF-IDF + Logistic Regression baseline pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=max_features,
                    min_df=2,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    n_jobs=1,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train_baseline(
    train_df: pd.DataFrame,
    task: str,
    max_features: int = 50000,
) -> Pipeline:
    """Train a baseline model for sentiment or rating prediction."""
    target_column = get_target_column(task)
    model = build_baseline_pipeline(max_features=max_features)
    model.fit(train_df[TEXT_COLUMN].fillna("").astype(str), train_df[target_column])
    return model


def predict_dataframe(model: Pipeline, df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Return a normalized prediction dataframe."""
    target_column = get_target_column(task)
    predictions = model.predict(df[TEXT_COLUMN].fillna("").astype(str))
    return pd.DataFrame(
        {
            "id": df[ID_COLUMN],
            "text": df[TEXT_COLUMN],
            "true_label": df[target_column],
            "pred_label": predictions,
        }
    )


def save_model(model: Pipeline, path: Path) -> None:
    """Persist a scikit-learn model with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Pipeline:
    """Load a scikit-learn model saved with joblib."""
    return joblib.load(path)
