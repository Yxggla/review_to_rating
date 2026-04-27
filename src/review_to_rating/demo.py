"""Demo inference utilities."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .config import ID_COLUMN, TEXT_COLUMN


def build_demo_dataframe(text: str) -> pd.DataFrame:
    """Build a one-row dataframe compatible with both task predictors."""
    return pd.DataFrame(
        {
            ID_COLUMN: ["demo_0"],
            TEXT_COLUMN: [text],
            "label_3class": ["neutral"],
            "stars": [3],
        }
    )


def predict_review_distilbert(text: str, sentiment_model_dir: Path, rating_model_dir: Path) -> dict[str, object]:
    """Predict sentiment and rating using saved DistilBERT models."""
    from .distilbert_model import predict_distilbert

    df = build_demo_dataframe(text)
    sentiment = predict_distilbert(df, "sentiment", sentiment_model_dir).iloc[0]["pred_label"]
    rating = predict_distilbert(df, "rating", rating_model_dir).iloc[0]["pred_label"]
    return {"backend": "distilbert", "sentiment": sentiment, "rating": int(rating)}


def predict_review_baseline(text: str, sentiment_model_path: Path, rating_model_path: Path) -> dict[str, object]:
    """Predict sentiment and rating using saved scikit-learn baseline models."""
    df = build_demo_dataframe(text)
    sentiment_model = joblib.load(sentiment_model_path)
    rating_model = joblib.load(rating_model_path)
    sentiment = sentiment_model.predict(df[TEXT_COLUMN])[0]
    rating = rating_model.predict(df[TEXT_COLUMN])[0]
    return {"backend": "baseline", "sentiment": sentiment, "rating": int(rating)}


def predict_review(
    text: str,
    sentiment_model_dir: Path,
    rating_model_dir: Path,
    backend: str = "auto",
    baseline_sentiment_model_path: Path | None = None,
    baseline_rating_model_path: Path | None = None,
) -> dict[str, object]:
    """Predict sentiment and rating with DistilBERT or baseline fallback."""
    distilbert_ready = (sentiment_model_dir / "config.json").exists() and (rating_model_dir / "config.json").exists()
    baseline_ready = (
        baseline_sentiment_model_path is not None
        and baseline_rating_model_path is not None
        and baseline_sentiment_model_path.exists()
        and baseline_rating_model_path.exists()
    )

    if backend in ("auto", "distilbert") and distilbert_ready:
        return predict_review_distilbert(text, sentiment_model_dir, rating_model_dir)
    if backend in ("auto", "baseline") and baseline_ready:
        return predict_review_baseline(text, baseline_sentiment_model_path, baseline_rating_model_path)

    if backend == "distilbert":
        raise FileNotFoundError("DistilBERT model files are not available yet.")
    if backend == "baseline":
        raise FileNotFoundError("Baseline model files are not available yet.")
    raise FileNotFoundError("No saved model is available. Train baseline or DistilBERT models first.")
