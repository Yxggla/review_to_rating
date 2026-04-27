#!/usr/bin/env python3
"""Run demo inference with saved DistilBERT models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_to_rating.config import DISTILBERT_RATING_DIR, DISTILBERT_SENTIMENT_DIR
from review_to_rating.config import MODELS_DIR
from review_to_rating.demo import predict_review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="English Amazon review text.")
    parser.add_argument("--backend", choices=["auto", "baseline", "distilbert"], default="auto")
    parser.add_argument("--sentiment-model-dir", type=Path, default=DISTILBERT_SENTIMENT_DIR)
    parser.add_argument("--rating-model-dir", type=Path, default=DISTILBERT_RATING_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_review(
        text=args.text,
        sentiment_model_dir=args.sentiment_model_dir,
        rating_model_dir=args.rating_model_dir,
        backend=args.backend,
        baseline_sentiment_model_path=MODELS_DIR / "baseline_sentiment.joblib",
        baseline_rating_model_path=MODELS_DIR / "baseline_rating.joblib",
    )
    print(f"Backend: {result['backend']}")
    print(f"Sentiment prediction: {result['sentiment']}")
    print(f"Rating prediction: {result['rating']} stars")


if __name__ == "__main__":
    main()
