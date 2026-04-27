#!/usr/bin/env python3
"""Fine-tune DistilBERT models for sentiment and rating tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_to_rating.config import (
    DISTILBERT_RATING_DIR,
    DISTILBERT_SENTIMENT_DIR,
    PREDICTIONS_DIR,
    ensure_project_dirs,
)
from review_to_rating.data_loader import read_split
from review_to_rating.data_loader import sample_dataframe
from review_to_rating.distilbert_model import predict_distilbert, train_distilbert
from review_to_rating.labels import get_target_column


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["sentiment", "rating", "both"], default="both")
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    return parser.parse_args()


def model_dir_for_task(task: str) -> Path:
    if task == "sentiment":
        return DISTILBERT_SENTIMENT_DIR
    if task == "rating":
        return DISTILBERT_RATING_DIR
    raise ValueError(task)


def run_task(args: argparse.Namespace, task: str) -> None:
    model_dir = model_dir_for_task(task)
    train_df = sample_dataframe(read_split("train"), n=args.max_train_samples, stratify_column=get_target_column(task))
    validation_df = sample_dataframe(
        read_split("validation"),
        n=args.max_validation_samples,
        stratify_column=get_target_column(task),
    )
    test_df = read_split("test")

    train_distilbert(
        train_df=train_df,
        validation_df=validation_df,
        task=task,
        model_dir=model_dir,
        base_model_name=args.base_model,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    predictions = predict_distilbert(test_df, task=task, model_dir=model_dir, max_length=args.max_length)
    output_path = PREDICTIONS_DIR / f"distilbert_{task}_predictions.csv"
    predictions.to_csv(output_path, index=False)
    print(f"Saved DistilBERT {task} predictions: {output_path}")


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    tasks = ["sentiment", "rating"] if args.task == "both" else [args.task]
    for task in tasks:
        run_task(args, task)


if __name__ == "__main__":
    main()
