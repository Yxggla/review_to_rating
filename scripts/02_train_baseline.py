#!/usr/bin/env python3
"""Train TF-IDF + Logistic Regression baselines for both tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_to_rating.baseline import predict_dataframe, save_model, train_baseline
from review_to_rating.config import MODELS_DIR, PREDICTIONS_DIR, ensure_project_dirs
from review_to_rating.data_loader import read_split, sample_dataframe
from review_to_rating.labels import get_target_column


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["sentiment", "rating", "both"], default="both")
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--max-train-samples", type=int, default=None)
    return parser.parse_args()


def run_task(task: str, max_features: int, max_train_samples: int | None) -> None:
    train_df = sample_dataframe(read_split("train"), n=max_train_samples, stratify_column=get_target_column(task))
    test_df = read_split("test")
    model = train_baseline(train_df, task=task, max_features=max_features)
    predictions = predict_dataframe(model, test_df, task=task)

    output_path = PREDICTIONS_DIR / f"baseline_{task}_predictions.csv"
    model_path = MODELS_DIR / f"baseline_{task}.joblib"
    predictions.to_csv(output_path, index=False)
    save_model(model, model_path)
    print(f"Saved baseline {task} predictions: {output_path}")
    print(f"Saved baseline {task} model: {model_path}")


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    tasks = ["sentiment", "rating"] if args.task == "both" else [args.task]
    for task in tasks:
        run_task(task, args.max_features, args.max_train_samples)


if __name__ == "__main__":
    main()
