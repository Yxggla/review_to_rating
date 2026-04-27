#!/usr/bin/env python3
"""Extract success and error cases from available prediction files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_to_rating.config import ERROR_ANALYSIS_DIR, PREDICTIONS_DIR, ensure_project_dirs
from review_to_rating.evaluation import load_predictions


EXPERIMENTS = [
    "baseline_sentiment",
    "baseline_rating",
    "distilbert_sentiment",
    "distilbert_rating",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-per-file", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    ERROR_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    for experiment_name in EXPERIMENTS:
        path = PREDICTIONS_DIR / f"{experiment_name}_predictions.csv"
        if not path.exists():
            print(f"Skipping missing prediction file: {path}")
            continue
        df = load_predictions(path)
        correct = df[df["true_label"].astype(str) == df["pred_label"].astype(str)].head(args.cases_per_file)
        incorrect = df[df["true_label"].astype(str) != df["pred_label"].astype(str)].head(args.cases_per_file)
        correct.to_csv(ERROR_ANALYSIS_DIR / f"{experiment_name}_success_cases.csv", index=False)
        incorrect.to_csv(ERROR_ANALYSIS_DIR / f"{experiment_name}_error_cases.csv", index=False)
        print(f"Saved cases for {experiment_name}")


if __name__ == "__main__":
    main()
