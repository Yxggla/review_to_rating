#!/usr/bin/env python3
"""Check dataset files and generate basic data figures."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_to_rating.config import DATA_DISTRIBUTION_FIGURES_DIR, METRICS_DIR, ensure_project_dirs
from review_to_rating.data_loader import (
    label_distribution,
    load_all_splits,
    load_summary,
    split_overview,
    validate_all_splits,
)
from review_to_rating.visualization import save_label_distribution_plots, save_text_length_plot


def main() -> None:
    ensure_project_dirs()
    summary = load_summary()
    splits = load_all_splits()
    validation_results = validate_all_splits(splits)

    print("Dataset source:", summary.get("source_note", "unknown"))
    print("\nValidation:")
    for result in validation_results:
        status = "OK" if result.is_valid else "CHECK"
        print(
            f"- {result.split}: {status}, rows={result.rows}, "
            f"missing={result.missing_columns}, empty_text={result.empty_text_count}, "
            f"invalid_sentiment={result.invalid_sentiment_count}, invalid_rating={result.invalid_rating_count}"
        )

    overview = split_overview(splits)
    overview_path = METRICS_DIR / "data_overview.csv"
    overview.to_csv(overview_path, index=False)

    distribution_frames = []
    for split, df in splits.items():
        frame = label_distribution(df)
        frame.insert(0, "split", split)
        distribution_frames.append(frame)
    distribution_path = METRICS_DIR / "label_distribution.csv"
    distribution = pd.concat(distribution_frames, ignore_index=True)
    distribution.to_csv(distribution_path, index=False)

    save_label_distribution_plots(splits, DATA_DISTRIBUTION_FIGURES_DIR)
    save_text_length_plot(splits, DATA_DISTRIBUTION_FIGURES_DIR)

    print(f"\nSaved overview: {overview_path}")
    print(f"Saved label distribution: {distribution_path}")
    print(f"Saved figures: {DATA_DISTRIBUTION_FIGURES_DIR}")


if __name__ == "__main__":
    main()
