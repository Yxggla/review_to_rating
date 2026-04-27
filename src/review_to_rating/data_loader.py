"""Data loading, validation, and dataset statistics helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import (
    ID_COLUMN,
    RAW_LABEL_COLUMN,
    RATING_COLUMN,
    SENTIMENT_COLUMN,
    SPLIT_FILES,
    SUMMARY_PATH,
    TEXT_COLUMN,
)
from .labels import RATING_LABELS, SENTIMENT_LABELS

REQUIRED_COLUMNS = [ID_COLUMN, RAW_LABEL_COLUMN, RATING_COLUMN, SENTIMENT_COLUMN, TEXT_COLUMN]


@dataclass(frozen=True)
class SplitValidationResult:
    split: str
    rows: int
    missing_columns: list[str]
    empty_text_count: int
    invalid_sentiment_count: int
    invalid_rating_count: int

    @property
    def is_valid(self) -> bool:
        return (
            not self.missing_columns
            and self.empty_text_count == 0
            and self.invalid_sentiment_count == 0
            and self.invalid_rating_count == 0
        )


def read_split(split: str, nrows: int | None = None) -> pd.DataFrame:
    """Read one processed CSV split."""
    if split not in SPLIT_FILES:
        raise ValueError(f"split must be one of {sorted(SPLIT_FILES)}, got {split!r}")
    path = SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return pd.read_csv(path, nrows=nrows)


def sample_dataframe(
    df: pd.DataFrame,
    n: int | None,
    stratify_column: str | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a stable optional sample, stratified when requested."""
    if n is None or n >= len(df):
        return df
    if stratify_column is None:
        return df.sample(n=n, random_state=random_state).reset_index(drop=True)

    pieces = []
    for _, group in df.groupby(stratify_column, sort=False):
        group_n = max(1, round(n * len(group) / len(df)))
        pieces.append(group.sample(n=min(group_n, len(group)), random_state=random_state))
    sampled = pd.concat(pieces, ignore_index=False)
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=random_state)
    elif len(sampled) < n:
        remaining = df.drop(index=sampled.index, errors="ignore")
        if not remaining.empty:
            sampled = pd.concat(
                [sampled, remaining.sample(n=min(n - len(sampled), len(remaining)), random_state=random_state)],
                ignore_index=True,
            )
    return sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)


def load_all_splits(nrows: int | None = None) -> dict[str, pd.DataFrame]:
    """Read train, validation, and test splits."""
    return {split: read_split(split, nrows=nrows) for split in SPLIT_FILES}


def load_summary(path: Path = SUMMARY_PATH) -> dict:
    """Load the dataset summary JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_dataframe(df: pd.DataFrame, split: str) -> SplitValidationResult:
    """Validate required columns, empty text, and label values."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        return SplitValidationResult(split, len(df), missing_columns, 0, 0, 0)

    text_series = df[TEXT_COLUMN].fillna("").astype(str)
    empty_text_count = int((text_series.str.strip() == "").sum())
    invalid_sentiment_count = int((~df[SENTIMENT_COLUMN].isin(SENTIMENT_LABELS)).sum())
    invalid_rating_count = int((~df[RATING_COLUMN].isin(RATING_LABELS)).sum())

    return SplitValidationResult(
        split=split,
        rows=len(df),
        missing_columns=missing_columns,
        empty_text_count=empty_text_count,
        invalid_sentiment_count=invalid_sentiment_count,
        invalid_rating_count=invalid_rating_count,
    )


def validate_all_splits(splits: dict[str, pd.DataFrame]) -> list[SplitValidationResult]:
    """Validate all loaded splits."""
    return [validate_dataframe(df, split) for split, df in splits.items()]


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return sentiment and rating label counts for one dataframe."""
    sentiment_counts = df[SENTIMENT_COLUMN].value_counts().reindex(SENTIMENT_LABELS, fill_value=0)
    rating_counts = df[RATING_COLUMN].value_counts().reindex(RATING_LABELS, fill_value=0)
    rows = [{"label_type": "sentiment", "label": label, "count": int(count)} for label, count in sentiment_counts.items()]
    rows += [{"label_type": "rating", "label": str(label), "count": int(count)} for label, count in rating_counts.items()]
    return pd.DataFrame(rows)


def text_length_stats(df: pd.DataFrame) -> dict[str, float]:
    """Compute simple text length statistics by word count."""
    word_counts = df[TEXT_COLUMN].fillna("").astype(str).str.split().str.len()
    return {
        "mean_words": float(word_counts.mean()),
        "median_words": float(word_counts.median()),
        "max_words": int(word_counts.max()),
        "min_words": int(word_counts.min()),
    }


def split_overview(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a compact overview table for all splits."""
    rows = []
    for split, df in splits.items():
        stats = text_length_stats(df)
        rows.append(
            {
                "split": split,
                "rows": len(df),
                **stats,
            }
        )
    return pd.DataFrame(rows)


def concat_with_split(splits: dict[str, pd.DataFrame], selected_splits: Iterable[str] | None = None) -> pd.DataFrame:
    """Concatenate splits and add a split column."""
    names = list(selected_splits) if selected_splits is not None else list(splits)
    frames = []
    for split in names:
        frame = splits[split].copy()
        frame["split"] = split
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
