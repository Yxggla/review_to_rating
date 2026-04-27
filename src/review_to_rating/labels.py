"""Label definitions for sentiment and rating tasks."""

from __future__ import annotations

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
RATING_LABELS = [1, 2, 3, 4, 5]

SENTIMENT_TO_ID = {label: index for index, label in enumerate(SENTIMENT_LABELS)}
ID_TO_SENTIMENT = {index: label for label, index in SENTIMENT_TO_ID.items()}

RATING_TO_ID = {label: index for index, label in enumerate(RATING_LABELS)}
ID_TO_RATING = {index: label for label, index in RATING_TO_ID.items()}


def stars_to_sentiment(stars: int) -> str:
    """Map a 1-5 star rating to the project's three sentiment labels."""
    stars = int(stars)
    if stars in (1, 2):
        return "negative"
    if stars == 3:
        return "neutral"
    if stars in (4, 5):
        return "positive"
    raise ValueError(f"stars must be in 1..5, got {stars!r}")


def get_task_labels(task: str) -> list[str] | list[int]:
    """Return labels in the canonical display order for a task."""
    if task == "sentiment":
        return SENTIMENT_LABELS
    if task == "rating":
        return RATING_LABELS
    raise ValueError(f"Unknown task: {task}")


def get_target_column(task: str) -> str:
    """Return the dataset target column for a task."""
    if task == "sentiment":
        return "label_3class"
    if task == "rating":
        return "stars"
    raise ValueError(f"Unknown task: {task}")
