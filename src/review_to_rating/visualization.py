"""Figure generation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud

from .config import RATING_COLUMN, SENTIMENT_COLUMN, TEXT_COLUMN
from .labels import RATING_LABELS, SENTIMENT_LABELS


def save_label_distribution_plots(splits: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Save sentiment and rating distribution bar charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, df in splits.items():
        for column, order, title in [
            (SENTIMENT_COLUMN, SENTIMENT_LABELS, "Sentiment Label Distribution"),
            (RATING_COLUMN, RATING_LABELS, "Rating Label Distribution"),
        ]:
            counts = df[column].value_counts().reindex(order, fill_value=0)
            plt.figure(figsize=(7, 4))
            sns.barplot(x=[str(item) for item in counts.index], y=counts.values)
            plt.title(f"{title} - {split}")
            plt.xlabel(column)
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(output_dir / f"{split}_{column}_distribution.png", dpi=200)
            plt.close()


def save_text_length_plot(splits: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Save a text length histogram for all splits."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, df in splits.items():
        lengths = df[TEXT_COLUMN].fillna("").astype(str).str.split().str.len()
        rows.append(pd.DataFrame({"split": split, "word_count": lengths}))
    combined = pd.concat(rows, ignore_index=True)
    plt.figure(figsize=(8, 4))
    sns.histplot(data=combined, x="word_count", hue="split", bins=50, element="step")
    plt.xlim(0, combined["word_count"].quantile(0.99))
    plt.title("Review Text Length Distribution")
    plt.xlabel("word count")
    plt.ylabel("reviews")
    plt.tight_layout()
    plt.savefig(output_dir / "text_length_distribution.png", dpi=200)
    plt.close()


def save_confusion_matrix(
    df: pd.DataFrame,
    labels: Sequence[str] | Sequence[int],
    title: str,
    output_path: Path,
) -> None:
    """Save one confusion matrix figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(df["true_label"], df["pred_label"], labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[str(label) for label in labels],
        yticklabels=[str(label) for label in labels],
    )
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def generate_wordcloud(texts: list[str], output_path: Path, colormap: str = "viridis") -> None:
    """Generate and save a word cloud image from a list of texts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_text = " ".join(texts)
    if not combined_text.strip():
        return
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap=colormap,
        max_words=100,
        random_state=42,
    ).generate(combined_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
