#!/usr/bin/env python3
"""Streamlit dashboard for the Amazon review project."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_to_rating.config import (  # noqa: E402
    CONFUSION_MATRIX_FIGURES_DIR,
    DISTILBERT_RATING_DIR,
    DISTILBERT_SENTIMENT_DIR,
    MODELS_DIR,
    SPLIT_FILES,
)
from review_to_rating.dashboard import (  # noqa: E402
    available_prediction_files,
    load_data_overview,
    load_label_distribution,
    load_prediction_preview,
    load_results_summary,
)


st.set_page_config(page_title="Amazon Review NLP Dashboard", layout="wide")
st.title("Amazon Review Sentiment and Rating Dashboard")

tabs = st.tabs(["Dataset", "Model Results", "Predictions", "Demo"])

with tabs[0]:
    st.subheader("Dataset Overview")
    missing = [str(path) for path in SPLIT_FILES.values() if not path.exists()]
    if missing:
        st.error("Dataset files are missing. Place the CSV files under data/amazon_reviews_multi_en/processed_3class/.")
        st.code("\n".join(missing))
    else:
        overview = load_data_overview()
        st.dataframe(overview, use_container_width=True)

        split = st.selectbox("Split", list(SPLIT_FILES.keys()))
        distribution = load_label_distribution(split)
        col1, col2 = st.columns(2)
        with col1:
            sentiment = distribution[distribution["label_type"] == "sentiment"]
            st.bar_chart(sentiment.set_index("label")["count"])
        with col2:
            rating = distribution[distribution["label_type"] == "rating"]
            st.bar_chart(rating.set_index("label")["count"])

with tabs[1]:
    st.subheader("Evaluation Results")
    results = load_results_summary()
    if results is None:
        st.info("No results summary found yet. Run scripts/04_evaluate_models.py after generating predictions.")
    else:
        st.dataframe(results, use_container_width=True)

    image_paths = sorted(CONFUSION_MATRIX_FIGURES_DIR.glob("*_confusion_matrix.png"))
    if image_paths:
        cols = st.columns(2)
        for index, image_path in enumerate(image_paths):
            with cols[index % 2]:
                st.image(str(image_path), caption=image_path.stem, use_container_width=True)
    else:
        st.info("No confusion matrix images found yet.")

with tabs[2]:
    st.subheader("Prediction Preview")
    prediction_files = available_prediction_files()
    if not prediction_files:
        st.info("No prediction files found yet. Run baseline or DistilBERT scripts first.")
    else:
        experiment = st.selectbox("Experiment", list(prediction_files))
        preview = load_prediction_preview(experiment)
        preview["correct"] = preview["true_label"].astype(str) == preview["pred_label"].astype(str)

        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Preview rows", len(preview))
        metric_col2.metric("Preview accuracy", f"{preview['correct'].mean():.3f}")

        st.dataframe(preview, use_container_width=True)
        errors = preview[~preview["correct"]]
        if not errors.empty:
            st.write("Error examples")
            st.dataframe(errors.head(20), use_container_width=True)

with tabs[3]:
    st.subheader("Interactive Demo")
    st.caption("Auto mode uses DistilBERT when available, otherwise it falls back to saved baseline models.")
    review_text = st.text_area(
        "Review text",
        value="The headphones are comfortable and the sound quality is great, but the battery life is shorter than expected.",
        height=120,
    )
    distilbert_ready = (DISTILBERT_SENTIMENT_DIR / "config.json").exists() and (DISTILBERT_RATING_DIR / "config.json").exists()
    baseline_ready = (MODELS_DIR / "baseline_sentiment.joblib").exists() and (MODELS_DIR / "baseline_rating.joblib").exists()
    if not distilbert_ready and baseline_ready:
        st.info("Using baseline fallback because DistilBERT model files are not available yet.")
    if not distilbert_ready and not baseline_ready:
        st.warning("No saved models are available yet. Run baseline training or train DistilBERT first.")
    if st.button("Predict", disabled=not (distilbert_ready or baseline_ready)):
        from review_to_rating.demo import predict_review

        result = predict_review(
            review_text,
            DISTILBERT_SENTIMENT_DIR,
            DISTILBERT_RATING_DIR,
            backend="auto",
            baseline_sentiment_model_path=MODELS_DIR / "baseline_sentiment.joblib",
            baseline_rating_model_path=MODELS_DIR / "baseline_rating.joblib",
        )
        st.write(pd.DataFrame([result]))
