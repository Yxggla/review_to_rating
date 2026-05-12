#!/usr/bin/env python3
"""Streamlit dashboard for the Amazon review project."""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_to_rating.config import (  # noqa: E402
    CONFUSION_MATRIX_FIGURES_DIR,
    DISTILBERT_RATING_DIR,
    DISTILBERT_SENTIMENT_DIR,
    KAGGLE_DISTILBERT_MODELS_DIR,
    MODELS_DIR,
    SPLIT_FILES,
)
from review_to_rating.dashboard import (  # noqa: E402
    available_prediction_files,
    load_all_results_summary,
    load_data_overview,
    load_label_distribution,
    load_prediction_preview,
)
from review_to_rating.demo import distilbert_runtime_error  # noqa: E402


TASK_LABELS = {
    "sentiment": "Sentiment / 情感分类",
    "rating": "Rating / 星级预测",
}

SPLIT_LABELS = {
    "train": "Train / 训练集",
    "validation": "Validation / 验证集",
    "test": "Test / 测试集",
}


def show_altair_chart_compat(chart: alt.Chart) -> None:
    """Render Altair chart across old/new Streamlit width APIs."""
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


def show_dataframe_compat(df_or_styler) -> None:
    """Render dataframe across old/new Streamlit width APIs."""
    try:
        st.dataframe(df_or_styler, width="stretch")
    except TypeError:
        st.dataframe(df_or_styler, use_container_width=True)


def show_image_compat(image_path: str, caption: str) -> None:
    """Render image across old/new Streamlit width APIs."""
    try:
        st.image(image_path, caption=caption, width="stretch")
    except TypeError:
        st.image(image_path, caption=caption, use_container_width=True)


def show_label_chart(distribution: pd.DataFrame, label_type: str, title: str, help_text: str) -> None:
    chart_data = distribution[distribution["label_type"] == label_type][["label", "count"]].copy()
    chart_data["label"] = chart_data["label"].astype(str)
    st.markdown(f"**{title}**")
    st.caption(help_text)
    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("label:N", title="Label / 标签", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("count:Q", title="Count / 数量"),
            tooltip=[
                alt.Tooltip("label:N", title="Label / 标签"),
                alt.Tooltip("count:Q", title="Count / 数量", format=","),
            ],
        )
        .properties(height=260)
    )
    show_altair_chart_compat(chart)
    show_dataframe_compat(chart_data.rename(columns={"label": "Label / 标签", "count": "Count / 数量"}))


def format_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    formatted["task"] = formatted["task"].map(TASK_LABELS).fillna(formatted["task"])
    formatted = formatted.rename(
        columns={
            "task": "Task / 任务",
            "model": "Model / 模型",
            "accuracy": "Accuracy / 准确率",
            "precision_macro": "Macro Precision / 宏平均精确率",
            "recall_macro": "Macro Recall / 宏平均召回率",
            "macro_f1": "Macro F1 / 宏平均 F1",
            "samples": "Test Samples / 测试样本数",
        }
    )
    return formatted


def show_metric_chart(results: pd.DataFrame, metric: str, title: str) -> None:
    chart_data = results[["task", "model", metric]].copy()
    chart_data["Task / 任务"] = chart_data["task"].map(TASK_LABELS).fillna(chart_data["task"])
    chart_data["Model / 模型"] = chart_data["model"]
    chart_data["Score / 分数"] = chart_data[metric]
    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("Task / 任务:N", title=None, axis=alt.Axis(labelAngle=0, labelLimit=180)),
            xOffset=alt.XOffset("Model / 模型:N"),
            y=alt.Y("Score / 分数:Q", scale=alt.Scale(domain=[0, 1]), title="Score / 分数"),
            color=alt.Color("Model / 模型:N", title="Model / 模型"),
            tooltip=[
                alt.Tooltip("Task / 任务:N"),
                alt.Tooltip("Model / 模型:N"),
                alt.Tooltip("Score / 分数:Q", format=".4f"),
            ],
        )
        .properties(height=280)
    )
    st.markdown(f"**{title}**")
    show_altair_chart_compat(chart)


st.set_page_config(page_title="Amazon Review NLP Dashboard", layout="wide")
st.title("Amazon Review Sentiment and Rating Dashboard / 亚马逊评论分类演示")
st.caption("Predict review sentiment and star rating, compare baseline and DistilBERT results. / 输入评论文本，预测好中坏和星级，并对比模型效果。")

tabs = st.tabs([
    "Dataset / 数据集",
    "Model Results / 模型结果",
    "Predictions / 预测样例",
    "Demo / 实时演示",
])

with tabs[0]:
    st.subheader("Dataset Overview / 数据集概览")
    st.write("This section shows how many reviews are in each split and how labels are distributed. / 这里展示训练集、验证集、测试集规模，以及标签分布。")
    missing = [str(path) for path in SPLIT_FILES.values() if not path.exists()]
    if missing:
        st.error("Dataset files are missing. / 数据集文件缺失。")
        st.code("\n".join(missing))
    else:
        overview = load_data_overview()
        overview_display = overview.rename(
            columns={
                "split": "Split / 数据划分",
                "rows": "Rows / 样本数",
                "avg_text_length": "Avg Text Length / 平均文本长度",
            }
        )
        show_dataframe_compat(overview_display)

        split = st.selectbox("Split / 数据划分", list(SPLIT_FILES.keys()), format_func=lambda value: SPLIT_LABELS[value])
        distribution = load_label_distribution(split)
        col1, col2 = st.columns(2)
        with col1:
            show_label_chart(
                distribution,
                "sentiment",
                "Sentiment Labels / 情感标签",
                "negative = bad, neutral = middle, positive = good / negative 表示差评，neutral 表示中性，positive 表示好评。",
            )
        with col2:
            show_label_chart(
                distribution,
                "rating",
                "Star Ratings / 星级标签",
                "1 to 5 star labels used for rating prediction. / 星级预测任务使用 1 到 5 星作为标签。",
            )

with tabs[1]:
    st.subheader("Evaluation Results / 模型评估结果")
    st.write("Higher accuracy and Macro F1 are better. Macro F1 is useful when classes are imbalanced. / 准确率和 Macro F1 越高越好；类别不均衡时 Macro F1 更有参考价值。")
    results = load_all_results_summary()
    if results is None:
        st.info("No results summary found yet. / 还没有找到模型结果。")
    else:
        sentiment_results = results[results["task"] == "sentiment"]
        rating_results = results[results["task"] == "rating"]

        col1, col2 = st.columns(2)
        for column, task_name, task_results in [
            (col1, "Sentiment / 情感分类", sentiment_results),
            (col2, "Rating / 星级预测", rating_results),
        ]:
            with column:
                st.markdown(f"**{task_name}**")
                if not task_results.empty:
                    best = task_results.sort_values("macro_f1", ascending=False).iloc[0]
                    st.metric("Best Macro F1 / 最佳宏平均 F1", f"{best['macro_f1']:.3f}", help=f"Best model: {best['model']}")
                    st.metric("Best Accuracy / 最佳准确率", f"{best['accuracy']:.3f}")

        show_dataframe_compat(
            format_metric_table(results).style.format(
                {
                    "Accuracy / 准确率": "{:.4f}",
                    "Macro Precision / 宏平均精确率": "{:.4f}",
                    "Macro Recall / 宏平均召回率": "{:.4f}",
                    "Macro F1 / 宏平均 F1": "{:.4f}",
                }
            )
        )

        show_metric_chart(results, "accuracy", "Accuracy Comparison / 准确率对比")
        show_metric_chart(results, "macro_f1", "Macro F1 Comparison / Macro F1 对比")

    image_paths = sorted(CONFUSION_MATRIX_FIGURES_DIR.glob("*_confusion_matrix.png"))
    if image_paths:
        st.markdown("**Confusion Matrices / 混淆矩阵**")
        st.caption("Rows are true labels and columns are predicted labels. The diagonal means correct predictions. / 行是真实标签，列是预测标签，对角线表示预测正确。")
        cols = st.columns(2)
        for index, image_path in enumerate(image_paths):
            with cols[index % 2]:
                readable_name = image_path.stem.replace("_", " ").title()
                show_image_compat(str(image_path), f"{readable_name} / 混淆矩阵")
    else:
        st.info("No confusion matrix images found yet. / 还没有混淆矩阵图片。")

with tabs[2]:
    st.subheader("Prediction Preview / 预测样例")
    st.write("This table previews saved prediction files. Correct rows mean the predicted label matches the test label. / 这里预览已保存的预测文件，correct 表示预测标签和测试集真实标签一致。")
    prediction_files = available_prediction_files()
    if not prediction_files:
        st.info("No prediction files found yet. / 还没有预测文件。")
    else:
        experiment = st.selectbox("Experiment / 实验", list(prediction_files))
        preview = load_prediction_preview(experiment)
        preview["correct"] = preview["true_label"].astype(str) == preview["pred_label"].astype(str)

        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Preview Rows / 预览行数", len(preview))
        metric_col2.metric("Preview Accuracy / 预览准确率", f"{preview['correct'].mean():.3f}")

        show_dataframe_compat(
            preview.rename(
                columns={
                    "id": "ID",
                    "text": "Review Text / 评论文本",
                    "true_label": "True Label / 真实标签",
                    "pred_label": "Predicted Label / 预测标签",
                    "correct": "Correct / 是否正确",
                }
            )
        )
        errors = preview[~preview["correct"]]
        if not errors.empty:
            st.markdown("**Error Examples / 错误样例**")
            st.caption("Useful for explaining where the model still struggles. / 这些样例可以用来说明模型仍然容易犯错的地方。")
            show_dataframe_compat(errors.head(20))

with tabs[3]:
    st.subheader("Interactive Demo / 实时预测演示")
    st.caption("Enter an English Amazon review and predict both sentiment and star rating. / 输入一条英文亚马逊评论，模型会预测情感类别和星级。")
    review_text = st.text_area(
        "Review Text / 评论文本",
        value="The headphones are comfortable and the sound quality is great, but the battery life is shorter than expected.",
        height=120,
    )

    distilbert_model_options = {
        "Local models/": (DISTILBERT_SENTIMENT_DIR, DISTILBERT_RATING_DIR),
        "Kaggle output/": (
            KAGGLE_DISTILBERT_MODELS_DIR / "distilbert_sentiment",
            KAGGLE_DISTILBERT_MODELS_DIR / "distilbert_rating",
        ),
    }
    available_distilbert_options = {
        label: paths
        for label, paths in distilbert_model_options.items()
        if (paths[0] / "config.json").exists() and (paths[1] / "config.json").exists()
    }
    distilbert_import_error = distilbert_runtime_error() if available_distilbert_options else None

    backend_options = ["auto", "baseline"]
    if available_distilbert_options and distilbert_import_error is None:
        backend_options.insert(1, "distilbert")

    backend = st.radio("Model Backend / 模型后端", backend_options, horizontal=True)
    sentiment_model_dir, rating_model_dir = next(
        iter(available_distilbert_options.values()),
        (DISTILBERT_SENTIMENT_DIR, DISTILBERT_RATING_DIR),
    )
    if available_distilbert_options and distilbert_import_error is None:
        model_source = st.selectbox("DistilBERT Model Source / DistilBERT 模型来源", list(available_distilbert_options))
        sentiment_model_dir, rating_model_dir = available_distilbert_options[model_source]

    distilbert_ready = bool(available_distilbert_options) and distilbert_import_error is None
    baseline_ready = (MODELS_DIR / "baseline_sentiment.joblib").exists() and (MODELS_DIR / "baseline_rating.joblib").exists()
    if backend == "auto" and not distilbert_ready and baseline_ready:
        st.info("Auto mode will use baseline because DistilBERT is unavailable in this environment. / 当前环境不能运行 DistilBERT，auto 会使用 baseline。")
    if backend == "distilbert" and not distilbert_ready:
        st.warning("DistilBERT is unavailable in this environment. / 当前环境不能运行 DistilBERT。")
    if distilbert_import_error is not None:
        st.caption(f"DistilBERT disabled in this environment: {distilbert_import_error}")
    if not distilbert_ready and not baseline_ready:
        st.warning("No saved models are available yet. / 还没有可用模型。")
    can_predict = (backend == "distilbert" and distilbert_ready) or (backend == "baseline" and baseline_ready) or (
        backend == "auto" and (distilbert_ready or baseline_ready)
    )
    if st.button("Predict / 开始预测", disabled=not can_predict):
        from review_to_rating.demo import predict_review

        try:
            result = predict_review(
                review_text,
                sentiment_model_dir,
                rating_model_dir,
                backend=backend,
                baseline_sentiment_model_path=MODELS_DIR / "baseline_sentiment.joblib",
                baseline_rating_model_path=MODELS_DIR / "baseline_rating.joblib",
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Backend / 后端", str(result["backend"]))
            col2.metric("Sentiment / 情感", str(result["sentiment"]))
            col3.metric("Rating / 星级", f"{result['rating']} stars")
            st.caption("Sentiment labels: negative = bad, neutral = middle, positive = good. / 情感标签：negative 差评，neutral 中性，positive 好评。")
            show_dataframe_compat(pd.DataFrame([result]))
