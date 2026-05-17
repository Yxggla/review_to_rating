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
    get_text_length_plot_path,
    get_wordcloud_paths,
    load_all_results_summary,
    load_data_overview,
    load_label_distribution,
    load_prediction_preview,
)
from review_to_rating.demo import distilbert_runtime_error  # noqa: E402


@st.cache_resource(
    show_spinner="Loading DistilBERT stack (first time can take tens of seconds) / 首次加载 DistilBERT 可能需数十秒…",
)
def _cached_distilbert_runtime_error() -> str | None:
    """Session-cached import check so the Demo tab is not blank while torch/transformers load."""
    return distilbert_runtime_error()


SAMPLE_REVIEWS = [
    "The headphones are comfortable and the sound quality is great, but the battery life is shorter than expected.",
    "Absolutely love this product! Fast shipping and excellent quality. Will definitely buy again.",
    "Terrible experience. The product broke after just one week and customer service was unhelpful.",
    "It's okay, nothing special. Does the job but there are better options available for the same price.",
    "The product arrived damaged and stopped working after two days. Complete waste of money.",
    "Pretty good for the price, but I've had better. Shipping was slow and packaging was dented.",
]


TASK_LABELS = {
    "sentiment": "Sentiment / 情感分类",
    "rating": "Rating / 星级预测",
}

MODEL_LABELS = {
    "baseline": "Baseline",
    "distilbert": "DistilBERT",
}

SPLIT_LABELS = {
    "train": "Train / 训练集",
    "validation": "Validation / 验证集",
    "test": "Test / 测试集",
}


def show_altair_chart_compat(chart: alt.Chart) -> None:
    """Render Altair chart across old/new Streamlit width APIs."""
    try:
        st.altair_chart(chart)
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


def format_experiment_label(experiment_name: str) -> str:
    model, task = experiment_name.split("_", 1)
    return f"{MODEL_LABELS.get(model, model.title())} - {TASK_LABELS.get(task, task.title())}"


def describe_sample(index: int) -> str:
    text = SAMPLE_REVIEWS[index].replace("\n", " ")
    preview = text[:72].rstrip()
    suffix = "..." if len(text) > 72 else ""
    return f"Sample {index + 1}: {preview}{suffix}"


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
        .properties(height=260, width="container")
    )
    show_altair_chart_compat(chart)
    show_dataframe_compat(chart_data.rename(columns={"label": "Label / 标签", "count": "Count / 数量"}))


def format_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    formatted["task"] = formatted["task"].map(TASK_LABELS).fillna(formatted["task"])
    formatted["model"] = formatted["model"].map(MODEL_LABELS).fillna(formatted["model"])
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
    chart_data["Model / 模型"] = chart_data["model"].map(MODEL_LABELS).fillna(chart_data["model"])
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
        .properties(height=280, width="container")
    )
    st.markdown(f"**{title}**")
    show_altair_chart_compat(chart)


st.set_page_config(page_title="Amazon Review NLP Dashboard", layout="wide")

# Theme toggle in sidebar
st.sidebar.markdown("### Theme / 主题")
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

if st.sidebar.button("🌙 Dark" if not st.session_state["dark_mode"] else "☀️ Light"):
    st.session_state["dark_mode"] = not st.session_state["dark_mode"]
    st.rerun()

# Apply theme CSS
if st.session_state["dark_mode"]:
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; color: #f0f2f6; }
        .stTabs [data-baseweb="tab-list"] { background-color: #1a1d24; }
        .stTabs [data-baseweb="tab"] { color: #f0f2f6; }
        h1, h2, h3, h4, p, .stMarkdown { color: #f0f2f6; }
        .stDataFrame { color: #f0f2f6; }
        </style>
    """, unsafe_allow_html=True)

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
                "mean_words": "Mean Words / 平均词数",
                "median_words": "Median Words / 中位词数",
                "max_words": "Max Words / 最大词数",
                "min_words": "Min Words / 最小词数",
            }
        )
        show_dataframe_compat(overview_display)

        split = st.selectbox("Split / 数据划分", list(SPLIT_FILES.keys()), format_func=lambda value: SPLIT_LABELS[value])
        distribution = load_label_distribution(split)
        st.caption("The table shows dataset size and typical review length. The charts below show whether labels are balanced or skewed. / 上表先看数据规模和评论长度；下方两张图用来说明标签分布是否均衡。")
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

        st.markdown("---")
        st.markdown("**Text Length Distribution / 文本长度分布**")
        text_length_path = get_text_length_plot_path()
        if text_length_path.exists():
            show_image_compat(str(text_length_path), "Text Length Distribution / 文本长度分布")
        else:
            st.info("Text length distribution plot not generated yet. / 文本长度分布图尚未生成。")

        st.markdown("**Word Clouds / 词云**")
        st.caption("Visualize most frequent words in positive and negative reviews. / 可视化正面和负面评论中的高频词汇。")
        wordcloud_paths = get_wordcloud_paths(split)
        if wordcloud_paths["positive"].exists() and wordcloud_paths["negative"].exists():
            col1, col2 = st.columns(2)
            with col1:
                show_image_compat(str(wordcloud_paths["positive"]), "Positive Reviews Word Cloud / 正面评论词云")
            with col2:
                show_image_compat(str(wordcloud_paths["negative"]), "Negative Reviews Word Cloud / 负面评论词云")
        else:
            st.info("Word clouds not generated yet. Run data check script to generate them. / 词云尚未生成，运行数据检查脚本生成。")

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
                    st.metric(
                        "Best Macro F1 / 最佳宏平均 F1",
                        f"{best['macro_f1']:.3f}",
                        help=f"Best model: {MODEL_LABELS.get(best['model'], best['model'])}",
                    )
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
        st.caption("Rows are true labels and columns are predicted labels. Darker diagonal blocks mean the model is often correct; off-diagonal blocks show where labels are easy to confuse. / 行是真实标签，列是预测标签；对角线越深说明预测越准，非对角线越深说明这些类别更容易混淆。")
        cols = st.columns(2)
        for index, image_path in enumerate(image_paths):
            with cols[index % 2]:
                readable_name = image_path.stem.replace("_", " ").title()
                show_image_compat(str(image_path), f"{readable_name} / 混淆矩阵")
    else:
        st.info("No confusion matrix images found yet. / 还没有混淆矩阵图片。")

with tabs[2]:
    st.subheader("Prediction Preview / 预测样例")
    st.write("This tab previews saved prediction files. Correct rows mean the predicted label matches the test label. / 这里预览预测结果；`correct` 表示预测标签和真实标签是否一致。")
    prediction_files = available_prediction_files()
    if not prediction_files:
        st.info("No prediction files found yet. / 还没有预测文件。")
    else:
        experiment = st.selectbox(
            "Experiment / 实验",
            sorted(prediction_files),
            format_func=format_experiment_label,
        )
        preview = load_prediction_preview(experiment)
        preview["correct"] = preview["true_label"].astype(str) == preview["pred_label"].astype(str)

        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Preview Rows / 预览行数", len(preview))
        metric_col2.metric("Preview Accuracy / 预览准确率", f"{preview['correct'].mean():.3f}")
        st.caption("Read this table row by row: text is the original review, true label is the expected answer, predicted label is the model output. / 这张表按行阅读即可：`text` 是原评论，`true_label` 是真实标签，`pred_label` 是模型输出。")

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
            st.caption("These rows are useful when you want to explain model mistakes with concrete examples. / 如果你要口头解释模型为什么会错，直接挑这里的例子讲就够了。")
            show_dataframe_compat(
                errors.head(20).rename(
                    columns={
                        "id": "ID",
                        "text": "Review Text / 评论文本",
                        "true_label": "True Label / 真实标签",
                        "pred_label": "Predicted Label / 预测标签",
                        "correct": "Correct / 是否正确",
                    }
                )
            )

with tabs[3]:
    st.subheader("Interactive Demo / 实时预测演示")
    st.caption("Enter an English Amazon review and predict both sentiment and star rating. / 输入一条英文亚马逊评论，模型会预测情感类别和星级。")

    st.markdown("**Sample Reviews / 示例评论**")
    st.caption("Pick a prepared example if you want a fast live demo. / 如果想快速演示，可以直接加载示例评论。")

    if "review_text_input" not in st.session_state:
        st.session_state["review_text_input"] = SAMPLE_REVIEWS[0]
    if "sample_review_index" not in st.session_state:
        st.session_state["sample_review_index"] = 0

    def _load_selected_sample() -> None:
        st.session_state["review_text_input"] = SAMPLE_REVIEWS[st.session_state["sample_review_index"]]

    sample_col1, sample_col2 = st.columns([3, 1])
    with sample_col1:
        st.selectbox(
            "Quick Sample / 快速示例",
            list(range(len(SAMPLE_REVIEWS))),
            key="sample_review_index",
            format_func=describe_sample,
        )
    with sample_col2:
        st.button("Load Sample / 加载示例", on_click=_load_selected_sample)

    review_text = st.text_area(
        "Review Text / 评论文本",
        key="review_text_input",
        height=160,
    )
    text_metric_col1, text_metric_col2 = st.columns(2)
    text_metric_col1.metric("Word Count / 词数", len(review_text.split()))
    text_metric_col2.metric("Character Count / 字符数", len(review_text))
    st.caption(
        "If DistilBERT checkpoints are present, the first load imports torch/transformers; "
        "model options and the predict button appear when that finishes. / "
        "若检测到 DistilBERT 权重，首次会加载推理环境，下方「模型后端」与按钮在加载结束后出现。"
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
    distilbert_import_error = _cached_distilbert_runtime_error() if available_distilbert_options else None

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

    col_btn, col_export = st.columns([1, 3])
    with col_btn:
        predict_clicked = st.button("Predict / 开始预测", disabled=not can_predict, type="primary")

    if predict_clicked:
        from review_to_rating.demo import predict_review

        with st.spinner("Predicting... / 预测中..."):
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
                result = None
        if result:
            col1, col2, col3 = st.columns(3)
            col1.metric("Backend / 后端", str(result["backend"]))
            col2.metric("Sentiment / 情感", str(result["sentiment"]))
            col3.metric("Rating / 星级", f"{result['rating']} stars")
            st.caption("Sentiment labels: negative = bad, neutral = middle, positive = good. / 情感标签：negative 差评，neutral 中性，positive 好评。")
            st.success(
                f"Interpretation / 解读: this review reads as `{result['sentiment']}` and lands around `{result['rating']}` stars. "
                "Use this box when presenting so viewers immediately understand what the numbers mean."
            )

            result_df = pd.DataFrame([result])
            show_dataframe_compat(result_df)

            with col_export:
                csv_data = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Result / 下载结果 (CSV)",
                    data=csv_data,
                    file_name="prediction_result.csv",
                    mime="text/csv",
                )
