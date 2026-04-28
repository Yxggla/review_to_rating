from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


DATA_DIR = Path("/kaggle/input/amazon-review-rating-processed-3class")
OUTPUT_DIR = Path("/kaggle/working/review_to_rating_distilbert")
BASE_MODEL = "distilbert-base-uncased"

# Full Kaggle training run.
MAX_TRAIN_SAMPLES = None
MAX_VALIDATION_SAMPLES = None
MAX_TEST_SAMPLES = None
EPOCHS = 2
BATCH_SIZE = 16
MAX_LENGTH = 192
LEARNING_RATE = 2e-5
SEED = 42

TEXT_COLUMN = "text"
ID_COLUMN = "id"
RATING_COLUMN = "stars"
SENTIMENT_COLUMN = "label_3class"

TASKS = {
    "sentiment": {
        "target": SENTIMENT_COLUMN,
        "labels": ["negative", "neutral", "positive"],
        "model_dir": OUTPUT_DIR / "models" / "distilbert_sentiment",
    },
    "rating": {
        "target": RATING_COLUMN,
        "labels": [1, 2, 3, 4, 5],
        "model_dir": OUTPUT_DIR / "models" / "distilbert_rating",
    },
}


def resolve_data_dir() -> Path:
    if (DATA_DIR / "train_3class.csv").exists():
        return DATA_DIR

    fallback_dir = Path("/kaggle/working/amazon-review-rating-processed-3class")
    if (fallback_dir / "train_3class.csv").exists():
        return fallback_dir

    print("Dataset is not mounted under /kaggle/input; trying Kaggle API download...")
    import subprocess

    fallback_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "yxggla/amazon-review-rating-processed-3class",
            "-p",
            str(fallback_dir),
            "--unzip",
        ],
        check=True,
    )
    return fallback_dir


class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict, labels: list[int] | None = None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[index])
        return item


def sample_dataframe(df: pd.DataFrame, n: int | None, stratify_column: str) -> pd.DataFrame:
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    return (
        df.groupby(stratify_column, group_keys=False)
        .apply(lambda group: group.sample(max(1, round(n * len(group) / len(df))), random_state=SEED))
        .sample(n=min(n, len(df)), random_state=SEED)
        .reset_index(drop=True)
    )


def encode_dataframe(
    tokenizer: AutoTokenizer,
    df: pd.DataFrame,
    target_column: str,
    label_to_id: dict,
    include_labels: bool = True,
) -> ReviewDataset:
    encodings = tokenizer(
        df[TEXT_COLUMN].fillna("").astype(str).tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    labels = None
    if include_labels:
        labels = [label_to_id[value] for value in df[target_column].tolist()]
    return ReviewDataset(encodings, labels)


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision,
        "recall_macro": recall,
        "macro_f1": f1,
    }


def train_task(task: str, train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    config = TASKS[task]
    target_column = config["target"]
    labels = config["labels"]
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: str(label) for index, label in enumerate(labels)}
    model_dir = config["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    train_sample = sample_dataframe(train_df, MAX_TRAIN_SAMPLES, target_column)
    validation_sample = sample_dataframe(validation_df, MAX_VALIDATION_SAMPLES, target_column)
    test_sample = sample_dataframe(test_df, MAX_TEST_SAMPLES, target_column)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id={str(key): value for key, value in label_to_id.items()},
    )

    train_dataset = encode_dataframe(tokenizer, train_sample, target_column, label_to_id)
    validation_dataset = encode_dataframe(tokenizer, validation_sample, target_column, label_to_id)
    test_dataset = encode_dataframe(tokenizer, test_sample, target_column, label_to_id, include_labels=False)

    fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(model_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        fp16=fp16,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    output = trainer.predict(test_dataset)
    pred_ids = np.argmax(output.predictions, axis=-1)
    pred_labels = [labels[int(pred_id)] for pred_id in pred_ids]
    predictions = pd.DataFrame(
        {
            "id": test_sample[ID_COLUMN],
            "text": test_sample[TEXT_COLUMN],
            "true_label": test_sample[target_column],
            "pred_label": pred_labels,
        }
    )

    predictions_dir = OUTPUT_DIR / "predictions"
    metrics_dir = OUTPUT_DIR / "metrics"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_dir / f"distilbert_{task}_predictions.csv", index=False)

    metrics = compute_metrics((output.predictions, [label_to_id[value] for value in test_sample[target_column]]))
    report = classification_report(
        test_sample[target_column],
        pred_labels,
        labels=labels,
        zero_division=0,
    )
    (metrics_dir / f"distilbert_{task}_classification_report.txt").write_text(report)

    metadata = {
        "task": task,
        "base_model": BASE_MODEL,
        "train_samples": len(train_sample),
        "validation_samples": len(validation_sample),
        "test_samples": len(test_sample),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "fp16": fp16,
        **metrics,
    }
    (model_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda_device:", torch.cuda.get_device_name(0))

    data_dir = resolve_data_dir()
    print("data_dir:", data_dir)
    train_df = pd.read_csv(data_dir / "train_3class.csv")
    validation_df = pd.read_csv(data_dir / "validation_3class.csv")
    test_df = pd.read_csv(data_dir / "test_3class.csv")

    summaries = []
    for task in ["sentiment", "rating"]:
        summaries.append(train_task(task, train_df, validation_df, test_df))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUTPUT_DIR / "metrics" / "distilbert_results_summary.csv", index=False)
    print(summary_df)


if __name__ == "__main__":
    main()
