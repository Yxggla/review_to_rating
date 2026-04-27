"""DistilBERT training and inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .config import ID_COLUMN, TEXT_COLUMN
from .labels import (
    ID_TO_RATING,
    ID_TO_SENTIMENT,
    RATING_TO_ID,
    SENTIMENT_TO_ID,
    get_target_column,
)


@dataclass(frozen=True)
class DistilBertTaskConfig:
    task: str
    model_dir: Path
    num_labels: int
    label_to_id: dict
    id_to_label: dict


class ReviewDataset(Dataset):
    """Torch dataset for tokenized review classification."""

    def __init__(self, encodings: dict, labels: list[int] | None = None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[index])
        return item

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])


def detect_accelerator() -> str:
    """Return the best available training accelerator name."""
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.get_device_name(0)}"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def should_use_fp16(fp16_mode: str) -> bool:
    """Resolve fp16 mode. Auto uses fp16 only on CUDA."""
    if fp16_mode == "true":
        return True
    if fp16_mode == "false":
        return False
    if fp16_mode == "auto":
        return torch.cuda.is_available()
    raise ValueError(f"fp16_mode must be auto, true, or false, got {fp16_mode!r}")


def get_task_config(task: str, model_dir: Path) -> DistilBertTaskConfig:
    """Return DistilBERT label mappings for a task."""
    if task == "sentiment":
        return DistilBertTaskConfig(task, model_dir, 3, SENTIMENT_TO_ID, ID_TO_SENTIMENT)
    if task == "rating":
        return DistilBertTaskConfig(task, model_dir, 5, RATING_TO_ID, ID_TO_RATING)
    raise ValueError(f"Unknown task: {task}")


def encode_dataframe(
    tokenizer,
    df: pd.DataFrame,
    task_config: DistilBertTaskConfig,
    max_length: int,
    include_labels: bool = True,
) -> ReviewDataset:
    """Tokenize a dataframe and optionally attach labels."""
    encodings = tokenizer(
        df[TEXT_COLUMN].fillna("").astype(str).tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    labels = None
    if include_labels:
        target_column = get_target_column(task_config.task)
        labels = [task_config.label_to_id[value] for value in df[target_column].tolist()]
    return ReviewDataset(encodings, labels)


def compute_metrics(eval_pred) -> dict[str, float]:
    """Compute metrics used during Trainer validation."""
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


def build_training_args(model_dir: Path, **kwargs) -> TrainingArguments:
    """Build TrainingArguments across transformers versions."""
    signature = inspect.signature(TrainingArguments.__init__)
    args = dict(kwargs)
    if "evaluation_strategy" in args and "evaluation_strategy" not in signature.parameters:
        args["eval_strategy"] = args.pop("evaluation_strategy")
    return TrainingArguments(output_dir=str(model_dir), **args)


def train_distilbert(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    task: str,
    model_dir: Path,
    base_model_name: str = "distilbert-base-uncased",
    max_length: int = 192,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    epochs: int = 2,
    fp16_mode: str = "auto",
    gradient_accumulation_steps: int = 1,
    save_total_limit: int = 2,
    logging_steps: int = 100,
    resume_from_checkpoint: str | None = None,
) -> None:
    """Fine-tune DistilBERT and save tokenizer/model to model_dir."""
    task_config = get_task_config(task, model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=task_config.num_labels,
        id2label={int(key): str(value) for key, value in task_config.id_to_label.items()},
        label2id={str(key): int(value) for key, value in task_config.label_to_id.items()},
    )

    train_dataset = encode_dataframe(tokenizer, train_df, task_config, max_length=max_length)
    validation_dataset = encode_dataframe(tokenizer, validation_df, task_config, max_length=max_length)

    training_args = build_training_args(
        model_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        fp16=should_use_fp16(fp16_mode),
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_dir=str(model_dir / "logs"),
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        report_to="none",
        seed=42,
    )

    metadata = {
        "task": task,
        "base_model_name": base_model_name,
        "accelerator": detect_accelerator(),
        "train_samples": len(train_df),
        "validation_samples": len(validation_df),
        "max_length": max_length,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "fp16_mode": fp16_mode,
        "fp16_resolved": should_use_fp16(fp16_mode),
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    (model_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))


def predict_distilbert(
    df: pd.DataFrame,
    task: str,
    model_dir: Path,
    max_length: int = 192,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Load a saved DistilBERT model and return normalized predictions."""
    task_config = get_task_config(task, model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    dataset = encode_dataframe(tokenizer, df, task_config, max_length=max_length, include_labels=False)

    args = build_training_args(
        model_dir / "predict_tmp",
        per_device_eval_batch_size=batch_size,
        report_to="none",
    )
    trainer = Trainer(model=model, args=args)
    output = trainer.predict(dataset)
    pred_ids = np.argmax(output.predictions, axis=-1)
    pred_labels = [task_config.id_to_label[int(pred_id)] for pred_id in pred_ids]
    target_column = get_target_column(task)
    return pd.DataFrame(
        {
            "id": df[ID_COLUMN],
            "text": df[TEXT_COLUMN],
            "true_label": df[target_column],
            "pred_label": pred_labels,
        }
    )
