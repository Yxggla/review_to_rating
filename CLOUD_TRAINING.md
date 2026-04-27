# Cloud Training Guide

Use this guide when training DistilBERT on Colab, Kaggle, a rented GPU server, or a university GPU machine.

## 1. Recommended Cloud Setup

Use a GPU runtime when possible. CUDA GPUs are the most reliable option for this project.

Recommended minimum:

- Python 3.9+
- 8 GB RAM or more
- NVIDIA GPU preferred
- Enough disk space for dataset, model checkpoints, and outputs

## 2. Clone Project

```bash
git clone https://github.com/Yxggla/review_to_rating.git
cd review_to_rating
```

## 3. Install Dependencies

For cloud GPU training:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch does not detect CUDA, install PyTorch using the command recommended for your cloud environment, then run:

```bash
python scripts/09_check_environment.py
```

You want to see:

```text
cuda_available: True
```

CPU training is possible but much slower.

## 4. Upload Dataset

Place the dataset in the same structure used locally:

```text
data/amazon_reviews_multi_en/
├── summary.json
└── processed_3class/
    ├── train_3class.csv
    ├── validation_3class.csv
    └── test_3class.csv
```

The raw parquet files are not required for training if the processed CSV files exist.

## 5. Check Environment and Data

```bash
python scripts/09_check_environment.py
python scripts/01_check_data.py
```

## 6. Small DistilBERT Test Run

Run this first to confirm the cloud setup works:

```bash
python scripts/03_train_distilbert.py \
  --task sentiment \
  --max-train-samples 1000 \
  --max-validation-samples 300 \
  --max-test-samples 500 \
  --epochs 1 \
  --batch-size 16 \
  --fp16 auto
```

## 7. Medium Training Run for Project Results

This is a practical setting for course project results:

```bash
python scripts/03_train_distilbert.py \
  --task sentiment \
  --max-train-samples 20000 \
  --max-validation-samples 2000 \
  --epochs 1 \
  --batch-size 16 \
  --fp16 auto
```

```bash
python scripts/03_train_distilbert.py \
  --task rating \
  --max-train-samples 20000 \
  --max-validation-samples 2000 \
  --epochs 1 \
  --batch-size 16 \
  --fp16 auto
```

Then evaluate:

```bash
python scripts/04_evaluate_models.py
python scripts/05_error_analysis.py
```

## 8. If GPU Memory Is Not Enough

Try these changes:

```bash
python scripts/03_train_distilbert.py \
  --task sentiment \
  --max-train-samples 20000 \
  --max-validation-samples 2000 \
  --epochs 1 \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --max-length 128 \
  --fp16 auto
```

This lowers memory use while keeping an effective batch size of about 16.

## 9. Resume Training

If training stops but checkpoints exist under `models/distilbert_sentiment/` or `models/distilbert_rating/`, resume with:

```bash
python scripts/03_train_distilbert.py \
  --task sentiment \
  --resume-from-checkpoint models/distilbert_sentiment/checkpoint-XXXX
```

Replace `checkpoint-XXXX` with the actual checkpoint folder.

## 10. Download Results

After training, keep these folders:

```text
models/distilbert_sentiment/
models/distilbert_rating/
outputs/predictions/
outputs/metrics/
outputs/figures/
```

For final submission, you usually need code, prediction/evaluation outputs, figures, report, and slides. Large model weights do not need to be pushed to GitHub.
