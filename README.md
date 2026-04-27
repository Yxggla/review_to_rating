# Amazon Review Sentiment and Rating Classification

This project predicts sentiment and star ratings from English Amazon reviews.

It contains two related NLP classification tasks:

- Sentiment classification: `negative`, `neutral`, `positive`
- Rating prediction: `1`, `2`, `3`, `4`, `5` stars

The project is organized as a Python codebase. Shared logic lives in `src/review_to_rating/`, while runnable experiment entry points live in `scripts/`.

## Project Structure

```text
review_to_rating/
├── data/                         # Local dataset files
├── src/review_to_rating/          # Reusable Python package
├── scripts/                       # Runnable project scripts
├── outputs/                       # Generated predictions, metrics, figures
├── models/                        # Local model checkpoints
├── project_docs_amazon_sentiment/  # Member responsibility documents
├── report/                        # Report drafts and final files
└── slides/                        # Presentation drafts and final files
```

## Setup

For Windows and macOS setup, use the quickstart guide:

```text
START_HERE.md
```

Install lightweight dependencies for data checks, baseline models, evaluation, dashboard, and demo:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-basic.txt
```

Install full dependencies only when training DistilBERT:

```bash
pip install -r requirements.txt
```

## Main Commands

Check data files and generate data distribution figures:

```bash
python scripts/01_check_data.py
```

Train TF-IDF + Logistic Regression baselines:

```bash
python scripts/02_train_baseline.py
```

Train DistilBERT models:

```bash
python scripts/03_train_distilbert.py --task both
```

Evaluate available prediction files:

```bash
python scripts/04_evaluate_models.py
```

Extract success and error cases:

```bash
python scripts/05_error_analysis.py
```

Run the demo:

```bash
python scripts/06_run_demo.py --text "The headphones are comfortable and the sound quality is great."
```

Open the visual dashboard:

```bash
streamlit run scripts/07_visual_app.py
```

For a quick local smoke test without training deep learning models:

```bash
python scripts/08_smoke_test.py
```

## Prediction File Format

All prediction CSV files use the same minimum schema:

```text
id,text,true_label,pred_label
```

The four standard experiment names are:

- `baseline_sentiment`
- `baseline_rating`
- `distilbert_sentiment`
- `distilbert_rating`

## Member Responsibilities

- Member 1: DistilBERT training, demo, final integration
- Member 2: data checks, statistics, TF-IDF baseline
- Member 3: evaluation, result tables, confusion matrices
- Member 4: dataset and task explanation
- Member 5: result analysis and discussion
- Member 6: report, slides, contribution, final submission

## Course Requirement Coverage

- Topic: NLP text analysis for Amazon review sentiment and rating prediction
- Programming: Python scripts and reusable package modules
- Deep learning: DistilBERT fine-tuning code for both tasks
- Findings/results: metrics tables, classification reports, confusion matrices, and error cases
- Report: draft folders and contribution template under `report/`
- Presentation: slide folders and outline under `slides/`
