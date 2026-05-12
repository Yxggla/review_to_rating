# Project Structure

This repository currently mixes three different concerns:

1. source code
2. training and evaluation artifacts
3. course/report coordination files

That is workable for a class project, but it becomes harder to share and maintain when model outputs and local datasets grow large.

## Recommended Mental Model

Treat the repository as four layers.

### 1. Core code

These files define the actual project logic and should stay small and stable.

```text
src/review_to_rating/
scripts/
requirements-basic.txt
requirements.txt
README.md
START_HERE.md
CLOUD_TRAINING.md
```

### 2. Local runtime data

These are needed for local experiments, but they are not the reusable codebase.

```text
data/datasets/
models/
outputs/
```

### 3. External or cloud artifacts

These are useful records of training results, but they are heavy and should be treated as generated artifacts.

```text
kaggle_outputs/
kaggle_upload/
release/
```

### 4. Course deliverables and coordination

These are not part of the runtime path.

```text
project_docs_amazon_sentiment/
report/
slides/
```

## What Is Clear Today

- `src/review_to_rating/` is the correct place for reusable Python logic.
- `scripts/` is the correct place for runnable entry points.
- `data/datasets/` is the actual dataset path used by the code.
- `release/` is now the right place for shareable runtime bundles.

## Main Structure Problems Found

### 1. Documentation path drift

Some docs described the dataset under `data/amazon_reviews_multi_en/`, but the code actually reads from `data/datasets/`.

Impact:

- new users can place files in the wrong folder
- setup instructions appear correct but fail at runtime

### 2. Artifact paths were partly hard-coded

Some code referenced Kaggle DistilBERT outputs through repeated literal paths instead of shared config constants.

Impact:

- structure changes require edits in multiple files
- dashboard and packaging scripts can drift apart

### 3. Large generated artifacts dominate the repository

Current large folders include:

- `data/`
- `data.zip`
- `kaggle_outputs/`
- `kaggle_upload/`

Impact:

- the real code footprint is hidden
- it is harder to tell what must be shared versus what can be regenerated

### 4. Course files live beside runtime files

This is not wrong, but it increases noise when someone just wants to run the model.

Impact:

- collaborators may open the wrong files first
- onboarding is slower than necessary

## Practical Usage Guide

### If you want to develop the project

Start here:

```text
README.md
START_HERE.md
src/review_to_rating/
scripts/
```

### If you want to run trained models only

Use:

```text
release/review_to_rating_runtime_baseline.zip
release/review_to_rating_runtime_distilbert.zip
```

### If you want to inspect cloud-trained DistilBERT results

Use:

```text
kaggle_outputs/distilbert/review_to_rating_distilbert/
```

## Suggested Next Cleanup

If you want the repository even cleaner, the next safe step is to move non-runtime materials into a dedicated top-level folder such as:

```text
docs/
artifacts/
```

A sensible future layout would be:

```text
review_to_rating/
├── src/
├── scripts/
├── data/
├── models/
├── outputs/
├── release/
├── docs/
│   ├── README.md
│   ├── START_HERE.md
│   ├── CLOUD_TRAINING.md
│   ├── report/
│   ├── slides/
│   └── course/
└── artifacts/
    ├── kaggle_outputs/
    └── kaggle_upload/
```

I did not move those folders yet because that would require broader path updates and carries more risk than the current cleanup pass.
