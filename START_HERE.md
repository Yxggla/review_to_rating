# Start Here: Windows and macOS Quickstart

This guide is for group members who only need to run the project locally.

Default setup runs the lightweight project flow:

- data checking
- TF-IDF + Logistic Regression baseline
- evaluation and confusion matrices
- error case extraction
- visual dashboard
- demo with baseline fallback

DistilBERT training is optional and requires the full dependencies. Use Python 3.10 or newer for DistilBERT.

## 1. Install Python

Use Python 3.10 or newer.

Check your version:

```bash
python --version
```

On Windows, if `python` does not work, try:

```powershell
py --version
```

## 2. Get the Project

Clone the repository:

```bash
git clone https://github.com/Yxggla/review_to_rating.git
cd review_to_rating
```

Or download the repository ZIP from GitHub and open the extracted folder in Terminal or PowerShell.

## 3. Put the Dataset in the Right Folder

Large data files are not stored in GitHub. Put the local dataset here:

```text
data/amazon_reviews_multi_en/
├── summary.json
├── processed_3class/
│   ├── train_3class.csv
│   ├── validation_3class.csv
│   └── test_3class.csv
└── raw_parquet/
    ├── train.parquet
    ├── validation.parquet
    └── test.parquet
```

For the lightweight flow, the three CSV files and `summary.json` are enough.

## 4. Create Environment and Install Lightweight Dependencies

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-basic.txt
```

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-basic.txt
```

If PowerShell blocks activation, run this once in the same PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 5. Run a Quick Project Test

This command runs the lightweight flow with a small baseline training sample:

```bash
python scripts/08_smoke_test.py
```

Expected result:

- data validation prints `OK`
- baseline prediction files are generated locally
- evaluation results are generated locally
- demo prints sentiment and star rating predictions

Generated outputs are saved under `outputs/` and are ignored by git.

## 6. Open the Visual Dashboard

```bash
streamlit run scripts/07_visual_app.py
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

The dashboard shows:

- dataset overview
- label distributions
- model results
- confusion matrices
- prediction previews
- interactive demo

## 7. Optional: Install Full DistilBERT Dependencies

Only do this if you need to train or run DistilBERT models.

```bash
pip install -r requirements.txt
```

Check whether the machine has a usable GPU:

```bash
python scripts/09_check_environment.py
```

Then train models:

```bash
python scripts/03_train_distilbert.py --task both
```

For a small test run:

```bash
python scripts/03_train_distilbert.py --task sentiment --max-train-samples 1000 --max-validation-samples 300 --epochs 1
```

For cloud GPU training, see `CLOUD_TRAINING.md`.

## Common Problems

If Python is not found:

- macOS: install Python from https://www.python.org/downloads/
- Windows: install Python and check "Add Python to PATH"

If dataset files are missing:

- confirm the CSV files are under `data/amazon_reviews_multi_en/processed_3class/`
- confirm filenames match exactly: `train_3class.csv`, `validation_3class.csv`, `test_3class.csv`

If the dashboard opens but no results appear:

- run `python scripts/08_smoke_test.py` first
