#!/usr/bin/env python3
"""Build standalone inference bundles for sharing trained models."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from textwrap import dedent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from review_to_rating.config import KAGGLE_DISTILBERT_MODELS_DIR, MODELS_DIR  # noqa: E402

RELEASE_ROOT = PROJECT_ROOT / "release"

BASELINE_FILES = {
    MODELS_DIR / "baseline_sentiment.joblib": Path("models/baseline_sentiment.joblib"),
    MODELS_DIR / "baseline_rating.joblib": Path("models/baseline_rating.joblib"),
}

DISTILBERT_TASKS = {
    "sentiment": KAGGLE_DISTILBERT_MODELS_DIR / "distilbert_sentiment",
    "rating": KAGGLE_DISTILBERT_MODELS_DIR / "distilbert_rating",
}

DISTILBERT_KEEP = {
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "training_metadata.json",
}

PREDICT_SCRIPT = dedent(
    '''
    #!/usr/bin/env python3
    from __future__ import annotations

    import argparse
    from pathlib import Path

    SCRIPT_DIR = Path(__file__).resolve().parent
    MODELS_DIR = SCRIPT_DIR / "models"


    def predict_baseline(text: str) -> dict[str, object]:
        import joblib

        sentiment_model = joblib.load(MODELS_DIR / "baseline_sentiment.joblib")
        rating_model = joblib.load(MODELS_DIR / "baseline_rating.joblib")
        sentiment = sentiment_model.predict([text])[0]
        rating = int(rating_model.predict([text])[0])
        return {"backend": "baseline", "sentiment": sentiment, "rating": rating}


    def predict_distilbert(text: str) -> dict[str, object]:
        import numpy as np
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        def run_one(model_dir: Path):
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            model.eval()
            inputs = tokenizer([text], truncation=True, padding=True, max_length=192, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
            return int(np.argmax(logits.detach().cpu().numpy(), axis=-1)[0]), model.config.id2label

        sentiment_id, sentiment_map = run_one(MODELS_DIR / "distilbert_sentiment")
        rating_id, rating_map = run_one(MODELS_DIR / "distilbert_rating")
        return {
            "backend": "distilbert",
            "sentiment": sentiment_map[sentiment_id],
            "rating": int(rating_map[rating_id]),
        }


    def main() -> None:
        parser = argparse.ArgumentParser(description="Predict sentiment and star rating from an English Amazon review.")
        parser.add_argument("--text", required=True, help="English review text")
        parser.add_argument("--backend", choices=["baseline", "distilbert", "auto"], default="auto")
        args = parser.parse_args()

        if args.backend == "baseline":
            result = predict_baseline(args.text)
        elif args.backend == "distilbert":
            result = predict_distilbert(args.text)
        else:
            if (MODELS_DIR / "distilbert_sentiment" / "config.json").exists() and (MODELS_DIR / "distilbert_rating" / "config.json").exists():
                result = predict_distilbert(args.text)
            else:
                result = predict_baseline(args.text)

        print(f"Backend: {result['backend']}")
        print(f"Sentiment prediction: {result['sentiment']}")
        print(f"Rating prediction: {result['rating']} stars")


    if __name__ == "__main__":
        main()
    '''
).strip() + "\n"


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dest: Path) -> None:
    ensure_exists(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_shared_files(bundle_dir: Path, requirements: str, readme: str) -> None:
    write_text(bundle_dir / "predict.py", PREDICT_SCRIPT)
    write_text(bundle_dir / "requirements.txt", requirements)
    write_text(bundle_dir / "README.md", readme)


def build_baseline_bundle() -> Path:
    bundle_dir = RELEASE_ROOT / "review_to_rating_runtime_baseline"
    reset_dir(bundle_dir)
    for src, rel_dest in BASELINE_FILES.items():
        copy_file(src, bundle_dir / rel_dest)

    requirements = dedent(
        '''
        joblib>=1.3
        scikit-learn>=1.3
        scipy>=1.10
        numpy>=1.24,<2
        '''
    ).lstrip()
    readme = dedent(
        '''
        # Review to Rating Runtime Bundle (Baseline)

        This is the smallest shareable package for direct inference.

        ## Included

        - sentiment model: `models/baseline_sentiment.joblib`
        - rating model: `models/baseline_rating.joblib`
        - one command-line runner: `predict.py`

        ## Setup

        ```bash
        python -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt
        ```

        On Windows PowerShell:

        ```powershell
        py -3 -m venv .venv
        .\\.venv\\Scripts\\Activate.ps1
        pip install -r requirements.txt
        ```

        ## Run

        ```bash
        python predict.py --backend baseline --text "The headphones are comfortable and the sound quality is great."
        ```
        '''
    ).lstrip()
    write_shared_files(bundle_dir, requirements, readme)
    return bundle_dir


def build_distilbert_bundle() -> Path:
    bundle_dir = RELEASE_ROOT / "review_to_rating_runtime_distilbert"
    reset_dir(bundle_dir)

    for src, rel_dest in BASELINE_FILES.items():
        copy_file(src, bundle_dir / rel_dest)

    for task, task_dir in DISTILBERT_TASKS.items():
        ensure_exists(task_dir)
        for name in DISTILBERT_KEEP:
            src = task_dir / name
            if src.exists():
                copy_file(src, bundle_dir / f"models/distilbert_{task}/{name}")

    requirements = dedent(
        '''
        numpy>=1.24,<2
        torch>=2.2
        transformers>=4.40
        safetensors>=0.4
        sentencepiece>=0.2
        joblib>=1.3
        scikit-learn>=1.3
        scipy>=1.10
        '''
    ).lstrip()
    readme = dedent(
        '''
        # Review to Rating Runtime Bundle (DistilBERT)

        This package contains already-trained DistilBERT models for direct inference.

        ## Included

        - sentiment model: `models/distilbert_sentiment/`
        - rating model: `models/distilbert_rating/`
        - baseline fallback models: `models/baseline_*.joblib`
        - one command-line runner: `predict.py`

        ## Environment

        - Python 3.10+
        - recommended: macOS/Linux/Windows with enough memory for Transformers inference

        ## Setup

        ```bash
        python -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt
        ```

        On Windows PowerShell:

        ```powershell
        py -3 -m venv .venv
        .\\.venv\\Scripts\\Activate.ps1
        pip install -r requirements.txt
        ```

        ## Run

        DistilBERT inference:

        ```bash
        python predict.py --backend distilbert --text "The headphones are comfortable and the sound quality is great."
        ```

        If the target machine is too weak or the full stack is inconvenient to install, baseline inference is still available:

        ```bash
        python predict.py --backend baseline --text "The headphones are comfortable and the sound quality is great."
        ```
        '''
    ).lstrip()
    write_shared_files(bundle_dir, requirements, readme)
    return bundle_dir


def zip_bundle(bundle_dir: Path) -> Path:
    archive_base = bundle_dir.parent / bundle_dir.name
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=bundle_dir.parent, base_dir=bundle_dir.name)
    return Path(archive_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build standalone inference bundles.")
    parser.add_argument("--variant", choices=["baseline", "distilbert", "all"], default="all")
    parser.add_argument("--zip", action="store_true", help="Also create zip archives")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    built: list[Path] = []
    if args.variant in {"baseline", "all"}:
        built.append(build_baseline_bundle())
    if args.variant in {"distilbert", "all"}:
        built.append(build_distilbert_bundle())

    for bundle_dir in built:
        print(bundle_dir)
        if args.zip:
            print(zip_bundle(bundle_dir))


if __name__ == "__main__":
    main()
