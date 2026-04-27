#!/usr/bin/env python3
"""Run the lightweight project flow without DistilBERT training."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEMO_TEXT = "The headphones are comfortable and the sound quality is great, but the battery life is shorter than expected."


def run_step(args: list[str]) -> None:
    """Run one project command with the current Python interpreter."""
    print("\n$", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    python = sys.executable
    run_step([python, "scripts/01_check_data.py"])
    run_step([python, "scripts/02_train_baseline.py", "--max-train-samples", "5000"])
    run_step([python, "scripts/04_evaluate_models.py"])
    run_step([python, "scripts/05_error_analysis.py", "--cases-per-file", "5"])
    run_step([python, "scripts/06_run_demo.py", "--text", DEMO_TEXT])
    print("\nSmoke test finished. Open the dashboard with:")
    print("streamlit run scripts/07_visual_app.py")


if __name__ == "__main__":
    main()
