#!/usr/bin/env python3
"""Print local or cloud environment information for model training."""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_to_rating.config import SPLIT_FILES  # noqa: E402


def package_status(name: str) -> str:
    """Return whether a package can be imported."""
    return "installed" if importlib.util.find_spec(name) is not None else "missing"


def print_torch_status() -> None:
    """Print torch accelerator details when torch is available."""
    if importlib.util.find_spec("torch") is None:
        print("torch: missing")
        return

    import torch

    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device_count: {torch.cuda.device_count()}")
        print(f"cuda_device_name: {torch.cuda.get_device_name(0)}")
    mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    print(f"mps_available: {mps_available}")


def main() -> None:
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"project_root: {ROOT}")
    print()

    for package in ["pandas", "sklearn", "matplotlib", "seaborn", "streamlit", "transformers", "accelerate", "datasets"]:
        print(f"{package}: {package_status(package)}")
    print_torch_status()
    print()

    print("data_files:")
    for split, path in SPLIT_FILES.items():
        print(f"- {split}: {'OK' if path.exists() else 'MISSING'} {path}")


if __name__ == "__main__":
    main()
