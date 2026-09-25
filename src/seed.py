"""Reproducibility utilities: seed Python, NumPy, and PyTorch (CPU + CUDA)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all relevant RNGs for reproducible experiments.

    Seeds Python's ``random``, NumPy, and PyTorch (CPU and, if available, all
    CUDA devices). cuDNN is switched to deterministic mode, which trades a
    small amount of speed for exact reproducibility across runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the CUDA device if available, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
