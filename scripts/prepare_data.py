#!/usr/bin/env python3
"""Discover the GlaS dataset, build the deterministic train/val/test split,
verify image-mask correspondence, and compute training-set normalization
statistics.

Usage:
    python scripts/prepare_data.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
from tqdm import tqdm

from src.config import load_config
from src.data.split import (
    ImagePair,
    discover_official_test_pairs,
    discover_official_train_pairs,
    make_train_val_split,
    save_split,
)
from src.data.transforms import save_normalization_stats


def verify_pairs(pairs: list[ImagePair]) -> None:
    """Verify every image/mask pair is readable and has matching spatial dims."""
    for pair in tqdm(pairs, desc="verifying pairs"):
        image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image {pair.image_path}")
        mask = cv2.imread(str(pair.mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask {pair.mask_path}")
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image/mask size mismatch for {pair.image_id}: "
                f"image {image.shape[:2]} vs mask {mask.shape[:2]}"
            )


def compute_normalization_stats(
    pairs: list[ImagePair], image_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std over the training split, in [0, 1] pixel space.

    Computed only from the training subset (never validation or test), so
    normalization statistics cannot leak information from held-out data.
    """
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for pair in tqdm(pairs, desc="computing normalization stats"):
        image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float64) / 255.0

        pixel_sum += image.sum(axis=(0, 1))
        pixel_sq_sum += (image**2).sum(axis=(0, 1))
        pixel_count += image.shape[0] * image.shape[1]

    mean = pixel_sum / pixel_count
    variance = pixel_sq_sum / pixel_count - mean**2
    std = np.sqrt(np.clip(variance, 1e-12, None))
    return mean.astype(np.float32).reshape(1, 1, 3), std.astype(np.float32).reshape(1, 1, 3)


def gland_pixel_proportion(pairs: list[ImagePair]) -> float:
    """Fraction of pixels labeled as gland (foreground) across a set of masks."""
    fg = 0
    total = 0
    for pair in pairs:
        mask = cv2.imread(str(pair.mask_path), cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        binary = mask > 0
        fg += int(binary.sum())
        total += binary.size
    return fg / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "config.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    config.ensure_directories()

    if not config.data.raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {config.data.raw_dir}\n"
            "Download and extract the GlaS dataset there first "
            "(see README.md 'Installation' / 'Dataset' sections)."
        )

    print(f"Discovering official GlaS pairs under {config.data.raw_dir} ...")
    train_pairs = discover_official_train_pairs(config.data.raw_dir)
    test_pairs = discover_official_test_pairs(config.data.raw_dir)
    print(
        f"Found {len(train_pairs)} official training images and "
        f"{len(test_pairs)} official test images."
    )

    print("Verifying image/mask correspondence ...")
    verify_pairs(train_pairs)
    verify_pairs(test_pairs)
    print("All image/mask pairs verified (matching filenames and spatial dimensions).")

    train_subset, val_subset = make_train_val_split(
        train_pairs, config.data.val_fraction, config.data.split_seed
    )
    print(
        f"Deterministic split (seed={config.data.split_seed}): "
        f"{len(train_subset)} train / {len(val_subset)} val / {len(test_pairs)} test"
    )

    save_split(train_subset, config.data.splits_dir / "train.json")
    save_split(val_subset, config.data.splits_dir / "val.json")
    save_split(test_pairs, config.data.splits_dir / "test.json")
    print(f"Split files written to {config.data.splits_dir}")

    mean, std = compute_normalization_stats(train_subset, config.data.image_size)
    save_normalization_stats(mean, std, config.data.splits_dir / "normalization_stats.json")
    print(f"Normalization stats (train split only): mean={mean.flatten()} std={std.flatten()}")

    train_gland_frac = gland_pixel_proportion(train_subset)
    print(f"Gland pixel proportion in training split: {train_gland_frac:.4f}")


if __name__ == "__main__":
    main()
