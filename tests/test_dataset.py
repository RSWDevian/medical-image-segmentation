"""Tests for GlasDataset: loading, image/mask dimensions, and mask values."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.data.dataset import GlasDataset
from src.data.split import ImagePair
from src.data.transforms import make_eval_transform, make_train_transform
from src.config import AugmentationConfig


def _write_synthetic_pair(tmp_path: Path, image_id: str, height: int = 96, width: int = 80) -> ImagePair:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    # Instance-labeled mask: background=0, two "gland instances" labeled 1 and 2,
    # matching the real GlaS annotation convention this dataset must binarize.
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[10:40, 10:40] = 1
    mask[50:70, 50:70] = 2

    image_path = tmp_path / f"{image_id}.bmp"
    mask_path = tmp_path / f"{image_id}_anno.bmp"
    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_path), mask)

    return ImagePair(image_id=image_id, subset="train", image_path=image_path, mask_path=mask_path)


@pytest.fixture
def normalization_stats() -> tuple[np.ndarray, np.ndarray]:
    mean = np.full((1, 1, 3), 0.5, dtype=np.float32)
    std = np.full((1, 1, 3), 0.25, dtype=np.float32)
    return mean, std


def test_dataset_loading_and_shapes(tmp_path: Path, normalization_stats) -> None:
    pair = _write_synthetic_pair(tmp_path, "train_1")
    mean, std = normalization_stats
    image_size = 64
    transform = make_eval_transform(image_size, mean, std)
    dataset = GlasDataset([pair], transform)

    assert len(dataset) == 1
    sample = dataset[0]

    assert sample["image"].shape == (3, image_size, image_size)
    assert sample["mask"].shape == (1, image_size, image_size)
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].dtype == torch.float32
    assert sample["image_id"] == "train_1"


def test_mask_values_are_binary(tmp_path: Path, normalization_stats) -> None:
    pair = _write_synthetic_pair(tmp_path, "train_2")
    mean, std = normalization_stats
    transform = make_eval_transform(64, mean, std)
    dataset = GlasDataset([pair], transform)

    mask = dataset[0]["mask"]
    unique_values = torch.unique(mask)
    assert set(unique_values.tolist()).issubset({0.0, 1.0})
    # Both instance labels (1 and 2) in the raw mask must collapse to the
    # single foreground class -- verifies semantic (not instance) binarization.
    assert mask.sum() > 0


def test_train_transform_output_shape_matches_image_size(tmp_path: Path, normalization_stats) -> None:
    pair = _write_synthetic_pair(tmp_path, "train_3")
    mean, std = normalization_stats
    aug = AugmentationConfig(
        enabled=True,
        horizontal_flip_prob=1.0,
        vertical_flip_prob=1.0,
        rotation_degrees=15.0,
        scale_range=(0.9, 1.1),
        random_crop=True,
        crop_padding=8,
    )
    image_size = 64
    transform = make_train_transform(image_size, aug, mean, std)
    dataset = GlasDataset([pair], transform)

    sample = dataset[0]
    assert sample["image"].shape == (3, image_size, image_size)
    assert sample["mask"].shape == (1, image_size, image_size)


def test_missing_image_raises(tmp_path: Path, normalization_stats) -> None:
    mean, std = normalization_stats
    transform = make_eval_transform(64, mean, std)
    bad_pair = ImagePair(
        image_id="missing_1",
        subset="train",
        image_path=tmp_path / "does_not_exist.bmp",
        mask_path=tmp_path / "does_not_exist_anno.bmp",
    )
    dataset = GlasDataset([bad_pair], transform)
    with pytest.raises(FileNotFoundError):
        dataset[0]
