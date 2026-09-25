"""Joint image/mask transforms for GlaS segmentation.

All spatial transforms (flip, rotation, scale, crop) are applied identically
to the image and its mask so the two stay pixel-aligned. Interpolation is
deliberately different per tensor type:

  - Images use bilinear interpolation: RGB intensities are continuous, so
    smooth resampling is appropriate and standard.
  - Masks use nearest-neighbor interpolation: a mask is a categorical label
    map ({0, 1}). Bilinear/bicubic interpolation would invent fractional
    "in-between" values (e.g. 0.4) that do not correspond to a real class
    and would corrupt the ground truth at every gland boundary.

Augmentations are intentionally limited to transformations that are
physically reasonable for a histology tile: glands have no canonical
orientation, so flips and rotations do not distort the label semantics the
way they might for, say, natural images of text or faces. No color/stain
augmentation is applied in this baseline (see README "Future Work").
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from src.config import AugmentationConfig

Transform = Callable[[np.ndarray, np.ndarray], tuple[torch.Tensor, torch.Tensor]]


def _resize(image: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return image, mask


def _random_flip(
    image: np.ndarray, mask: np.ndarray, h_prob: float, v_prob: float
) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < h_prob:
        image = np.ascontiguousarray(image[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
    if random.random() < v_prob:
        image = np.ascontiguousarray(image[::-1, :])
        mask = np.ascontiguousarray(mask[::-1, :])
    return image, mask


def _random_rotate_scale(
    image: np.ndarray, mask: np.ndarray, max_degrees: float, scale_range: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    angle = random.uniform(-max_degrees, max_degrees)
    scale = random.uniform(*scale_range)
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    image = cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    mask = cv2.warpAffine(
        mask,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return image, mask


def _random_crop_with_padding(
    image: np.ndarray, mask: np.ndarray, size: int, padding: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pad then randomly crop back to `size`, giving mild translation jitter."""
    image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REFLECT_101)
    mask = cv2.copyMakeBorder(
        mask, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0
    )
    h, w = image.shape[:2]
    top = random.randint(0, h - size)
    left = random.randint(0, w - size)
    image = image[top : top + size, left : left + size]
    mask = mask[top : top + size, left : left + size]
    return image, mask


def _to_tensor(
    image: np.ndarray, mask: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    image = image.astype(np.float32) / 255.0
    image = (image - mean) / std
    image_t = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
    mask_t = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0).contiguous()
    return image_t, mask_t


def make_train_transform(
    image_size: int, aug: AugmentationConfig, mean: np.ndarray, std: np.ndarray
) -> Transform:
    """Build the training transform: resize -> augment -> normalize -> tensor."""

    def transform(image: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = _resize(image, mask, image_size)
        if aug.enabled:
            image, mask = _random_flip(image, mask, aug.horizontal_flip_prob, aug.vertical_flip_prob)
            image, mask = _random_rotate_scale(image, mask, aug.rotation_degrees, aug.scale_range)
            if aug.random_crop:
                image, mask = _random_crop_with_padding(image, mask, image_size, aug.crop_padding)
        return _to_tensor(image, mask, mean, std)

    return transform


def make_eval_transform(image_size: int, mean: np.ndarray, std: np.ndarray) -> Transform:
    """Build the validation/test/inference transform: resize -> normalize -> tensor.

    No randomness: evaluation must be deterministic given the same input.
    """

    def transform(image: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = _resize(image, mask, image_size)
        return _to_tensor(image, mask, mean, std)

    return transform


def save_normalization_stats(mean: np.ndarray, std: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump({"mean": mean.flatten().tolist(), "std": std.flatten().tolist()}, f, indent=2)


def load_normalization_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Normalization stats not found at {path}. Run scripts/prepare_data.py first."
        )
    with path.open("r") as f:
        raw = json.load(f)
    mean = np.array(raw["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.array(raw["std"], dtype=np.float32).reshape(1, 1, 3)
    return mean, std
