"""Single-image inference for the trained GlaS U-Net.

Preprocessing here intentionally mirrors src/data/transforms.py's eval
transform (resize with bilinear interpolation, then normalize) so that
inference-time behavior matches what the model saw during validation.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from src.config import ModelConfig
from src.models.unet import UNet


def load_model(checkpoint_path: Path, model_config: ModelConfig, device: torch.device) -> UNet:
    """Instantiate a UNet from config and load trained weights from a checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Train a model first with scripts/train.py."
        )
    model = UNet(
        in_channels=model_config.in_channels,
        out_channels=model_config.out_channels,
        base_channels=model_config.base_channels,
        depth=model_config.depth,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def preprocess_image(
    image_path: Path, image_size: int, mean: np.ndarray, std: np.ndarray
) -> tuple[torch.Tensor, tuple[int, int], np.ndarray]:
    """Load and preprocess an RGB image for inference.

    Returns:
        image_tensor: shape (1, 3, image_size, image_size), normalized.
        original_size: (H, W) of the source image, for resizing predictions back.
        original_rgb: the source image as uint8 RGB, for visualization.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image at {image_path} (unsupported or corrupt file).")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    original_size = (rgb.shape[0], rgb.shape[1])

    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - mean) / std
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, original_size, rgb


@torch.no_grad()
def predict_mask(
    model: UNet,
    image_tensor: torch.Tensor,
    device: torch.device,
    threshold: float,
    original_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Run the model, apply sigmoid then threshold at `threshold`, return a binary mask.

    If `original_size` is given and differs from the model's working
    resolution, the mask is resized back with nearest-neighbor
    interpolation -- masks are categorical, so any other interpolation
    would invent fractional class values.
    """
    image_tensor = image_tensor.to(device)
    logits = model(image_tensor)
    probs = torch.sigmoid(logits)
    mask = (probs >= threshold).squeeze().float().cpu().numpy().astype(np.uint8)

    if original_size is not None and tuple(mask.shape) != tuple(original_size):
        mask = cv2.resize(
            mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST
        )
    return mask
