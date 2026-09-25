#!/usr/bin/env python3
"""Run inference with a trained GlaS U-Net checkpoint.

Supports a single image or batch inference over a directory of images.
For each input image, saves the predicted binary mask and a red-overlay
visualization.

Usage:
    python scripts/predict.py --image path/to/image.png --checkpoint checkpoints/best_model.pth
    python scripts/predict.py --image-dir path/to/images/ --checkpoint checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch

from src.config import Config, load_config
from src.data.transforms import load_normalization_stats
from src.inference.predict import load_model, predict_mask, preprocess_image
from src.models.unet import UNet
from src.seed import get_device
from src.visualization.visualize import make_overlay

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def run_single(
    image_path: Path,
    model: UNet,
    device: torch.device,
    image_size: int,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
    output_dir: Path,
) -> None:
    tensor, original_size, original_rgb = preprocess_image(image_path, image_size, mean, std)
    mask = predict_mask(model, tensor, device, threshold, original_size=original_size)

    output_dir.mkdir(parents=True, exist_ok=True)

    mask_path = output_dir / f"{image_path.stem}_mask.png"
    cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))

    overlay = make_overlay(
        original_rgb.astype(np.float32) / 255.0, mask, color=(1.0, 0.0, 0.0), alpha=0.4
    )
    overlay_path = output_dir / f"{image_path.stem}_overlay.png"
    cv2.imwrite(str(overlay_path), cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    gland_fraction = mask.sum() / mask.size
    print(
        f"[{image_path.name}] gland pixels: {int(mask.sum())}/{mask.size} "
        f"({100 * gland_fraction:.1f}%) -> saved {mask_path.name}, {overlay_path.name}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=str, default=None, help="Path to a single input image.")
    parser.add_argument(
        "--image-dir", type=str, default=None, help="Path to a directory of input images (batch mode)."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to a trained model checkpoint (.pth)."
    )
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "config.yaml"))
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Defaults to the predictions_dir from config."
    )
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Provide either --image or --image-dir.")

    config: Config = load_config(args.config)
    config.ensure_directories()
    device = get_device()
    print(f"Using device: {device}")

    mean, std = load_normalization_stats(config.data.splits_dir / "normalization_stats.json")
    model = load_model(Path(args.checkpoint), config.model, device)
    output_dir = Path(args.output_dir) if args.output_dir else config.paths.predictions_dir

    if args.image:
        run_single(
            Path(args.image), model, device, config.data.image_size, mean, std,
            config.inference.threshold, output_dir,
        )
    else:
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not image_paths:
            raise FileNotFoundError(f"No supported image files found in {image_dir}")
        print(f"Running batch inference on {len(image_paths)} images ...")
        for image_path in image_paths:
            run_single(
                image_path, model, device, config.data.image_size, mean, std,
                config.inference.threshold, output_dir,
            )


if __name__ == "__main__":
    main()
