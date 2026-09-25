#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the official 80-image GlaS test set.

The test set is discovered independently of the train/val split and is
never used for hyperparameter tuning -- this script is meant to be run
exactly once per model as a final, honest report of generalization
performance.

Usage:
    python scripts/evaluate.py --config configs/config.yaml \
        --checkpoint checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import GlasDataset
from src.data.split import load_split
from src.data.transforms import load_normalization_stats, make_eval_transform
from src.inference.predict import load_model
from src.losses.dice_loss import build_loss
from src.seed import get_device, set_seed
from src.training.validate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "config.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint .pth file. Defaults to <checkpoints_dir>/best_model.pth",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config.ensure_directories()
    set_seed(config.training.seed)
    device = get_device()
    print(f"Using device: {device}")

    test_split_path = config.data.splits_dir / "test.json"
    if not test_split_path.exists():
        raise FileNotFoundError(f"{test_split_path} not found. Run scripts/prepare_data.py first.")
    test_pairs = load_split(test_split_path)

    mean, std = load_normalization_stats(config.data.splits_dir / "normalization_stats.json")
    eval_transform = make_eval_transform(config.data.image_size, mean, std)
    test_dataset = GlasDataset(test_pairs, eval_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint else config.paths.checkpoints_dir / "best_model.pth"
    )
    model = load_model(checkpoint_path, config.model, device)
    loss_fn = build_loss(config.loss.name, config.loss.bce_weight, config.loss.dice_weight)

    print(f"Evaluating {checkpoint_path} on {len(test_pairs)} official GlaS test images ...")
    avg_loss, avg_metrics, running = evaluate(
        model, test_loader, loss_fn, device, config.inference.threshold
    )

    results = {
        "checkpoint": str(checkpoint_path),
        "num_test_images": len(test_pairs),
        "threshold": config.inference.threshold,
        "test_loss": avg_loss,
        "test_dice": avg_metrics["dice"],
        "test_iou": avg_metrics["iou"],
        "test_precision": avg_metrics["precision"],
        "test_recall": avg_metrics["recall"],
        "per_image": running.per_image(),
    }

    out_path = config.paths.metrics_dir / "test_metrics.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(
        f"Test results -- Dice: {avg_metrics['dice']:.4f} | IoU: {avg_metrics['iou']:.4f} | "
        f"Precision: {avg_metrics['precision']:.4f} | Recall: {avg_metrics['recall']:.4f}"
    )
    print(f"Full results (including per-image breakdown) written to {out_path}")


if __name__ == "__main__":
    main()
