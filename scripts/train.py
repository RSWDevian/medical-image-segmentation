#!/usr/bin/env python3
"""Train the U-Net gland segmentation model.

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --epochs 2   # smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import GlasDataset
from src.data.split import load_split
from src.data.transforms import load_normalization_stats, make_eval_transform, make_train_transform
from src.losses.dice_loss import build_loss
from src.models.unet import UNet
from src.seed import get_device, set_seed
from src.training.train import train_model
from src.visualization.visualize import plot_training_curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "config.yaml"))
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override config epochs (useful for smoke tests)."
    )
    parser.add_argument(
        "--run-name", type=str, default=None, help="Optional subfolder under checkpoints/ for this run."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config.ensure_directories()
    set_seed(config.training.seed)
    device = get_device()
    print(f"Using device: {device}")

    for name in ("train", "val"):
        split_path = config.data.splits_dir / f"{name}.json"
        if not split_path.exists():
            raise FileNotFoundError(
                f"{split_path} not found. Run scripts/prepare_data.py before training."
            )

    train_pairs = load_split(config.data.splits_dir / "train.json")
    val_pairs = load_split(config.data.splits_dir / "val.json")
    mean, std = load_normalization_stats(config.data.splits_dir / "normalization_stats.json")

    train_transform = make_train_transform(config.data.image_size, config.augmentation, mean, std)
    eval_transform = make_eval_transform(config.data.image_size, mean, std)

    train_dataset = GlasDataset(train_pairs, train_transform)
    val_dataset = GlasDataset(val_pairs, eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Train images: {len(train_dataset)} | Val images: {len(val_dataset)}")

    model = UNet(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        base_channels=config.model.base_channels,
        depth=config.model.depth,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: UNet(base_channels={config.model.base_channels}, depth={config.model.depth}) "
          f"-- {num_params:,} parameters")

    loss_fn = build_loss(config.loss.name, config.loss.bce_weight, config.loss.dice_weight)
    print(f"Loss: {config.loss.name} (bce_weight={config.loss.bce_weight}, dice_weight={config.loss.dice_weight})")

    checkpoint_dir = config.paths.checkpoints_dir
    if args.run_name:
        checkpoint_dir = checkpoint_dir / args.run_name

    epochs = args.epochs if args.epochs is not None else config.training.epochs

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        epochs=epochs,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        optimizer_name=config.training.optimizer,
        early_stopping_patience=config.training.early_stopping_patience,
        amp_enabled=config.training.amp,
        grad_clip_norm=config.training.grad_clip_norm,
        checkpoint_dir=checkpoint_dir,
        threshold=config.inference.threshold,
    )

    curves_path = config.paths.metrics_dir / "training_curves.png"
    plot_training_curves(history, curves_path)
    print(f"Training curves saved to {curves_path}")

    best_dice = max(history["val_dice"]) if history["val_dice"] else float("nan")
    print(f"Best validation Dice achieved: {best_dice:.4f}")
    print(f"Best checkpoint: {checkpoint_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
