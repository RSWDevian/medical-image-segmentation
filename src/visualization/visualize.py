"""Matplotlib visualization utilities: prediction panels and training curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def denormalize_image(image: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Undo z-score normalization, returning an RGB image in [0, 1] for display.

    Args:
        image: (H, W, 3) float array in normalized space.
        mean, std: per-channel stats used during normalization.
    """
    mean = mean.reshape(1, 1, 3)
    std = std.reshape(1, 1, 3)
    image = image * std + mean
    return np.clip(image, 0.0, 1.0)


def make_overlay(
    image: np.ndarray, mask: np.ndarray, color: tuple[float, float, float], alpha: float = 0.4
) -> np.ndarray:
    """Blend a binary mask onto an RGB image in [0, 1] as a translucent color overlay."""
    overlay = image.copy()
    color_layer = np.empty_like(image)
    color_layer[..., 0] = color[0]
    color_layer[..., 1] = color[1]
    color_layer[..., 2] = color[2]
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (1 - alpha) * image[mask_bool] + alpha * color_layer[mask_bool]
    return overlay


def plot_prediction_panel(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    image_id: str,
    save_path: Path,
) -> None:
    """Save a 4-panel figure: Original | Ground Truth | Prediction | Overlay.

    The overlay shows ground truth in green and the model's prediction in
    red (overlapping/correct regions read as yellow), which makes both
    false positives and false negatives immediately visible.
    """
    overlay = make_overlay(image, ground_truth, color=(0.0, 1.0, 0.0), alpha=0.35)
    overlay = make_overlay(overlay, prediction, color=(1.0, 0.0, 0.0), alpha=0.35)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = [
        (image, "Original", None),
        (ground_truth, "Ground Truth", "gray"),
        (prediction, "Prediction", "gray"),
        (overlay, "Overlay (GT=green, Pred=red)", None),
    ]
    for ax, (data, title, cmap) in zip(axes, panels):
        ax.imshow(data, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(image_id)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: dict, save_path: Path) -> None:
    """Plot train/val loss and val Dice/IoU curves from a training history dict."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="train loss")
    axes[0].plot(epochs, history["val_loss"], label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["val_dice"], label="val Dice")
    axes[1].plot(epochs, history["val_iou"], label="val IoU")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Validation Dice / IoU")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
