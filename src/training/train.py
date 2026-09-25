"""Training loop for U-Net gland segmentation.

Handles optimizer construction, mixed-precision training (automatically
enabled only on CUDA), per-epoch validation, best-checkpoint selection on
validation Dice, and early stopping.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics.segmentation import RunningMetrics, compute_metrics_from_logits
from src.training.early_stopping import EarlyStopping
from src.training.validate import evaluate


def _build_optimizer(
    model: torch.nn.Module, name: str, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{name}'. Expected 'adam' or 'adamw'.")


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    grad_clip_norm: float,
    threshold: float = 0.5,
) -> tuple[float, dict[str, float]]:
    model.train()
    running = RunningMetrics()
    total_loss = 0.0
    num_samples = 0

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = loss_fn(logits, masks)

        scaler.scale(loss).backward()
        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size

        with torch.no_grad():
            batch_metrics = compute_metrics_from_logits(logits.float(), masks, threshold=threshold)
            running.update(batch_metrics)

    avg_loss = total_loss / num_samples
    return avg_loss, running.average()


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    optimizer_name: str,
    early_stopping_patience: int,
    amp_enabled: bool,
    grad_clip_norm: float,
    checkpoint_dir: Path,
    threshold: float = 0.5,
) -> dict[str, list[float]]:
    """Run the full training loop with per-epoch validation and checkpointing.

    The best checkpoint (by validation Dice) is saved to
    `checkpoint_dir/best_model.pth`; the most recent epoch is always saved
    to `checkpoint_dir/last_model.pth`. Training stops early if validation
    Dice does not improve for `early_stopping_patience` consecutive epochs.

    Returns:
        A history dict with one list per tracked quantity (train_loss,
        val_loss, val_dice, val_iou, val_precision, val_recall), one entry
        per epoch actually run. Also written to `checkpoint_dir/history.json`.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    optimizer = _build_optimizer(model, optimizer_name, learning_rate, weight_decay)
    amp_enabled = amp_enabled and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    early_stopping = EarlyStopping(patience=early_stopping_patience)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_precision": [],
        "val_recall": [],
    }

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler, amp_enabled, grad_clip_norm, threshold
        )
        val_loss, val_metrics, _ = evaluate(model, val_loader, loss_fn, device, threshold)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_metrics["dice"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])

        elapsed = time.time() - epoch_start
        print(
            f"epoch {epoch:03d}/{epochs} | "
            f"train_loss {train_loss:.4f} train_dice {train_metrics['dice']:.4f} | "
            f"val_loss {val_loss:.4f} val_dice {val_metrics['dice']:.4f} val_iou {val_metrics['iou']:.4f} | "
            f"{elapsed:.1f}s"
        )

        is_best = early_stopping.step(val_metrics["dice"])
        if is_best:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_dice": val_metrics["dice"],
                    "val_iou": val_metrics["iou"],
                },
                checkpoint_dir / "best_model.pth",
            )

        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict()},
            checkpoint_dir / "last_model.pth",
        )

        if early_stopping.should_stop:
            print(
                f"Early stopping triggered after epoch {epoch} "
                f"(no val_dice improvement for {early_stopping_patience} epochs)."
            )
            break

    with (checkpoint_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)

    return history
