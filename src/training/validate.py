"""Evaluation loop shared by per-epoch validation and final test-set evaluation."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.metrics.segmentation import RunningMetrics, compute_metrics_from_logits


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[float, dict[str, float], RunningMetrics]:
    """Run the model over `loader` in eval mode and compute loss + metrics.

    Args:
        model: model to evaluate (switched into eval() mode internally).
        loader: DataLoader yielding {"image", "mask", "image_id"} batches.
        loss_fn: loss used only for reporting (not backpropagated here).
        device: device to run inference on.
        threshold: probability threshold for binarizing predictions.

    Returns:
        (average_loss, averaged_metrics, running_metrics). `running_metrics`
        additionally holds per-image Dice/IoU/precision/recall, used by
        scripts/evaluate.py to write a per-image breakdown for the test set.
    """
    model.eval()
    running = RunningMetrics()
    total_loss = 0.0
    num_samples = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        image_ids = batch["image_id"]

        logits = model(images)
        loss = loss_fn(logits, masks)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size

        batch_metrics = compute_metrics_from_logits(logits, masks, threshold=threshold)
        running.update(batch_metrics, image_ids=list(image_ids))

    avg_loss = total_loss / num_samples
    return avg_loss, running.average(), running
