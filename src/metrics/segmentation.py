"""Segmentation evaluation metrics: Dice, IoU, precision, recall.

All metrics are computed on binarized (thresholded) predictions against a
binary ground-truth mask, using the standard confusion-matrix quantities:

    TP = predicted gland AND actual gland
    FP = predicted gland AND actual background
    FN = predicted background AND actual gland

    Dice      = 2*TP / (2*TP + FP + FN)
    IoU       = TP / (TP + FP + FN)
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)

Dice vs. IoU: both measure overlap between prediction and ground truth, but
Dice is the harmonic mean of precision and recall and counts the
intersection twice, while IoU counts it once against the full union. For
any given prediction, IoU <= Dice, and the two are related exactly by
Dice = 2*IoU / (1 + IoU). Dice is more forgiving of small boundary
disagreements; IoU penalizes the same disagreement more heavily because it
shrinks a singly-counted intersection over the union rather than a
doubly-counted one.

Every denominator here is guarded against division by zero: a zero
denominator means there were no predicted and/or no actual positive pixels
at all, in which case the metric is defined as 1.0 (a trivially correct
result, e.g. an all-background prediction matching an all-background
target) instead of propagating NaN.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _confusion_counts(
    preds: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    preds = preds.float()
    targets = targets.float()
    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1 - targets)).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * targets).sum(dim=(1, 2, 3))
    return tp, fp, fn


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp(min=1e-12),
        torch.ones_like(denominator),
    )


@dataclass
class BatchMetrics:
    """Per-sample metric tensors, each of shape (N,) for a batch of N images."""

    dice: torch.Tensor
    iou: torch.Tensor
    precision: torch.Tensor
    recall: torch.Tensor

    def mean(self) -> dict[str, float]:
        return {
            "dice": self.dice.mean().item(),
            "iou": self.iou.mean().item(),
            "precision": self.precision.mean().item(),
            "recall": self.recall.mean().item(),
        }


def compute_metrics_from_logits(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> BatchMetrics:
    """Compute per-sample Dice/IoU/precision/recall from raw model logits.

    Args:
        logits: (N, 1, H, W) raw model output (pre-sigmoid).
        targets: (N, 1, H, W) binary ground-truth mask, values in {0, 1}.
        threshold: probability threshold used to binarize predictions.

    Returns:
        BatchMetrics with one value per sample, so callers can average
        within a batch or accumulate per-image values across a dataset.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    tp, fp, fn = _confusion_counts(preds, targets)

    dice = _safe_divide(2 * tp, 2 * tp + fp + fn)
    iou = _safe_divide(tp, tp + fp + fn)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)

    return BatchMetrics(dice=dice, iou=iou, precision=precision, recall=recall)


class RunningMetrics:
    """Accumulates per-image metric values across an epoch or evaluation run."""

    def __init__(self) -> None:
        self.dice: list[float] = []
        self.iou: list[float] = []
        self.precision: list[float] = []
        self.recall: list[float] = []
        self.image_ids: list[str] = []

    def update(self, batch_metrics: BatchMetrics, image_ids: list[str] | None = None) -> None:
        self.dice.extend(batch_metrics.dice.tolist())
        self.iou.extend(batch_metrics.iou.tolist())
        self.precision.extend(batch_metrics.precision.tolist())
        self.recall.extend(batch_metrics.recall.tolist())
        if image_ids is not None:
            self.image_ids.extend(image_ids)

    def average(self) -> dict[str, float]:
        n = len(self.dice)
        if n == 0:
            raise RuntimeError("RunningMetrics.average() called with no accumulated samples.")
        return {
            "dice": sum(self.dice) / n,
            "iou": sum(self.iou) / n,
            "precision": sum(self.precision) / n,
            "recall": sum(self.recall) / n,
        }

    def per_image(self) -> list[dict]:
        return [
            {
                "image_id": self.image_ids[i] if i < len(self.image_ids) else str(i),
                "dice": self.dice[i],
                "iou": self.iou[i],
                "precision": self.precision[i],
                "recall": self.recall[i],
            }
            for i in range(len(self.dice))
        ]
