"""Segmentation loss functions: Dice loss and a combined BCE + Dice loss.

Why Dice loss:
    Binary cross-entropy treats every pixel independently and weighs all
    pixels equally. Measured directly on this dataset (scripts/prepare_data.py),
    gland pixels average ~50% of a GlaS tile overall, but per-image
    proportion actually ranges from ~11% to ~89% -- a meaningful number of
    individual training tiles are substantially foreground- or background-
    dominated even though the dataset is balanced in aggregate. On those
    tiles, and in general in medical segmentation tasks where the
    foreground class can be much smaller, BCE alone tends to be dominated
    by the easy majority class and gives a weak training signal for
    boundary pixels. Dice loss instead directly optimizes a differentiable
    approximation of the Dice coefficient (2*|A n B| / (|A| + |B|)), which
    is an overlap ratio that is naturally robust to class imbalance because
    it normalizes by the size of both the prediction and the target rather
    than by the total pixel count.

Why combine BCE + Dice:
    Dice loss alone has noisy, unstable gradients early in training (when
    predictions are near-random, both the numerator and denominator of the
    Dice ratio are small and volatile). BCE provides a smooth, well-behaved
    per-pixel gradient that stabilizes early optimization, while Dice
    directly pushes for good region overlap. Using both losses together
    (a simple weighted sum) is a standard, well-documented combination for
    binary medical image segmentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation, computed from logits.

    Dice = 2 * sum(p * t) / (sum(p) + sum(t) + eps)
    Loss  = 1 - Dice

    `p` is the sigmoid of the model's logits (a soft/probabilistic mask,
    not thresholded), so the loss is differentiable everywhere. `eps` keeps
    the loss well-defined when both prediction and target are all-zero.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Weighted sum of BCEWithLogitsLoss and DiceLoss.

    Args:
        bce_weight: weight applied to the BCE term.
        dice_weight: weight applied to the Dice term.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def build_loss(name: str, bce_weight: float = 0.5, dice_weight: float = 0.5) -> nn.Module:
    """Factory used by the training script to select the loss from config.yaml."""
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "dice":
        return DiceLoss()
    if name == "bce_dice":
        return BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)
    raise ValueError(f"Unknown loss name '{name}'. Expected one of: bce, dice, bce_dice.")
