"""Tests for Dice/IoU/precision/recall computation."""

from __future__ import annotations

import math

import torch

from src.metrics.segmentation import compute_metrics_from_logits
from src.losses.dice_loss import BCEDiceLoss, DiceLoss


def _logits_from_binary(mask: torch.Tensor, confidence: float = 10.0) -> torch.Tensor:
    """Turn a {0,1} mask into logits that sigmoid back to ~mask (for testing)."""
    return (mask * 2 - 1) * confidence


def test_perfect_prediction_gives_dice_and_iou_of_one() -> None:
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, :2, :2] = 1.0
    logits = _logits_from_binary(target)

    metrics = compute_metrics_from_logits(logits, target, threshold=0.5)

    assert math.isclose(metrics.dice.item(), 1.0, abs_tol=1e-4)
    assert math.isclose(metrics.iou.item(), 1.0, abs_tol=1e-4)
    assert math.isclose(metrics.precision.item(), 1.0, abs_tol=1e-4)
    assert math.isclose(metrics.recall.item(), 1.0, abs_tol=1e-4)


def test_no_overlap_gives_dice_and_iou_of_zero() -> None:
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, :2, :2] = 1.0
    pred_mask = torch.zeros(1, 1, 4, 4)
    pred_mask[0, 0, 2:, 2:] = 1.0
    logits = _logits_from_binary(pred_mask)

    metrics = compute_metrics_from_logits(logits, target, threshold=0.5)

    assert math.isclose(metrics.dice.item(), 0.0, abs_tol=1e-4)
    assert math.isclose(metrics.iou.item(), 0.0, abs_tol=1e-4)


def test_empty_prediction_and_target_defaults_to_one_not_nan() -> None:
    """Zero denominators (no positive pixels anywhere) must not produce NaN."""
    target = torch.zeros(1, 1, 4, 4)
    logits = _logits_from_binary(target)

    metrics = compute_metrics_from_logits(logits, target, threshold=0.5)

    assert not torch.isnan(metrics.dice).any()
    assert not torch.isnan(metrics.iou).any()
    assert metrics.dice.item() == 1.0
    assert metrics.iou.item() == 1.0


def test_dice_iou_relationship_matches_known_formula() -> None:
    """Dice = 2*IoU / (1 + IoU) for the same prediction/target pair."""
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, :2, :] = 1.0  # 8 positive pixels
    pred_mask = torch.zeros(1, 1, 4, 4)
    pred_mask[0, 0, :2, :3] = 1.0  # 6 predicted positive pixels, 6 overlap with target
    logits = _logits_from_binary(pred_mask)

    metrics = compute_metrics_from_logits(logits, target, threshold=0.5)
    iou = metrics.iou.item()
    dice = metrics.dice.item()

    assert math.isclose(dice, 2 * iou / (1 + iou), abs_tol=1e-4)


def test_partial_overlap_precision_and_recall() -> None:
    # target: 8 pixels positive; prediction: 6 pixels positive, all within target (TP=6, FP=0, FN=2)
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, :2, :] = 1.0
    pred_mask = torch.zeros(1, 1, 4, 4)
    pred_mask[0, 0, :2, :3] = 1.0
    logits = _logits_from_binary(pred_mask)

    metrics = compute_metrics_from_logits(logits, target, threshold=0.5)

    assert math.isclose(metrics.precision.item(), 1.0, abs_tol=1e-4)  # no false positives
    assert math.isclose(metrics.recall.item(), 6 / 8, abs_tol=1e-4)


def test_dice_loss_is_low_for_near_perfect_prediction() -> None:
    target = torch.zeros(2, 1, 8, 8)
    target[:, :, 2:6, 2:6] = 1.0
    logits = _logits_from_binary(target)

    loss = DiceLoss()(logits, target)

    assert loss.item() < 0.05


def test_bce_dice_loss_runs_and_is_finite() -> None:
    target = torch.randint(0, 2, (2, 1, 8, 8)).float()
    logits = torch.randn(2, 1, 8, 8)

    loss = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)(logits, target)

    assert torch.isfinite(loss)
    assert loss.item() >= 0
