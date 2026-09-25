"""Tests for the U-Net model: forward pass and output shape/dtype."""

from __future__ import annotations

import torch

from src.models.unet import UNet


def test_forward_pass_output_shape() -> None:
    model = UNet(in_channels=3, out_channels=1, base_channels=8, depth=2)
    batch = torch.randn(2, 3, 64, 64)

    logits = model(batch)

    assert logits.shape == (2, 1, 64, 64)
    assert logits.dtype == torch.float32


def test_output_is_logits_not_probabilities() -> None:
    """The model must not apply a final sigmoid -- outputs should be able to
    fall outside [0, 1], unlike a probability map."""
    model = UNet(in_channels=3, out_channels=1, base_channels=8, depth=2)
    batch = torch.randn(4, 3, 64, 64) * 5.0  # push activations to a wider range

    logits = model(batch)

    assert torch.isfinite(logits).all()
    assert (logits.min() < 0) or (logits.max() > 1)


def test_variable_batch_and_channel_config() -> None:
    model = UNet(in_channels=3, out_channels=1, base_channels=16, depth=3)
    batch = torch.randn(1, 3, 96, 96)

    logits = model(batch)

    assert logits.shape == (1, 1, 96, 96)


def test_depth_one_smallest_config_runs() -> None:
    """Even the smallest valid depth should build and run without shape errors."""
    model = UNet(in_channels=3, out_channels=1, base_channels=4, depth=1)
    batch = torch.randn(1, 3, 32, 32)

    logits = model(batch)

    assert logits.shape == (1, 1, 32, 32)
