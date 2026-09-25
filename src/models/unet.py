"""U-Net for binary gland segmentation, implemented from scratch in PyTorch.

Reference: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
Image Segmentation" (2015), https://arxiv.org/abs/1505.04597

Architecture summary:
  Encoder   -- repeated (conv -> BN -> ReLU) x2 blocks, each followed by a
               2x2 max-pool that halves spatial resolution while doubling
               channel depth. This builds a feature hierarchy from local
               texture (early layers) to gland-scale shape/context (deep
               layers).
  Bottleneck-- the deepest, lowest-resolution conv block. It sees the
               largest receptive field and captures global context about
               tissue structure.
  Decoder   -- repeated transposed-convolution upsampling followed by a
               (conv -> BN -> ReLU) x2 block, mirroring the encoder in
               reverse to recover full input resolution.
  Skip connections -- at each decoder stage, the upsampled feature map is
               concatenated (channel-wise) with the encoder feature map of
               matching resolution *before* upsampling. Pooling in the
               encoder discards precise spatial detail (exact gland
               boundaries); skip connections hand that detail back to the
               decoder directly, which is what lets U-Net produce sharp,
               well-localized segmentation boundaries instead of blurry
               ones.

The final layer is a 1x1 convolution producing a single output channel of
raw logits (no sigmoid). Logits are used directly with
BCEWithLogitsLoss for numerical stability; sigmoid is applied only at
inference/metric time (see src/metrics/segmentation.py and
src/inference/predict.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm2d -> ReLU) x2.

    BatchNorm stabilizes and speeds up training and reduces the model's
    sensitivity to the exact input intensity normalization used -- useful
    here since H&E stained images have significant color variation across
    slides.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Encoder stage: max-pool downsample followed by a DoubleConv block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Decoder stage: transposed-conv upsample, concat skip connection, DoubleConv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Guard against off-by-one spatial mismatches (can occur when the
        # input size is not perfectly divisible by 2**depth) by center-
        # cropping/padding the upsampled tensor to match the skip tensor.
        diff_h = skip.shape[2] - x.shape[2]
        diff_w = skip.shape[3] - x.shape[3]
        if diff_h != 0 or diff_w != 0:
            x = nn.functional.pad(
                x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2]
            )

        x = torch.cat([skip, x], dim=1)  # skip connection: fuse encoder detail with decoder context
        return self.conv(x)


class UNet(nn.Module):
    """Configurable-depth U-Net for binary semantic segmentation.

    Args:
        in_channels: number of input image channels (3 for RGB).
        out_channels: number of output channels (1 for binary segmentation;
            the single channel holds raw logits, not a probability).
        base_channels: number of feature channels produced by the first
            encoder block. Doubles at every subsequent encoder stage.
        depth: number of downsampling stages between the input and the
            bottleneck (the original U-Net paper uses depth=4).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        # ----- Encoder -----
        # First block operates directly on the input; each subsequent Down
        # block halves spatial resolution and doubles channel count.
        encoder_channels = [base_channels * (2**i) for i in range(depth)]
        self.in_conv = DoubleConv(in_channels, encoder_channels[0])
        self.down_blocks = nn.ModuleList(
            [
                Down(encoder_channels[i], encoder_channels[i + 1])
                for i in range(depth - 1)
            ]
        )

        # ----- Bottleneck -----
        # Deepest, lowest-resolution representation; largest receptive field.
        bottleneck_channels = base_channels * (2**depth)
        self.bottleneck = Down(encoder_channels[-1], bottleneck_channels)

        # ----- Decoder -----
        # Mirrors the encoder: each Up block halves channel count and
        # doubles spatial resolution, fusing in the matching skip connection.
        up_in_channels = [bottleneck_channels] + [
            encoder_channels[i] for i in reversed(range(1, depth))
        ]
        skip_channels = list(reversed(encoder_channels))
        self.up_blocks = nn.ModuleList(
            [
                Up(up_in_channels[i], skip_channels[i], skip_channels[i])
                for i in range(depth)
            ]
        )

        # ----- Final projection -----
        # 1x1 conv maps decoder features to raw logits, one channel per class.
        # No activation here: BCEWithLogitsLoss expects logits, and sigmoid
        # is applied downstream only where a probability is actually needed.
        self.out_conv = nn.Conv2d(encoder_channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.in_conv(x)
        skips.append(x)

        for down in self.down_blocks:
            x = down(x)
            skips.append(x)

        x = self.bottleneck(x)

        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip)

        return self.out_conv(x)
