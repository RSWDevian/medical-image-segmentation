"""PyTorch Dataset for GlaS gland segmentation image/mask pairs."""

from __future__ import annotations

import cv2
import numpy as np
from torch.utils.data import Dataset

from src.data.split import ImagePair
from src.data.transforms import Transform


class GlasDataset(Dataset):
    """Loads GlaS histology images and their binary gland masks.

    The original GlaS annotation masks are instance-labeled: background is
    0 and each individual gland has its own positive integer label
    (1, 2, 3, ...). This project performs semantic (not instance)
    segmentation, so every nonzero label is collapsed to the single
    foreground class "gland" (1) during loading.
    """

    def __init__(self, pairs: list[ImagePair], transform: Transform) -> None:
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]

        image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {pair.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(pair.mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask at {pair.mask_path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        # Binarize: background = 0, any gland instance label -> 1.
        mask = (mask > 0).astype(np.uint8)

        image_t, mask_t = self.transform(image, mask)

        return {"image": image_t, "mask": mask_t, "image_id": pair.image_id}
