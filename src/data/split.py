"""Discovery and reproducible train/val/test splitting for the GlaS dataset.

The official GlaS release ships 85 training images (train_1..train_85) and
80 test images split into testA (60) and testB (20). This module locates
those files under an arbitrary extraction layout -- different mirrors of the
dataset nest the files inside different folder names -- pairs each image
with its annotation mask, and produces a deterministic 68/17 train/validation
split of the official training set. The official test set is never touched
by this split: it is discovered separately and only used once, at final
evaluation time.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split

_IMAGE_RE = re.compile(r"^(train|testA|testB)_(\d+)\.bmp$", re.IGNORECASE)
_MASK_RE = re.compile(r"^(train|testA|testB)_(\d+)_anno\.bmp$", re.IGNORECASE)


@dataclass(frozen=True)
class ImagePair:
    """An RGB histology image paired with its ground-truth gland mask."""

    image_id: str
    subset: str  # "train", "testA", or "testB"
    image_path: Path
    mask_path: Path

    def to_json(self) -> dict:
        data = asdict(self)
        data["image_path"] = str(self.image_path)
        data["mask_path"] = str(self.mask_path)
        return data

    @staticmethod
    def from_json(data: dict) -> "ImagePair":
        return ImagePair(
            image_id=data["image_id"],
            subset=data["subset"],
            image_path=Path(data["image_path"]),
            mask_path=Path(data["mask_path"]),
        )


def _discover_pairs(raw_dir: Path, subset_names: tuple[str, ...]) -> list[ImagePair]:
    """Find all (image, mask) pairs for the given subset prefixes under raw_dir."""
    masks: dict[tuple[str, str], Path] = {}
    images: dict[tuple[str, str], Path] = {}

    for bmp_path in raw_dir.rglob("*.bmp"):
        name = bmp_path.name
        mask_match = _MASK_RE.match(name)
        if mask_match:
            subset, idx = mask_match.group(1), mask_match.group(2)
            if subset in subset_names:
                masks[(subset, idx)] = bmp_path
            continue
        image_match = _IMAGE_RE.match(name)
        if image_match:
            subset, idx = image_match.group(1), image_match.group(2)
            if subset in subset_names:
                images[(subset, idx)] = bmp_path

    pairs = []
    for key, image_path in images.items():
        mask_path = masks.get(key)
        if mask_path is None:
            subset, idx = key
            raise FileNotFoundError(
                f"No matching annotation mask found for image {image_path}. "
                f"Expected a file named '{subset}_{idx}_anno.bmp' under {raw_dir}."
            )
        subset, idx = key
        pairs.append(
            ImagePair(
                image_id=f"{subset}_{idx}",
                subset=subset,
                image_path=image_path,
                mask_path=mask_path,
            )
        )

    def sort_key(pair: ImagePair) -> tuple[str, int]:
        return (pair.subset, int(pair.image_id.split("_")[-1]))

    return sorted(pairs, key=sort_key)


def discover_official_train_pairs(raw_dir: Path) -> list[ImagePair]:
    """Return the 85 official GlaS training image/mask pairs.

    Raises:
        ValueError: If the raw dataset directory does not contain exactly
            85 train_*.bmp / train_*_anno.bmp pairs.
    """
    pairs = _discover_pairs(raw_dir, ("train",))
    if len(pairs) != 85:
        raise ValueError(
            f"Expected 85 official GlaS training images under {raw_dir}, found {len(pairs)}. "
            "Check that the dataset was extracted correctly (see README dataset setup)."
        )
    return pairs


def discover_official_test_pairs(raw_dir: Path) -> list[ImagePair]:
    """Return the 80 official GlaS test image/mask pairs (testA: 60, testB: 20).

    Raises:
        ValueError: If the raw dataset directory does not contain exactly
            80 test image/mask pairs across testA and testB.
    """
    pairs = _discover_pairs(raw_dir, ("testA", "testB"))
    if len(pairs) != 80:
        raise ValueError(
            f"Expected 80 official GlaS test images under {raw_dir}, found {len(pairs)}. "
            "Check that the dataset was extracted correctly (see README dataset setup)."
        )
    return pairs


def make_train_val_split(
    train_pairs: list[ImagePair], val_fraction: float, seed: int
) -> tuple[list[ImagePair], list[ImagePair]]:
    """Deterministically split the official training set into train/val subsets.

    A fixed random_state means the same seed always yields the same
    partition of image ids into train and validation -- this is what makes
    the split reproducible across machines and runs, not just within one
    process.
    """
    train_subset, val_subset = train_test_split(
        train_pairs, test_size=val_fraction, random_state=seed, shuffle=True
    )
    return train_subset, val_subset


def save_split(pairs: list[ImagePair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump([pair.to_json() for pair in pairs], f, indent=2)


def load_split(path: Path) -> list[ImagePair]:
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}. Run scripts/prepare_data.py first."
        )
    with path.open("r") as f:
        raw = json.load(f)
    return [ImagePair.from_json(item) for item in raw]
