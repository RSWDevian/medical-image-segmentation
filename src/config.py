"""Typed configuration loading for the GlaS segmentation pipeline.

All paths declared in configs/config.yaml are relative to the repository
root; this module resolves them to absolute pathlib.Path objects so the
rest of the codebase never has to reason about the current working
directory or hard-code absolute paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DataConfig:
    raw_dir: Path
    processed_dir: Path
    splits_dir: Path
    image_size: int
    val_fraction: float
    split_seed: int


@dataclass
class ModelConfig:
    name: str
    in_channels: int
    out_channels: int
    base_channels: int
    depth: int


@dataclass
class LossConfig:
    name: str
    bce_weight: float
    dice_weight: float


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    num_workers: int
    seed: int
    early_stopping_patience: int
    amp: bool
    grad_clip_norm: float


@dataclass
class AugmentationConfig:
    enabled: bool
    horizontal_flip_prob: float
    vertical_flip_prob: float
    rotation_degrees: float
    scale_range: tuple[float, float]
    random_crop: bool
    crop_padding: int


@dataclass
class PathsConfig:
    checkpoints_dir: Path
    outputs_dir: Path
    predictions_dir: Path
    visualizations_dir: Path
    metrics_dir: Path


@dataclass
class InferenceConfig:
    threshold: float


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    training: TrainingConfig
    augmentation: AugmentationConfig
    paths: PathsConfig
    inference: InferenceConfig

    def ensure_directories(self) -> None:
        """Create all output/cache directories declared in the config."""
        for path in (
            self.data.processed_dir,
            self.data.splits_dir,
            self.paths.checkpoints_dir,
            self.paths.predictions_dir,
            self.paths.visualizations_dir,
            self.paths.metrics_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(config_path: str | Path, repo_root: Path | None = None) -> Config:
    """Load a YAML config file into a typed, path-resolved Config object.

    Args:
        config_path: Path to a YAML config file (e.g. configs/config.yaml).
        repo_root: Root directory used to resolve relative paths. Defaults
            to the repository root inferred from this file's location.

    Returns:
        A populated Config instance.

    Raises:
        FileNotFoundError: If config_path does not exist.
        KeyError: If a required section/key is missing from the YAML file.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    root = repo_root or REPO_ROOT
    with config_path.open("r") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    data_raw = raw["data"]
    data = DataConfig(
        raw_dir=_resolve(root, data_raw["raw_dir"]),
        processed_dir=_resolve(root, data_raw["processed_dir"]),
        splits_dir=_resolve(root, data_raw["splits_dir"]),
        image_size=int(data_raw["image_size"]),
        val_fraction=float(data_raw["val_fraction"]),
        split_seed=int(data_raw["split_seed"]),
    )

    model = ModelConfig(**raw["model"])
    loss = LossConfig(**raw["loss"])
    training = TrainingConfig(**raw["training"])

    aug_raw = raw["augmentation"]
    augmentation = AugmentationConfig(
        enabled=bool(aug_raw["enabled"]),
        horizontal_flip_prob=float(aug_raw["horizontal_flip_prob"]),
        vertical_flip_prob=float(aug_raw["vertical_flip_prob"]),
        rotation_degrees=float(aug_raw["rotation_degrees"]),
        scale_range=tuple(aug_raw["scale_range"]),
        random_crop=bool(aug_raw["random_crop"]),
        crop_padding=int(aug_raw["crop_padding"]),
    )

    paths_raw = raw["paths"]
    paths = PathsConfig(
        checkpoints_dir=_resolve(root, paths_raw["checkpoints_dir"]),
        outputs_dir=_resolve(root, paths_raw["outputs_dir"]),
        predictions_dir=_resolve(root, paths_raw["predictions_dir"]),
        visualizations_dir=_resolve(root, paths_raw["visualizations_dir"]),
        metrics_dir=_resolve(root, paths_raw["metrics_dir"]),
    )

    inference = InferenceConfig(threshold=float(raw["inference"]["threshold"]))

    return Config(
        data=data,
        model=model,
        loss=loss,
        training=training,
        augmentation=augmentation,
        paths=paths,
        inference=inference,
    )
