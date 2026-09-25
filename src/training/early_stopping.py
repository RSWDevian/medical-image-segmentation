"""Early stopping based on a monitored validation metric (higher is better)."""

from __future__ import annotations

from typing import Optional


class EarlyStopping:
    """Stops training when a monitored metric has not improved for `patience` epochs.

    An "improvement" means `metric > best_metric + min_delta`. This project
    monitors validation Dice, so higher is always better here.
    """

    def __init__(self, patience: int, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric: Optional[float] = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """Update state with the latest epoch's metric value.

        Returns:
            True if `metric` is a new best (the caller should checkpoint),
            False otherwise.
        """
        if self.best_metric is None or metric > self.best_metric + self.min_delta:
            self.best_metric = metric
            self.num_bad_epochs = 0
            return True

        self.num_bad_epochs += 1
        if self.num_bad_epochs >= self.patience:
            self.should_stop = True
        return False
