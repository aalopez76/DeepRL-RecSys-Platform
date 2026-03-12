"""Training callbacks — hooks for logging, checkpointing, early stopping."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TrainingCallback(ABC):
    """Base class for training callbacks."""

    @abstractmethod
    def on_step_end(self, step: int, metrics: dict[str, Any]) -> None:
        """Called at the end of each training step."""

    def on_train_begin(self) -> None:
        """Called at the start of training."""

    def on_train_end(self, metrics: dict[str, Any]) -> None:
        """Called at the end of training."""


class CheckpointCallback(TrainingCallback):
    """Saves model checkpoints at regular intervals (stub)."""

    def __init__(self, interval: int = 500, output_dir: str = "artifacts/checkpoints") -> None:
        self.interval = interval
        self.output_dir = output_dir

    def on_step_end(self, step: int, metrics: dict[str, Any]) -> None:
        if step % self.interval == 0:
            pass  # TODO: save checkpoint


class EarlyStoppingCallback(TrainingCallback):
    """Stops training when metric stops improving (stub)."""

    def __init__(self, patience: int = 10, metric: str = "loss") -> None:
        self.patience = patience
        self.metric = metric
        self._best: float | None = None
        self._wait = 0

    def on_step_end(self, step: int, metrics: dict[str, Any]) -> None:
        pass  # TODO: implement early stopping logic
