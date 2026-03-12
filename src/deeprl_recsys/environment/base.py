"""Base environment interface for RecSys environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEnvironment(ABC):
    """Abstract base for recommendation environments."""

    @abstractmethod
    def reset(self) -> dict[str, Any]:
        """Reset the environment and return initial observation."""

    @abstractmethod
    def step(self, action: int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Take an action and return (observation, reward, done, info)."""
