"""Base simulator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSimulator(ABC):
    """Abstract base for user behaviour simulators."""

    @abstractmethod
    def simulate_response(self, user: dict[str, Any], item: int) -> float:
        """Simulate user response (reward) for a given item."""
