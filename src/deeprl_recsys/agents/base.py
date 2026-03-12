"""Abstract base class for all agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseAgent(ABC):
    """Base interface for recommendation agents.

    All agents (baselines + RL) must implement this interface so they
    can be used interchangeably by the training loop and serving runtime.
    """

    @abstractmethod
    def act(self, observation: dict[str, Any], candidates: list[int]) -> list[int]:
        """Select actions given an observation and candidate items.

        Args:
            observation: Context dictionary.
            candidates: List of candidate item IDs.

        Returns:
            Ordered list of selected item IDs.
        """

    def get_action_probabilities(
        self, observation: dict[str, Any], candidates: list[int]
    ) -> dict[int, float]:
        """Get the probability distribution over candidate actions.

        This is required for Off-Policy Evaluation (OPE) to compute
        importance weights.

        Args:
            observation: Context dictionary.
            candidates: List of candidate item IDs.

        Returns:
            Dictionary mapping candidate ID to its selection probability.
        """
        # Default fallback: deterministic policy (prob 1.0 for the chosen action)
        chosen = self.act(observation, candidates)
        probs = {c: 0.0 for c in candidates}
        if chosen:
            probs[chosen[0]] = 1.0
        return probs

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """Update the agent from a batch of experiences.

        Args:
            batch: Dictionary containing experience data.

        Returns:
            Dictionary of training metrics (e.g. loss).
        """
        return {}

    def save(self, path: str) -> None:
        """Serialize agent state to *path*."""

    def load(self, path: str) -> None:
        """Load agent state from *path*."""

    @property
    def name(self) -> str:
        """Return the agent's registered name."""
        return self.__class__.__name__
