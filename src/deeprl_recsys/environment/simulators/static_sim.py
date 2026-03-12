"""Static simulator — fixed-table reward model."""

from __future__ import annotations

from typing import Any

from deeprl_recsys.environment.simulators.base_sim import BaseSimulator


class StaticSimulator(BaseSimulator):
    """Simulates user responses using fixed probabilities."""

    def __init__(self, click_prob: float = 0.05, **kwargs: Any) -> None:
        self.click_prob = click_prob

    def simulate_response(self, user: dict[str, Any], item: int) -> float:
        import numpy as np
        return float(np.random.random() < self.click_prob)
