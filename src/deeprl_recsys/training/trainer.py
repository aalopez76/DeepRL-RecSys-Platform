"""Training loop — pure fit/train_step logic.

No CLI, no hard-coded paths. Everything enters via config.
"""

from __future__ import annotations

from typing import Any

from deeprl_recsys.agents.base import BaseAgent
from deeprl_recsys.core.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    """Runs the training loop for a given agent.

    Args:
        agent: The agent to train.
        max_steps: Maximum training steps.
        eval_interval: Steps between evaluations.
        callbacks: Optional list of callback objects.
    """

    def __init__(
        self,
        agent: BaseAgent,
        max_steps: int = 1000,
        eval_interval: int = 100,
        callbacks: list[Any] | None = None,
    ) -> None:
        self.agent = agent
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.callbacks = callbacks or []

    def fit(self, data: Any) -> dict[str, Any]:
        """Run the full training loop.

        Args:
            data: Training data (format depends on agent type).

        Returns:
            Dictionary of final training metrics.
        """
        metrics: dict[str, Any] = {"steps_completed": 0}
        logger.info("training_start", agent=self.agent.name, max_steps=self.max_steps)

        for step in range(1, self.max_steps + 1):
            step_metrics = self.train_step(data, step)
            metrics.update(step_metrics)
            metrics["steps_completed"] = step

            # Structured per-step logging for analysis
            logger.info(
                "train_step",
                step=step,
                **{k: round(v, 6) if isinstance(v, float) else v for k, v in step_metrics.items()},
            )

            if step % self.eval_interval == 0:
                logger.info("eval_checkpoint", step=step, metrics=step_metrics)

        logger.info("training_complete", total_steps=self.max_steps)
        return metrics

    def train_step(self, data: Any, step: int) -> dict[str, float]:
        """Execute a single training step.

        Args:
            data: Training data batch.
            step: Current step number.

        Returns:
            Step-level metrics.
        """
        return self.agent.update({"data": data, "step": step})
