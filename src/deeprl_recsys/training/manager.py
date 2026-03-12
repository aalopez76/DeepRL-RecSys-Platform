"""Training session manager — orchestrates config, agent, env, and callbacks."""

from __future__ import annotations

from typing import Any

from deeprl_recsys.core.logging import get_logger
from deeprl_recsys.core.registry import create
from deeprl_recsys.core.seeding import set_global_seed
from deeprl_recsys.training.trainer import Trainer

logger = get_logger(__name__)


class TrainingManager:
    """Orchestrates a training session from a resolved config.

    Args:
        config: Fully-resolved platform config dictionary.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def run(self) -> dict[str, Any]:
        """Execute the complete training session.

        Returns:
            Final training metrics.
        """
        seed = self.config.get("seed", 42)
        set_global_seed(seed)
        logger.info("session_start", seed=seed)

        agent_cfg = self.config.get("agent", {})
        agent = create("agents", agent_cfg.get("name", "random"), **agent_cfg.get("hyperparams", {}))

        trainer = Trainer(
            agent=agent,
            max_steps=self.config.get("training", {}).get("max_steps", 1000),
            eval_interval=self.config.get("training", {}).get("eval_interval", 100),
        )

        # TODO: load actual data via data_pipeline
        metrics = trainer.fit(data=None)
        logger.info("session_complete", metrics=metrics)
        return metrics
