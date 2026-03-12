"""Pipeline: train — resolve config, instantiate agent, run training loop.

Orchestrates:
1. Set seed for reproducibility
2. Instantiate agent based on config
3. Run training iterations (simulated for baselines)
4. Save checkpoints and logs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from deeprl_recsys.agents.baselines import GreedyAgent, RandomAgent, TopKAgent
from deeprl_recsys.core import registry
from deeprl_recsys.core.logging import get_logger
from deeprl_recsys.core.seeding import set_global_seed

logger = get_logger(__name__)


def run_train(config: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    """Execute the training pipeline.

    Args:
        config: Resolved platform config.  Expected keys:

            - ``seed`` — global random seed
            - ``agent.name`` — agent to instantiate
            - ``agent.params`` — agent constructor kwargs
            - ``training.max_steps`` — number of training steps
            - ``training.checkpoint_dir`` — directory for checkpoints

    Returns:
        Training metrics: ``agent_name``, ``steps_completed``,
        ``model_path``, and ``metrics``.
    """
    seed = config.get("seed", 42)
    set_global_seed(seed)

    agent_cfg = config.get("agent", {})
    training_cfg = config.get("training", {})

    agent_name = agent_cfg.get("name", "random")
    agent_params = agent_cfg.get("params", {})
    max_steps = training_cfg.get("max_steps", 5)
    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "artifacts/checkpoints"))

    logger.info("train_start", agent=agent_name, max_steps=max_steps, seed=seed)

    # 1. Instantiate agent via registry
    try:
        agent = registry.create("agents", agent_name, seed=seed, **agent_params)
    except KeyError as exc:
        raise ValueError(f"Unknown agent: {agent_name!r}") from exc

    # 2. Simulated training loop
    rng = np.random.default_rng(seed)
    metrics: list[dict[str, float]] = []
    for step in range(max_steps):
        # Simulate a training step with synthetic data
        reward = float(rng.uniform(0, 1))
        metrics.append({"step": step, "reward": reward})

    logger.info("train_loop_done", steps=max_steps)

    # 3. Save checkpoint
    if dry_run:
        logger.info("train_dry_run", msg="Skipping checkpoint save")
        model_path = str(checkpoint_dir / "model.pt")
    else:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_path = checkpoint_dir / "model.pt"
        agent.save(str(model_path))

        # Save training log
        log_path = checkpoint_dir / "train_log.json"
        log_path.write_text(
            json.dumps({"agent": agent_name, "seed": seed, "metrics": metrics}, indent=2),
            encoding="utf-8",
        )
        model_path = str(model_path)

    logger.info("train_done", model_path=str(model_path))

    return {
        "agent_name": agent_name,
        "steps_completed": max_steps,
        "model_path": str(model_path),
        "metrics": metrics,
    }


if __name__ == "__main__":
    run_train({"seed": 42, "agent": {"name": "random"}, "training": {"max_steps": 10}})
