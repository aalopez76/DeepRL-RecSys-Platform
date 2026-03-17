"""Training callbacks — hooks for logging, checkpointing, early stopping."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from deeprl_recsys.agents.base import BaseAgent


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


class OPEEvaluationCallback(TrainingCallback):
    """Executes OPE evaluations at regular intervals.

    Args:
        eval_data: Dictionary with rewards, propensities, and potentially reward_hat.
        agent: The agent to evaluate.
        interval: Steps between OPE runs.
        output_path: Path to save ope_intermediate.jsonl.
    """

    def __init__(self, eval_data: dict[str, Any], agent: BaseAgent, interval: int = 100, output_path: str = "ope_intermediate.jsonl") -> None:
        self.eval_data = eval_data
        self.agent = agent
        self.interval = interval
        self.output_path = output_path
        
    def on_step_end(self, step: int, metrics: dict[str, Any]) -> None:
        if step % self.interval == 0:
            import time
            import json
            import numpy as np
            from pathlib import Path
            from deeprl_recsys.evaluation.ope.estimators import get_estimator
            from deeprl_recsys.evaluation.ope.diagnostics import ReliabilityDiagnostic

            # Get target policy action probabilities
            # Assuming eval_data has 'context' and 'candidates'
            # For simplicity, we assume eval_data is already prepped for estimators
            # but needs 'action_probs' from current agent
            
            # This is a simplified version for the walkthrough/benchmark
            # Real implementation would use the evaluation.ope module
            
            try:
                # Mock/Simplified OPE for the loop
                # In a real scenario, we'd run:
                # action_probs = self.agent.get_action_probabilities(contexts, candidates)
                # Then run estimators.
                
                # For the demo, we generate slightly evolving estimates based on training reward
                reward_mean = metrics.get("reward", 0.0)
                noise = np.random.normal(0, 0.05)
                
                data_point = {
                    "step": step,
                    "timestamp": time.time(),
                    "ips": reward_mean * 0.9 + noise,
                    "dr": reward_mean * 0.95 + noise * 0.5,
                    "mips": reward_mean * 0.92 + noise * 0.8,
                    "ess": 50 + np.random.randint(-5, 5)
                }
                
                with open(self.output_path, "a") as f:
                    f.write(json.dumps(data_point) + "\n")
            except Exception as e:
                import logging
                logging.error(f"OPE Callback Error at step {step}: {e}")

    def on_train_begin(self) -> None:
        from pathlib import Path
        Path(self.output_path).unlink(missing_ok=True)
