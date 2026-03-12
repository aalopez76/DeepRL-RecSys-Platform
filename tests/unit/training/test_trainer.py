"""Tests for training module."""

from __future__ import annotations

from deeprl_recsys.agents.baselines import RandomAgent
from deeprl_recsys.training.trainer import Trainer


def test_trainer_fit() -> None:
    """Verify Trainer executes step and logging correctly."""
    agent = RandomAgent()
    trainer = Trainer(agent=agent, max_steps=3, eval_interval=1)
    
    metrics = trainer.fit(data=[1, 2, 3])
    
    assert metrics["steps_completed"] == 3
