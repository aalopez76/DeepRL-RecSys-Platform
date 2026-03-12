"""Integration tests for advanced PyTorch agents (DQN, PPO, SAC).

Verifies that:
1. Training loop doesn't crash (10 dummy updates).
2. They generate action probabilities suitable for OPE.
3. OPE diagnostics produce valid ESS metrics.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

# Use find_spec to detect torch WITHOUT importing it at collection time.
# This preserves sys.modules cleanliness for extras_isolation tests.
_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

from deeprl_recsys.agents.dqn import DQNAgent
from deeprl_recsys.agents.ppo import PPOAgent
from deeprl_recsys.agents.sac import SACAgent
from deeprl_recsys.evaluation.ope.diagnostics import run_diagnostics



@pytest.mark.integration
@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not available or fails to load DLLs")
@pytest.mark.parametrize("agent_cls", [DQNAgent, PPOAgent, SACAgent])
def test_advanced_agent_training_and_ope(agent_cls: type) -> None:
    """Verify agent can train and produce OPE-compatible action probabilities."""
    agent = agent_cls(num_items=10, embedding_dim=16, seed=42)

    # 1. Dummy training loop (10 steps)
    for step in range(10):
        metrics = agent.update({"step": step, "data": []})
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)

    # 2. Generate action probabilities for a mock dataset
    # Suppose we have 3 logged interactions where the behavior policy
    # chose items 1, 3, and 5 with some propensity.
    logged_actions = [1, 3, 5]
    propensities = [0.2, 0.5, 0.1]
    candidates = [0, 1, 2, 3, 4, 5, 6, 7]

    action_probs = []
    for action in logged_actions:
        probs_dict = agent.get_action_probabilities(
            observation={"user_id": 0}, candidates=candidates
        )
        # Agent must assign some probability to all candidates
        assert len(probs_dict) == len(candidates)
        assert pytest.approx(sum(probs_dict.values()), rel=1e-5) == 1.0

        # Extract the probability the agent would have chosen the logged action
        action_probs.append(probs_dict[action])

    # 3. Verify OPE diagnostics integration
    data = {
        "propensities": np.array(propensities),
        "action_probs": np.array(action_probs),
        "rewards": np.array([1.0, 0.0, 1.0]),
    }

    verdict = run_diagnostics(data, config={"min_ess": 1.0})
    
    # Verify ESS was calculated properly
    assert "ess" in verdict.stats
    assert np.isfinite(verdict.stats["ess"])
    assert verdict.stats["ess"] >= 0.0
    
    # Detailed ESS math check
    weights = data["action_probs"] / np.clip(data["propensities"], 0.01, None)
    expected_ess = (weights.sum() ** 2) / (weights ** 2).sum()
    assert pytest.approx(verdict.stats["ess"]) == expected_ess
