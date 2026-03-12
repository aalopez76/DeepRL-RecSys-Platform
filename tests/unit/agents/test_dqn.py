"""Unit tests for DQNAgent.

Verifies initialisation, act, update, save/load, and get_action_probabilities
using a tiny item catalogue (20 items, 8-dim embeddings) for speed.
"""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Check for torch WITHOUT importing it (so collection doesn't pollute sys.modules)
_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_skip_no_torch = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="PyTorch not installed — skipping DQN tests"
)

from deeprl_recsys.agents.dqn import DQNAgent  # noqa: E402

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
NUM_ITEMS = 20
EMB_DIM = 8
CANDIDATES = list(range(NUM_ITEMS))
OBS: dict[str, Any] = {"user_id": 0}
DUMMY_BATCH: dict[str, Any] = {"step": 0, "data": []}


@pytest.fixture()
def agent() -> DQNAgent:
    """Small DQN agent with deterministic seed."""
    return DQNAgent(num_items=NUM_ITEMS, embedding_dim=EMB_DIM, seed=0)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@_skip_no_torch
@pytest.mark.unit
class TestDQNAgentInit:
    """Initialisation checks."""

    def test_name(self, agent: DQNAgent) -> None:
        """Agent name property returns 'dqn'."""
        assert agent.name == "dqn"

    def test_q_net_is_not_none(self, agent: DQNAgent) -> None:
        """Q-network is built when PyTorch is available."""
        assert agent.q_net is not None

    def test_custom_hyperparams_stored(self) -> None:
        """Hyperparams are forwarded to _config."""
        ag = DQNAgent(num_items=10, embedding_dim=4, lr=5e-4, seed=1, foo="bar")
        assert ag._config.get("foo") == "bar"


@_skip_no_torch
@pytest.mark.unit
class TestDQNAgentAct:
    """act() method checks."""

    def test_act_returns_list(self, agent: DQNAgent) -> None:
        """act() returns a list."""
        result = agent.act(OBS, CANDIDATES)
        assert isinstance(result, list)

    def test_act_same_length_as_candidates(self, agent: DQNAgent) -> None:
        """act() returns the same number of items as candidates."""
        result = agent.act(OBS, CANDIDATES)
        assert len(result) == len(CANDIDATES)

    def test_act_returns_valid_item_ids(self, agent: DQNAgent) -> None:
        """act() only returns item IDs that were in the candidate list."""
        result = agent.act(OBS, CANDIDATES)
        assert set(result) == set(CANDIDATES)

    def test_act_empty_candidates(self, agent: DQNAgent) -> None:
        """act() on an empty candidate list returns an empty list."""
        assert agent.act(OBS, []) == []


@_skip_no_torch
@pytest.mark.unit
class TestDQNAgentUpdate:
    """update() method checks."""

    def test_update_returns_loss_dict(self, agent: DQNAgent) -> None:
        """update() returns a dict containing 'loss' as a float."""
        metrics = agent.update(DUMMY_BATCH)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)

    def test_update_multiple_steps_no_error(self, agent: DQNAgent) -> None:
        """10 consecutive update steps run without raising exceptions."""
        for i in range(10):
            m = agent.update({"step": i})
            assert isinstance(m["loss"], float)


@_skip_no_torch
@pytest.mark.unit
class TestDQNGetActionProbabilities:
    """get_action_probabilities() checks."""

    def test_probs_cover_all_candidates(self, agent: DQNAgent) -> None:
        """Returns one probability per candidate."""
        probs = agent.get_action_probabilities(OBS, CANDIDATES)
        assert set(probs.keys()) == set(CANDIDATES)

    def test_probs_sum_to_one(self, agent: DQNAgent) -> None:
        """Probabilities sum to 1.0 within floating-point tolerance."""
        probs = agent.get_action_probabilities(OBS, CANDIDATES)
        assert abs(sum(probs.values()) - 1.0) < 1e-5

    def test_probs_are_non_negative(self, agent: DQNAgent) -> None:
        """Every probability is in [0, 1]."""
        probs = agent.get_action_probabilities(OBS, CANDIDATES)
        assert all(0.0 <= p <= 1.0 for p in probs.values())

    def test_probs_empty_candidates_fallback(self, agent: DQNAgent) -> None:
        """Empty candidates returns an empty dict."""
        probs = agent.get_action_probabilities(OBS, [])
        assert probs == {}


@_skip_no_torch
@pytest.mark.unit
class TestDQNSaveLoad:
    """save() / load() round-trip checks."""

    def test_save_creates_file(self, agent: DQNAgent) -> None:
        """save() writes a file to the given path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "dqn.pt")
            agent.save(path)
            assert Path(path).exists()

    def test_load_does_not_raise(self, agent: DQNAgent) -> None:
        """load() on a valid checkpoint raises no exceptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "dqn.pt")
            agent.save(path)
            agent2 = DQNAgent(num_items=NUM_ITEMS, embedding_dim=EMB_DIM, seed=99)
            agent2.load(path)  # must not raise

    def test_load_restores_weights(self, agent: DQNAgent) -> None:
        """Loaded agent produces the same action ranking as the saved agent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "dqn.pt")
            expected = agent.act(OBS, CANDIDATES)
            agent.save(path)
            agent2 = DQNAgent(num_items=NUM_ITEMS, embedding_dim=EMB_DIM, seed=99)
            agent2.load(path)
            assert agent2.act(OBS, CANDIDATES) == expected

    def test_load_nonexistent_path_no_crash(self, agent: DQNAgent) -> None:
        """load() with a missing file silently does nothing."""
        agent.load("/nonexistent/path/dqn.pt")  # must not raise
