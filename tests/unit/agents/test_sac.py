"""Unit tests for SACAgent.

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
    not _TORCH_AVAILABLE, reason="PyTorch not installed — skipping SAC tests"
)

from deeprl_recsys.agents.sac import SACAgent  # noqa: E402

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
NUM_ITEMS = 20
EMB_DIM = 8
CANDIDATES = list(range(NUM_ITEMS))
OBS: dict[str, Any] = {"user_id": 0}
DUMMY_BATCH: dict[str, Any] = {"step": 0, "data": []}


@pytest.fixture()
def agent() -> SACAgent:
    """Small SAC agent with deterministic seed."""
    return SACAgent(num_items=NUM_ITEMS, embedding_dim=EMB_DIM, seed=0)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@_skip_no_torch
@pytest.mark.unit
class TestSACAgentInit:
    """Initialisation checks."""

    def test_name(self, agent: SACAgent) -> None:
        """Agent name property returns 'sac'."""
        assert agent.name == "sac"

    def test_network_is_not_none(self, agent: SACAgent) -> None:
        """SAC Actor-Critic network is built when PyTorch is available."""
        assert agent.network is not None

    def test_custom_hyperparams_stored(self) -> None:
        """Extra keyword args are forwarded to _config."""
        ag = SACAgent(num_items=10, embedding_dim=4, lr=1e-3, seed=1, tau=0.005)
        assert ag._config.get("tau") == 0.005


@_skip_no_torch
@pytest.mark.unit
class TestSACAgentAct:
    """act() method checks."""

    def test_act_returns_list(self, agent: SACAgent) -> None:
        assert isinstance(agent.act(OBS, CANDIDATES), list)

    def test_act_same_length_as_candidates(self, agent: SACAgent) -> None:
        assert len(agent.act(OBS, CANDIDATES)) == len(CANDIDATES)

    def test_act_returns_valid_item_ids(self, agent: SACAgent) -> None:
        assert set(agent.act(OBS, CANDIDATES)) == set(CANDIDATES)

    def test_act_empty_candidates(self, agent: SACAgent) -> None:
        assert agent.act(OBS, []) == []


@_skip_no_torch
@pytest.mark.unit
class TestSACAgentUpdate:
    """update() method checks."""

    def test_update_returns_loss_dict(self, agent: SACAgent) -> None:
        metrics = agent.update(DUMMY_BATCH)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)

    def test_update_multiple_steps_no_error(self, agent: SACAgent) -> None:
        for i in range(10):
            m = agent.update({"step": i})
            assert isinstance(m["loss"], float)


@_skip_no_torch
@pytest.mark.unit
class TestSACGetActionProbabilities:
    """get_action_probabilities() checks."""

    def test_probs_cover_all_candidates(self, agent: SACAgent) -> None:
        probs = agent.get_action_probabilities(OBS, CANDIDATES)
        assert set(probs.keys()) == set(CANDIDATES)

    def test_probs_sum_to_one(self, agent: SACAgent) -> None:
        probs = agent.get_action_probabilities(OBS, CANDIDATES)
        assert abs(sum(probs.values()) - 1.0) < 1e-5

    def test_probs_are_non_negative(self, agent: SACAgent) -> None:
        probs = agent.get_action_probabilities(OBS, CANDIDATES)
        assert all(0.0 <= p <= 1.0 for p in probs.values())

    def test_probs_empty_candidates_fallback(self, agent: SACAgent) -> None:
        assert agent.get_action_probabilities(OBS, []) == {}


@_skip_no_torch
@pytest.mark.unit
class TestSACSaveLoad:
    """save() / load() round-trip checks."""

    def test_save_creates_file(self, agent: SACAgent) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "sac.pt")
            agent.save(path)
            assert Path(path).exists()

    def test_load_does_not_raise(self, agent: SACAgent) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "sac.pt")
            agent.save(path)
            agent2 = SACAgent(num_items=NUM_ITEMS, embedding_dim=EMB_DIM, seed=99)
            agent2.load(path)

    def test_load_restores_weights(self, agent: SACAgent) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "sac.pt")
            expected = agent.act(OBS, CANDIDATES)
            agent.save(path)
            agent2 = SACAgent(num_items=NUM_ITEMS, embedding_dim=EMB_DIM, seed=99)
            agent2.load(path)
            assert agent2.act(OBS, CANDIDATES) == expected

    def test_load_nonexistent_path_no_crash(self, agent: SACAgent) -> None:
        agent.load("/nonexistent/path/sac.pt")
