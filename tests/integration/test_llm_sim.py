"""Integration tests for LLMSimulator."""

from __future__ import annotations

import importlib.util

import pytest

# Detect transformers WITHOUT importing it at collection time.
# Importing transformers at module level would transitively import torch,
# breaking the extras_isolation tests which check sys.modules cleanliness.
_TRANSFORMERS_AVAILABLE = (
    importlib.util.find_spec("transformers") is not None
    and importlib.util.find_spec("torch") is not None
)


@pytest.mark.integration
@pytest.mark.skipif(
    not _TRANSFORMERS_AVAILABLE,
    reason="transformers+torch not available or DLL load failed",
)
def test_llm_sim_initializes_and_predicts() -> None:
    """Verify LLMSimulator can instantiate GPT-2 and generate a reward.

    Uses device=-1 (CPU) for compatibility across test environments.
    """
    # Import inside the test so it only runs when torch/transformers are present
    from deeprl_recsys.environment.simulators.llm_sim import LLMSimulator  # noqa: PLC0415

    # Use max_new_tokens=1 to make it fast
    sim = LLMSimulator(model_name="gpt2", device=-1, max_new_tokens=1)

    user = {"user_id": 123}
    item = 456

    reward = sim.simulate_response(user, item)

    assert isinstance(reward, float)
    assert 0.0 <= reward <= 1.0
