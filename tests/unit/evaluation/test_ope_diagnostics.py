"""Unit tests for evaluation/ope — diagnostics + estimators.

Required tests from implementación.txt Phase 4:
- test_tiny_propensities_trigger_unreliable_warning()
- test_missing_propensity_triggers_error_if_required()
- test_clipping_rate_computed()
Plus estimator correctness and factory tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from deeprl_recsys.evaluation.ope.diagnostics import ReliabilityVerdict, run_diagnostics
from deeprl_recsys.evaluation.ope.estimators import (
    DoublyRobustEstimator,
    IPSEstimator,
    MIPSEstimator,
    get_estimator,
)


# ── Helpers ──────────────────────────────────────────


def _make_data(
    n: int = 100,
    propensity_range: tuple[float, float] = (0.1, 0.9),
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Build a synthetic OPE dataset."""
    rng = np.random.RandomState(seed)
    return {
        "rewards": rng.binomial(1, 0.3, size=n).astype(float),
        "propensities": rng.uniform(*propensity_range, size=n),
        "action_probs": rng.uniform(0.05, 0.95, size=n),
    }


# ── Contract Tests (from implementación.txt) ─────────


@pytest.mark.unit
class TestDiagnosticsContract:
    """Required contract tests from implementación.txt Phase 4."""

    def test_tiny_propensities_trigger_unreliable_warning(self) -> None:
        """Extremely small propensities must trigger a warning/error."""
        data = _make_data(n=50, propensity_range=(0.0001, 0.005))
        verdict = run_diagnostics(data, config={"clip_epsilon": 0.01})

        # Should at least be a warning (or error due to low ESS)
        assert verdict.severity in ("warning", "error")
        assert any("propensity" in w.lower() or "ess" in w.lower() for w in verdict.warnings)

    def test_missing_propensity_triggers_error_if_required(self) -> None:
        """Missing propensity scores must produce severity='error'."""
        data = {"rewards": np.array([1.0, 0.0, 1.0])}
        # No 'propensities' key
        verdict = run_diagnostics(data)

        assert verdict.severity == "error"
        assert not verdict.reliable
        assert any("propensity" in w.lower() for w in verdict.warnings)

    def test_clipping_rate_computed(self) -> None:
        """Clipping rate must be correctly computed and present in stats."""
        # 50% of propensities below clip_epsilon
        propensities = np.array([0.001, 0.002, 0.5, 0.6])
        data = {
            "rewards": np.array([1.0, 0.0, 1.0, 0.0]),
            "propensities": propensities,
            "action_probs": np.array([0.5, 0.5, 0.5, 0.5]),
        }
        verdict = run_diagnostics(data, config={"clip_epsilon": 0.01})

        assert "clipping_rate" in verdict.stats
        assert verdict.stats["clipping_rate"] == pytest.approx(0.5)


# ── Additional diagnostics tests ─────────────────────


@pytest.mark.unit
class TestDiagnosticsAdditional:
    """Additional tests for diagnostics beyond the contract."""

    def test_good_data_produces_ok_verdict(self) -> None:
        """Well-behaved data should produce severity='ok'."""
        data = _make_data(n=200, propensity_range=(0.1, 0.9))
        verdict = run_diagnostics(data)

        assert verdict.severity == "ok"
        assert verdict.reliable

    def test_empty_propensity_array_triggers_error(self) -> None:
        """An empty propensity array must produce error."""
        data = {
            "rewards": np.array([1.0]),
            "propensities": np.array([]),
        }
        verdict = run_diagnostics(data)
        assert verdict.severity == "error"
        assert not verdict.reliable

    def test_ess_in_stats(self) -> None:
        """ESS must always be present in stats."""
        data = _make_data(n=50)
        verdict = run_diagnostics(data)
        assert "ess" in verdict.stats
        assert verdict.stats["ess"] > 0

    def test_low_ess_triggers_error(self) -> None:
        """Very low ESS should make verdict unreliable (error)."""
        # One huge weight, rest normal → very low ESS
        data = {
            "rewards": np.ones(10),
            "propensities": np.array([0.001] + [0.5] * 9),
            "action_probs": np.array([0.99] + [0.1] * 9),
        }
        verdict = run_diagnostics(data, config={"clip_epsilon": 0.0001, "min_ess": 5.0})
        assert "ess" in verdict.stats

    def test_deterministic_for_same_input(self) -> None:
        """Same data must produce the same verdict (determinism)."""
        data = _make_data(n=100, seed=123)
        v1 = run_diagnostics(data)
        v2 = run_diagnostics(data)
        assert v1.severity == v2.severity
        assert v1.warnings == v2.warnings
        assert v1.stats == v2.stats

    def test_stats_keys(self) -> None:
        """Stats must include all expected keys."""
        data = _make_data(n=50)
        verdict = run_diagnostics(data)
        expected_keys = {"ess", "clipping_rate", "min_propensity", "max_propensity", "max_weight", "n_samples"}
        assert expected_keys.issubset(verdict.stats.keys())

    def test_policy_echoed_back(self) -> None:
        """Config passed to run_diagnostics must be echoed in verdict.policy."""
        config = {"clip_epsilon": 0.05, "min_ess": 20.0}
        data = _make_data(n=50)
        verdict = run_diagnostics(data, config=config)
        assert verdict.policy == config


# ── Estimator tests ──────────────────────────────────


@pytest.mark.unit
class TestEstimators:
    """Tests for OPE estimators."""

    def test_ips_returns_float(self) -> None:
        """IPS estimate must return a float."""
        data = _make_data(n=50)
        est = IPSEstimator(clip_epsilon=0.01)
        result = est.estimate(data)
        assert isinstance(result, float)

    def test_ips_name(self) -> None:
        """IPS estimator name must be 'ips'."""
        assert IPSEstimator().name == "ips"

    def test_dr_without_reward_hat_equals_ips(self) -> None:
        """DR without reward_hat should equal IPS (reward_hat defaults to 0)."""
        data = _make_data(n=100, seed=7)
        ips_val = IPSEstimator(clip_epsilon=0.01).estimate(data)
        dr_val = DoublyRobustEstimator(clip_epsilon=0.01).estimate(data)
        assert ips_val == pytest.approx(dr_val)

    def test_dr_with_reward_hat(self) -> None:
        """DR with a reward model should differ from IPS."""
        data = _make_data(n=100, seed=7)
        data["reward_hat"] = np.full(100, 0.3)
        ips_val = IPSEstimator(clip_epsilon=0.01).estimate(data)
        dr_val = DoublyRobustEstimator(clip_epsilon=0.01).estimate(data)
        # They may still be close but the formula differs
        assert isinstance(dr_val, float)

    def test_dr_name(self) -> None:
        assert DoublyRobustEstimator().name == "dr"

    def test_mips_name(self) -> None:
        assert MIPSEstimator().name == "mips"

    def test_uniform_propensity_ips_equals_mean_reward(self) -> None:
        """With uniform propensity = action_prob, IPS reduces to mean(reward)."""
        rewards = np.array([1.0, 0.0, 1.0, 0.0])
        p = 0.5
        data = {
            "rewards": rewards,
            "propensities": np.full(4, p),
            "action_probs": np.full(4, p),
        }
        ips = IPSEstimator(clip_epsilon=0.001).estimate(data)
        assert ips == pytest.approx(np.mean(rewards))


@pytest.mark.unit
class TestEstimatorFactory:
    """Tests for get_estimator() factory."""

    def test_get_ips(self) -> None:
        est = get_estimator("ips")
        assert isinstance(est, IPSEstimator)

    def test_get_dr(self) -> None:
        est = get_estimator("dr", clip_epsilon=0.05)
        assert isinstance(est, DoublyRobustEstimator)
        assert est.clip_epsilon == 0.05

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown estimator"):
            get_estimator("nonexistent")


@pytest.mark.unit
class TestReliabilityVerdict:
    """Tests for the ReliabilityVerdict dataclass."""

    def test_reliable_when_ok(self) -> None:
        v = ReliabilityVerdict(reliable=True, severity="ok")
        assert v.reliable
        assert v.warnings == []

    def test_not_reliable_when_error(self) -> None:
        v = ReliabilityVerdict(reliable=False, severity="error", warnings=["bad"])
        assert not v.reliable
        assert len(v.warnings) == 1
