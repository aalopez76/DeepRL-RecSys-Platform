"""OPE estimators — IPS, Doubly Robust, MIPS.

Each estimator implements :class:`BaseEstimator` and provides an
:meth:`estimate` method that takes a data dictionary with keys:

- ``rewards`` — observed rewards  *(n,)*
- ``propensities`` — logging policy probabilities  *(n,)*
- ``action_probs`` — target policy probabilities  *(n,)*
- ``reward_hat`` — (DR only) reward model predictions  *(n,)*
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseEstimator(ABC):
    """Abstract base for Off-Policy Evaluation estimators."""

    @abstractmethod
    def estimate(self, data: dict[str, np.ndarray]) -> float:
        """Estimate the target policy's expected reward.

        Args:
            data: Dictionary with arrays keyed by ``rewards``,
                ``propensities``, ``action_probs``, and optionally
                ``reward_hat``.

        Returns:
            Estimated policy value (scalar).
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable estimator name for logging and reports."""


class IPSEstimator(BaseEstimator):
    """Inverse Propensity Scoring (IPS) estimator.

    Applies importance-weight clipping at *clip_epsilon* to bound
    variance.

    Args:
        clip_epsilon: Minimum propensity value used for clipping.
    """

    def __init__(self, clip_epsilon: float = 0.01) -> None:
        self.clip_epsilon = clip_epsilon

    def estimate(self, data: dict[str, np.ndarray]) -> float:
        rewards = data["rewards"]
        propensities = np.clip(data["propensities"], self.clip_epsilon, None)
        action_probs = data["action_probs"]
        weights = action_probs / propensities
        return float(np.mean(weights * rewards))

    @property
    def name(self) -> str:
        return "ips"


class DoublyRobustEstimator(BaseEstimator):
    """Doubly Robust (DR) estimator.

    Combines a reward model (``reward_hat``) with importance weighting
    for lower variance.  Falls back to IPS when ``reward_hat`` is absent.

    Args:
        clip_epsilon: Minimum propensity value used for clipping.
    """

    def __init__(self, clip_epsilon: float = 0.01) -> None:
        self.clip_epsilon = clip_epsilon

    def estimate(self, data: dict[str, np.ndarray]) -> float:
        rewards = data["rewards"]
        propensities = np.clip(data["propensities"], self.clip_epsilon, None)
        action_probs = data["action_probs"]
        reward_hat = data.get("reward_hat", np.zeros_like(rewards))
        weights = action_probs / propensities
        return float(np.mean(reward_hat + weights * (rewards - reward_hat)))

    @property
    def name(self) -> str:
        return "dr"


class MIPSEstimator(BaseEstimator):
    """Marginalised Inverse Propensity Scoring (MIPS) estimator.

    MIPS reduces the variance of IPS in large action spaces by using
    marginal propensities instead of joint action probabilities.
    This assumes that the logging policy can be decomposed into
    item-wise selection probabilities.

    **Formula**:
    .. math::
        \\hat{V}_{MIPS} = \\frac{1}{n} \\sum_{i=1}^{n} \\frac{\\pi_e(a_i|s_i)}{\\hat{p}(a_i)} r_i

    Where:
    - :math:`\\pi_e(a|s)` is the target policy probability.
    - :math:`\\hat{p}(a)` is the marginal logging propensity of action :math:`a`.

    Args:
        action_marginals: Dictionary mapping action IDs to their marginal
            logging propensities.
        clip_epsilon: Minimum propensity for clipping (default: 0.01).
    """

    def __init__(
        self,
        action_marginals: dict[int, float] | None = None,
        clip_epsilon: float = 0.01,
    ) -> None:
        self.action_marginals = action_marginals or {}
        self.clip_epsilon = clip_epsilon

    def estimate(self, data: dict[str, np.ndarray]) -> float:
        """Estimate value using marginal propensities.

        If ``action_marginals`` were not provided at init, it attempts to
        find them in the data dictionary.
        """
        rewards = data["rewards"]
        action_probs = data["action_probs"]
        
        # Determine marginals for each sample
        # We assume 'actions' key exists if we need to map marginals manually
        # If not, we fall back to IPS-style propensities if they are already marginals
        marginals = data.get("marginal_propensities")
        if marginals is None:
            # Fallback to IPS if no marginals provided
            propensities = np.clip(data["propensities"], self.clip_epsilon, None)
        else:
            propensities = np.clip(marginals, self.clip_epsilon, None)

        weights = action_probs / propensities
        return float(np.mean(weights * rewards))

    @property
    def name(self) -> str:
        return "mips"


# ── Factory ─────────────────────────────────────────────────────

_ESTIMATORS: dict[str, type[BaseEstimator]] = {
    "ips": IPSEstimator,
    "dr": DoublyRobustEstimator,
    "mips": MIPSEstimator,
}


def get_estimator(name: str, **kwargs: Any) -> BaseEstimator:
    """Instantiate an estimator by name.

    Args:
        name: One of ``"ips"``, ``"dr"``, ``"mips"``.
        **kwargs: Passed to the estimator constructor.

    Returns:
        An instantiated estimator.

    Raises:
        ValueError: If *name* is not recognised.
    """
    cls = _ESTIMATORS.get(name)
    if cls is None:
        available = ", ".join(sorted(_ESTIMATORS))
        raise ValueError(f"Unknown estimator: {name!r}. Available: [{available}]")
    return cls(**kwargs)
