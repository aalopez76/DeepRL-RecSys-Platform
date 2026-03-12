"""OPE diagnostics — ESS, overlap, clipping, and :class:`ReliabilityVerdict`.

:func:`run_diagnostics` inspects the propensity weights and produces a
deterministic :class:`ReliabilityVerdict` with severity, warnings, and
summary statistics.

Severity rules:

- ``"error"``  — missing propensities **or** ESS < ``min_ess``
- ``"warning"`` — min propensity below ``clip_epsilon`` or clipping rate > 10 %
- ``"ok"``     — all checks pass
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from deeprl_recsys.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReliabilityVerdict:
    """Standardised verdict for OPE reliability.

    Attributes:
        reliable: ``True`` when severity is *not* ``"error"``.
        severity: Overall severity level.
        warnings: Human-readable warning messages.
        stats: Diagnostic statistics (ESS, clipping_rate, etc.).
        policy: Configuration used for the diagnostic run.
    """

    reliable: bool
    severity: Literal["ok", "warning", "error"]
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)


def run_diagnostics(
    data: dict[str, np.ndarray],
    config: dict[str, Any] | None = None,
) -> ReliabilityVerdict:
    """Run OPE diagnostic checks and produce a reliability verdict.

    The verdict is **deterministic** for the same *data* and *config*.

    Args:
        data: Dictionary with at least ``rewards`` and ``propensities``.
            Optionally includes ``action_probs`` (defaults to uniform 1.0).
        config: OPE configuration. Recognised keys:

            - ``clip_epsilon`` (float, default 0.01)
            - ``min_ess`` (float, default 10.0)
            - ``max_clipping_rate`` (float, default 0.10)

    Returns:
        A :class:`ReliabilityVerdict` summarising reliability.
    """
    config = config or {}
    clip_epsilon: float = config.get("clip_epsilon", 0.01)
    min_ess: float = config.get("min_ess", 10.0)
    max_clipping_rate: float = config.get("max_clipping_rate", 0.10)

    warnings: list[str] = []
    stats: dict[str, float] = {}

    propensities = data.get("propensities")

    # ── Check: propensity exists ─────────────────────────────
    if propensities is None or len(propensities) == 0:
        return ReliabilityVerdict(
            reliable=False,
            severity="error",
            warnings=["No propensity scores available — OPE is not reliable"],
            stats={},
            policy=config,
        )

    # ── Compute importance weights ───────────────────────────
    action_probs = data.get("action_probs", np.ones_like(propensities))
    clipped_props = np.clip(propensities, clip_epsilon, None)
    weights = action_probs / clipped_props

    # ── Statistics ────────────────────────────────────────────
    n_samples = len(propensities)
    stats["n_samples"] = float(n_samples)

    # Effective Sample Size
    w_sum = float(np.sum(weights))
    w_sq_sum = float(np.sum(weights**2))
    ess = (w_sum**2 / w_sq_sum) if w_sq_sum > 0 else 0.0
    stats["ess"] = ess

    # Clipping rate
    clipping_rate = float(np.mean(propensities < clip_epsilon))
    stats["clipping_rate"] = clipping_rate

    # Propensity stats
    min_prop = float(np.min(propensities))
    max_prop = float(np.max(propensities))
    stats["min_propensity"] = min_prop
    stats["max_propensity"] = max_prop

    # Weight stats
    max_weight = float(np.max(weights))
    stats["max_weight"] = max_weight

    logger.info(
        "diagnostics_stats",
        ess=round(ess, 4),
        clipping_rate=round(clipping_rate, 4),
        min_propensity=round(min_prop, 6),
        max_propensity=round(max_prop, 6),
        max_weight=round(max_weight, 4),
        n_samples=n_samples,
    )

    # ── Evaluate severity (order matters: error > warning > ok) ──
    severity: Literal["ok", "warning", "error"] = "ok"

    if min_prop < clip_epsilon:
        warnings.append(
            f"Min propensity ({min_prop:.6f}) below clip_epsilon ({clip_epsilon})"
        )
        if severity == "ok":
            severity = "warning"

    if clipping_rate > max_clipping_rate:
        warnings.append(
            f"High clipping rate: {clipping_rate:.2%} of propensities clipped "
            f"(threshold: {max_clipping_rate:.2%})"
        )
        if severity == "ok":
            severity = "warning"

    if ess < min_ess:
        warnings.append(
            f"Very low ESS ({ess:.1f}) — estimates may be unreliable "
            f"(threshold: {min_ess:.1f})"
        )
        severity = "error"

    reliable = severity != "error"

    return ReliabilityVerdict(
        reliable=reliable,
        severity=severity,
        warnings=warnings,
        stats=stats,
        policy=config,
    )
