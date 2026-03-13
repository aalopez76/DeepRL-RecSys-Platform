"""Pipeline: evaluate — run OPE estimators + diagnostics.

Orchestrates:
1. Load prepared dataset
2. Run OPE estimators (IPS, DR)
3. Run diagnostics → ReliabilityVerdict
4. Optionally exit(1) if severity == "error" and fail_on == "error"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from deeprl_recsys.core.logging import get_logger
from deeprl_recsys.evaluation.ope.diagnostics import run_diagnostics
from deeprl_recsys.evaluation.ope.estimators import get_estimator

logger = get_logger(__name__)


def run_evaluate(config: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    """Execute the evaluation pipeline.

    Args:
        config: Resolved platform config.  Expected keys:

            - ``ope.estimators`` — list of estimator names
            - ``ope.clip_epsilon`` — clipping threshold
            - ``ope.fail_on`` — if ``"error"``, exit(1) on verdict error
            - ``ope.data`` — dict with ``rewards``, ``propensities``,
              ``action_probs`` arrays (or generate synthetic data)
            - ``seed`` — for reproducible synthetic data

    Returns:
        Dictionary with ``estimates``, ``verdict``, and ``severity``.
    """
    ope_cfg = config.get("ope", {})
    seed = config.get("seed", 42)

    estimator_names = ope_cfg.get("estimators", ["ips", "dr"])
    clip_epsilon = ope_cfg.get("clip_epsilon", 0.01)
    fail_on = ope_cfg.get("fail_on", None)

    logger.info("evaluate_start", estimators=estimator_names, clip_epsilon=clip_epsilon)

    # Build data dict — use provided data or generate synthetic
    data = _build_ope_data(ope_cfg, seed)

    # 1. Run estimators
    estimates: dict[str, float] = {}
    for name in estimator_names:
        est = get_estimator(name, clip_epsilon=clip_epsilon)
        value = est.estimate(data)
        estimates[name] = value
        logger.info("estimator_result", estimator=name, value=value)

    # 2. Run diagnostics
    diag_config = {"clip_epsilon": clip_epsilon}
    verdict = run_diagnostics(data, config=diag_config)

    logger.info(
        "evaluate_verdict",
        reliable=verdict.reliable,
        severity=verdict.severity,
        warnings=verdict.warnings,
    )

    result = {
        "estimates": estimates,
        "verdict": {
            "reliable": verdict.reliable,
            "severity": verdict.severity,
            "warnings": verdict.warnings,
            "stats": verdict.stats,
        },
        "severity": verdict.severity,
    }

    # 3. Generate report if output dir is configured or fallback to artifacts/checkpoints
    reports_dir = ope_cfg.get("reports_dir", "artifacts/checkpoints")
    if reports_dir and not dry_run:
        from deeprl_recsys.evaluation.report import generate_report
        import json
        from pathlib import Path
        
        out = Path(reports_dir)
        out.mkdir(parents=True, exist_ok=True)

        generate_report(estimates, verdict, output_dir=reports_dir, format="markdown")
        
        # Guardar OPE JSON robusto con importance weights para el dashboard
        action_probs = data.get("action_probs", np.ones_like(data.get("propensities", [])))
        clipped_props = np.clip(data.get("propensities", [1.0]), clip_epsilon, None)
        importance_weights = (action_probs / clipped_props).tolist()
        
        report_data = {
            "estimates": estimates,
            "verdict": {
                "reliable": verdict.reliable,
                "severity": verdict.severity,
                "warnings": verdict.warnings,
                "stats": verdict.stats,
            },
            "importance_weights": importance_weights
        }
        
        with open(out / "ope_report.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

    # 4. Fail if required
    if fail_on == "error" and verdict.severity == "error":
        logger.error("evaluate_fail", msg="Verdict is 'error' and fail_on='error'")
        raise SystemExit(1)

    return result


def _build_ope_data(ope_cfg: dict[str, Any], seed: int) -> dict[str, np.ndarray]:
    """Build OPE data from config or generate synthetic data."""
    if "data" in ope_cfg:
        raw = ope_cfg["data"]
        return {
            "rewards": np.asarray(raw["rewards"], dtype=float),
            "propensities": np.asarray(raw["propensities"], dtype=float),
            "action_probs": np.asarray(raw["action_probs"], dtype=float),
        }

    # Generate synthetic (for testing / demo)
    rng = np.random.default_rng(seed)
    n = ope_cfg.get("n_samples", 100)
    return {
        "rewards": rng.binomial(1, 0.3, size=n).astype(float),
        "propensities": rng.uniform(0.1, 0.9, size=n),
        "action_probs": rng.uniform(0.05, 0.8, size=n),
    }


if __name__ == "__main__":
    run_evaluate({"seed": 42, "ope": {"estimators": ["ips", "dr"]}})
