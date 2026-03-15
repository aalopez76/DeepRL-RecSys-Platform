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
    try:
        import torch
    except ImportError:
        raise RuntimeError(
            "PyTorch no está instalado. La evaluación OPE real requiere torch. "
            "Instálalo con: poetry add torch"
        )

    ope_cfg = config.get("ope", {})
    seed = config.get("seed", 42)

    estimator_names = ope_cfg.get("estimators", ["ips", "dr"])
    clip_epsilon = ope_cfg.get("clip_epsilon", 0.01)
    fail_on = ope_cfg.get("fail_on", None)

    logger.info("evaluate_start", estimators=estimator_names, clip_epsilon=clip_epsilon)

    # Build data dict — use provided data or generate synthetic
    data = _build_ope_data(ope_cfg, seed, config)

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

    # 3. Generate report if output dir is configured or fallback to paths.artifact_dir
    paths_cfg = config.get("paths", {})
    reports_dir = paths_cfg.get("artifact_dir", "artifacts/checkpoints")
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


def _build_ope_data(ope_cfg: dict[str, Any], seed: int, config: dict[str, Any] = None) -> dict[str, np.ndarray]:
    """Build OPE data from config or generate synthetic data."""
    if "data" in ope_cfg:
        raw = ope_cfg["data"]
        return {
            "rewards": np.asarray(raw["rewards"], dtype=float),
            "propensities": np.asarray(raw["propensities"], dtype=float),
            "action_probs": np.asarray(raw["action_probs"], dtype=float),
        }

    dataset_cfg = config.get("dataset", {}) if config else {}
    if dataset_cfg.get("path"):
        import pandas as pd
        import json
        from deeprl_recsys.serving.runtime import ServingRuntime
        
        data_path = dataset_cfg["path"]
        df = pd.read_parquet(data_path)
        
        n = ope_cfg.get("n_samples", 5000)
        if len(df) > n:
            df = df.sample(n=n, random_state=seed)
            
        rewards = df["reward"].values.astype(float)
        propensities = df["propensity"].values.astype(float)
        actions = df["action"].values
        contexts = df["context"].values
        
        # Load Agent
        paths_cfg = config.get("paths", {})
        artifact_dir = paths_cfg.get("artifact_dir", "artifacts/models/latest")
        
        from deeprl_recsys.core.registry import create
        import copy
        
        agent_name = config.get("agent", {}).get("name", "sac")
        agent_hp = config.get("agent", {}).get("hyperparams", {})
        try:
            agent = create("agents", agent_name, **copy.deepcopy(agent_hp))
            model_pt = Path(artifact_dir) / "model.pt"
            if model_pt.exists():
                agent.load(str(model_pt))
        except Exception as e:
            agent = None
            
        action_probs = []
        candidates = list(range(50)) # Assuming 50 items for this dataset
        import sys, os
        # Silence verbose logging from agent
        _original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            for ctx_str, act in zip(contexts, actions):
                ctx = json.loads(ctx_str)
                if agent is not None:
                    probs = agent.get_action_probabilities(ctx, candidates)
                    action_probs.append(probs.get(int(act), 0.05))
                else:
                    action_probs.append(0.0)
        finally:
            sys.stdout.close()
            sys.stdout = _original_stdout
            
        return {
            "rewards": rewards,
            "propensities": propensities,
            "action_probs": np.asarray(action_probs, dtype=float),
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
