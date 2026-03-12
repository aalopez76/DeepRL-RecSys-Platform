"""Builtin registrations — explicit name → ``"module:Symbol"`` maps.

This module MUST NOT import any heavy dependencies. Values are strings
of the form ``"package.module:ClassName"`` that the :mod:`registry`
resolves lazily via ``importlib`` only when requested.
"""

from __future__ import annotations

# ── Agents ────────────────────────────────────────────
AGENTS: dict[str, str] = {
    "random": "deeprl_recsys.agents.baselines:RandomAgent",
    "greedy": "deeprl_recsys.agents.baselines:GreedyAgent",
    "topk": "deeprl_recsys.agents.baselines:TopKAgent",
    "dqn": "deeprl_recsys.agents.dqn:DQNAgent",
    "ppo": "deeprl_recsys.agents.ppo:PPOAgent",
    "sac": "deeprl_recsys.agents.sac:SACAgent",
}

# ── Environments ──────────────────────────────────────
ENVIRONMENTS: dict[str, str] = {
    "rec_env": "deeprl_recsys.environment.gym_wrappers:RecEnv",
}

# ── Simulators ────────────────────────────────────────
SIMULATORS: dict[str, str] = {
    "static": "deeprl_recsys.environment.simulators.static_sim:StaticSimulator",
    "llm": "deeprl_recsys.environment.simulators.llm_sim:LLMSimulator",
}

# ── OPE Estimators ────────────────────────────────────
ESTIMATORS: dict[str, str] = {
    "ips": "deeprl_recsys.evaluation.ope.estimators:IPSEstimator",
    "dr": "deeprl_recsys.evaluation.ope.estimators:DoublyRobustEstimator",
    "mips": "deeprl_recsys.evaluation.ope.estimators:MIPSEstimator",
}

# ── Metrics ───────────────────────────────────────────
METRICS: dict[str, str] = {
    "ctr": "deeprl_recsys.evaluation.metrics:ctr",
    "ndcg": "deeprl_recsys.evaluation.metrics:ndcg",
    "hit_rate": "deeprl_recsys.evaluation.metrics:hit_rate",
    "mrr": "deeprl_recsys.evaluation.metrics:mrr",
}
