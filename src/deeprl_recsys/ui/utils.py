"""Utility functions for Streamlit Dashboard.

Includes caching logic for reading artifacts, logs, and generated OPE reports.

Fixes applied (2026-03-24):
- scan_artifacts: now discovers artifacts by ope_report.json when metadata.json is absent
- load_train_log: tries both train_log.json and train_log.jsonl
- AGENT_STUB_NAMES: constant set for stub agents shown in the playground
"""

from typing import Any, Dict, List
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# Agents known to be functional stubs (identical OPE values expected)
AGENT_STUB_NAMES = {"dqn", "ppo"}


def compute_ess(weights: List[float]) -> float:
    """Computes the Effective Sample Size (ESS) using the formula: (sum(w))^2 / sum(w^2)"""
    if not weights:
        return 0.0
    w_arr = np.array(weights)
    sum_w = np.sum(w_arr)
    sum_w2 = np.sum(w_arr**2)
    return float(sum_w**2 / sum_w2) if sum_w2 > 0 else 0.0


def _infer_agent_from_folder(folder_name: str) -> str:
    """Infer agent type from benchmark folder name (e.g. 'benchmark_dqn_random' → 'dqn')."""
    name_lower = folder_name.lower()
    for agent in ("sac", "dqn", "ppo", "greedy", "topk", "random"):
        if agent in name_lower:
            return agent
    return "unknown"


@st.cache_data(ttl=60)
def scan_artifacts(base_dir: str | Path) -> pd.DataFrame:
    """Finds all model artifacts in base_dir and extracts their metadata.

    Strategy (priority order for each subfolder):
    1. Read metadata.json if it exists.
    2. Fall back to checking for ope_report.json (present in all benchmarks).

    Args:
        base_dir: Directory containing model subdirectories.

    Returns:
        DataFrame with columns: artifact_id, path, agent_name, schema_version, created_at.
    """
    base_path = Path(base_dir)
    if not base_path.exists() or not base_path.is_dir():
        return pd.DataFrame()

    records = []

    # Iterate direct children and also check two levels deep.
    # This handles both flat (benchmark_bts/) and nested (benchmark_sac_synthetic/) layouts.
    candidate_dirs: list[Path] = []
    for child in base_path.iterdir():
        if child.is_dir() and not child.name.startswith("."):
            candidate_dirs.append(child)

    for artifact_dir in sorted(candidate_dirs):
        # Skip utility dirs that aren't real artifacts
        if artifact_dir.name in {"checkpoints", "logs", "plots"}:
            continue

        ope_file = artifact_dir / "ope_report.json"
        meta_file = artifact_dir / "metadata.json"

        # We require at least one of these to consider it a real artifact
        if not ope_file.exists() and not meta_file.exists():
            continue

        # Read metadata (preferred), otherwise reconstruct from folder name
        agent_name = _infer_agent_from_folder(artifact_dir.name)
        schema_version = "bandit_v1"
        created_at = None
        description = ""

        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                agent_name = data.get("agent_name", agent_name)
                schema_version = data.get("schema_version", schema_version)
                created_at = data.get("created_at")
                description = data.get("description", "")
            except Exception:
                pass

        records.append(
            {
                "artifact_id": artifact_dir.name,
                "path": str(artifact_dir),
                "agent_name": agent_name,
                "schema_version": schema_version,
                "created_at": created_at,
                "description": description,
            }
        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True).dt.tz_localize(None)
        df = df.sort_values(by=["agent_name", "artifact_id"], ascending=True)
    return df


@st.cache_data(ttl=60)
def load_ope_report(artifact_path: str) -> Dict[str, Any]:
    """Loads ope_report.json from the artifact path."""
    path = Path(artifact_path) / "ope_report.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_train_log_inner(log_path: Path) -> pd.DataFrame:
    """Helper to parse a json or jsonl file into a DataFrame with columns [step, reward(, loss)]."""
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Format: {"agent": ..., "seed": ..., "metrics": [{step, reward, ...}, ...]}
        if isinstance(data, dict) and "metrics" in data:
            return pd.DataFrame(data["metrics"])
        # Format: plain list of records
        if isinstance(data, list):
            return pd.DataFrame(data)
    except json.JSONDecodeError:
        pass

    # JSONL fallback: one JSON object per line
    records = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass

    return pd.DataFrame(records)


@st.cache_data(ttl=60)
def load_train_log(artifact_path: str) -> pd.DataFrame:
    """Loads training log from the artifact path.

    Tries (in order):
    1. train_log.json   – used by benchmark_* artifacts
    2. train_log.jsonl  – used by some legacy runs
    """
    base = Path(artifact_path)
    for filename in ("train_log.json", "train_log.jsonl"):
        path = base / filename
        if path.exists():
            df = _load_train_log_inner(path)
            if not df.empty:
                return df
    return pd.DataFrame()


@st.cache_resource
def load_serving_runtime(artifact_path: str | Path):
    """Loads and caches a ServingRuntime instance for a specific artifact.

    Using st.cache_resource ensures we only load each model once and
    reuse the instance across sessions, preventing memory overhead.
    """
    from deeprl_recsys.serving.runtime import ServingRuntime

    runtime = ServingRuntime()
    runtime.load(artifact_path)
    return runtime


def check_reports_extra() -> bool:
    """Check if weasyprint and reportlab are installed."""
    try:
        import weasyprint  # noqa: F401
        import reportlab  # noqa: F401

        return True
    except ImportError:
        return False
