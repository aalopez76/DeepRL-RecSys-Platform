"""Utility functions for Streamlit Dashboard.

Includes caching logic for reading artifacts, logs, and generated OPE reports.
"""

from typing import Any, Dict, List
import pandas as pd
import json
from pathlib import Path
import streamlit as st
import numpy as np

def compute_ess(weights: List[float]) -> float:
    """Computes the Effective Sample Size (ESS) using the formula: (sum(w))^2 / sum(w^2)"""
    if not weights:
        return 0.0
    w_arr = np.array(weights)
    sum_w = np.sum(w_arr)
    sum_w2 = np.sum(w_arr ** 2)
    return float(sum_w ** 2 / sum_w2) if sum_w2 > 0 else 0.0

@st.cache_data(ttl=60)
def scan_artifacts(base_dir: str | Path) -> pd.DataFrame:
    """Finds all models in base_dir and extracts their metadata.

    Args:
        base_dir: Directory containing models.

    Returns:
        DataFrame with artifact details.
    """
    base_path = Path(base_dir)
    if not base_path.exists() or not base_path.is_dir():
        return pd.DataFrame()
        
    records = []
    # Search for all metadata.json files in subdirectories
    for metadata_file in base_path.rglob("metadata.json"):
        artifact_dir = metadata_file.parent
        # Skip if the directory itself is 'models' or something
        try:
            with open(metadata_file, "r") as f:
                data = json.load(f)
                
            records.append({
                "artifact_id": artifact_dir.name,
                "path": str(artifact_dir),
                "agent_name": data.get("agent_name", "Unknown"),
                "schema_version": data.get("schema_version", "Unknown"),
                "created_at": data.get("created_at", "Unknown"),
            })
        except Exception:
            # Skip invalid or unreadable metadata
            pass
            
    if not records:
        return pd.DataFrame()
        
    df = pd.DataFrame(records)
    # Sort by created_at descending if possible
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True).dt.tz_localize(None)
        df = df.sort_values(by="created_at", ascending=False)
    return df

@st.cache_data(ttl=60)
def load_ope_report(artifact_path: str) -> Dict[str, Any]:
    """Loads ope_report.json from purely the artifact path."""
    path = Path(artifact_path) / "ope_report.json"
    if not path.exists():
        return {}
    
    with open(path, "r") as f:
        return json.load(f)

def _load_train_log_inner(log_path: Path) -> pd.DataFrame:
    """Helper to parse a jsonl or json file into a DataFrame."""
    try:
        with open(log_path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict) and "metrics" in data:
                return pd.DataFrame(data["metrics"])
    except json.JSONDecodeError:
        pass
        
    records = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return pd.DataFrame(records)

@st.cache_data(ttl=60)
def load_train_log(artifact_path: str) -> pd.DataFrame:
    """Loads train_log.jsonl from purely the artifact path."""
    path = Path(artifact_path) / "train_log.jsonl"
    if not path.exists():
        return pd.DataFrame()
    return _load_train_log_inner(path)

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
