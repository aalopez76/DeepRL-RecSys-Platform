import json
import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse

# Scientific styling
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
PALETTE = sns.color_palette("colorblind")

def load_json_metrics(file_path: Path) -> pd.DataFrame:
    """Loads a JSON file with a 'metrics' key into a DataFrame."""
    with open(file_path, "r") as f:
        data = json.load(f)
    if "metrics" in data:
        return pd.DataFrame(data["metrics"])
    return pd.DataFrame()

def load_jsonl(file_path: Path) -> pd.DataFrame:
    """Loads a JSONL file into a DataFrame."""
    records = []
    if not file_path.exists():
        return pd.DataFrame()
    with open(file_path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)

def plot_curves(
    results_dir: Path, 
    smoothing_window: int = 10,
    output_name: str = "learning_curves"
):
    """Generates professional learning curves from multi-seed results."""
    
    # 1. Find all train_log.json and ope_intermediate.jsonl across seeds
    # Structure assumed: results_dir / agent_scenario / seed_X / ...
    
    train_dfs = []
    ope_dfs = []
    
    for log_file in results_dir.rglob("train_log.json"):
        df = load_json_metrics(log_file)
        if not df.empty:
            # Extract agent and seed from path or metadata
            # For simplicity, we assume agent name is in the path
            agent = log_file.parent.parent.name
            df["agent"] = agent
            train_dfs.append(df)
            
        ope_file = log_file.parent / "ope_intermediate.jsonl"
        if ope_file.exists():
            df_ope = load_jsonl(ope_file)
            if not df_ope.empty:
                df_ope["agent"] = agent
                ope_dfs.append(df_ope)

    if not train_dfs:
        print("No training logs found.")
        return

    df_train = pd.concat(train_dfs)
    
    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
    
    # Subplot 1: Average Reward
    ax = axes[0]
    sns.lineplot(
        data=df_train,
        x="step",
        y="reward",
        hue="agent",
        ax=ax,
        palette=PALETTE,
        # smoothing here would be better if applied per seed before plotting
        errorbar=("ci", 95)
    )
    
    # Baseline Random (Average)
    if "benchmark_random" in df_train["agent"].unique():
        random_val = df_train[df_train["agent"] == "benchmark_random"]["reward"].mean()
        ax.axhline(random_val, ls="--", color="gray", label="Random Baseline")

    ax.set_title("Learning Progress (Average Reward)", fontweight="bold")
    ax.set_ylabel("Average Reward")
    ax.set_xlabel("Training Steps")
    
    # Subplot 2: OPE Stability (IPS, DR, MIPS) for a specific agent (e.g., SAC)
    ax = axes[1]
    if ope_dfs:
        df_ope = pd.concat(ope_dfs)
        # Melt for plotting multiple estimators
        df_melt = df_ope.melt(id_vars=["step", "agent"], value_vars=["ips", "dr", "mips"], var_name="Estimator", value_name="Value")
        
        # Filter for a high-performing agent like SAC
        sac_ope = df_melt[df_melt["agent"].str.contains("sac", case=False)]
        if not sac_ope.empty:
            sns.lineplot(
                data=sac_ope,
                x="step",
                y="Value",
                hue="Estimator",
                ax=ax,
                palette=PALETTE,
                errorbar=("ci", 95)
            )
            ax.set_title("OPE Estimator Stability (SAC Agent)", fontweight="bold")
            ax.set_ylabel("Estimated Policy Value")
            ax.set_xlabel("Training Steps")

    plt.tight_layout()
    
    # Save in multiple formats
    out_dir = Path("docs/benchmarks/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(out_dir / f"{output_name}.png", dpi=300)
    plt.savefig(out_dir / f"{output_name}.pdf")
    print(f"Figures saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="artifacts/models")
    parser.add_argument("--smoothing_window", type=int, default=10)
    parser.add_argument("--output_name", type=str, default="learning_curves")
    args = parser.parse_args()
    
    plot_curves(Path(args.results_dir), args.smoothing_window, args.output_name)
