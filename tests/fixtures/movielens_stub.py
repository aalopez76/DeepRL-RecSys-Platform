"""Synthetic MovieLens-like fixture for e2e testing.

Generates a small but realistic bandit-format dataset (100 interactions)
with user_ids, item_ids, ratings, timestamps, actions, rewards, and
propensity scores.  This avoids downloading real data in tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def generate_movielens_stub(
    output_dir: Path,
    n_interactions: int = 100,
    n_users: int = 20,
    n_items: int = 50,
    seed: int = 42,
) -> Path:
    """Generate a synthetic MovieLens-like dataset.

    Args:
        output_dir: Directory to write the dataset CSV.
        n_interactions: Number of interactions to generate.
        n_users: Number of unique users.
        n_items: Number of unique items.
        seed: Random seed for reproducibility.

    Returns:
        Path to the generated CSV file.
    """
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "user_id": rng.integers(1, n_users + 1, size=n_interactions),
        "item_id": rng.integers(1, n_items + 1, size=n_interactions),
        "action": rng.integers(0, n_items, size=n_interactions),
        "reward": rng.choice([0.0, 1.0], size=n_interactions, p=[0.7, 0.3]),
        "propensity": rng.uniform(0.05, 0.95, size=n_interactions).round(4),
        "timestamp": np.sort(rng.uniform(1e9, 1.1e9, size=n_interactions)).round(1),
    })

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "movielens_stub.csv"
    df.to_csv(csv_path, index=False)

    return csv_path
