"""Data splitting strategies — temporal, user-based, random.

Stub for Phase 2+ implementation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame by time.

    Args:
        df: Input DataFrame sorted by timestamp.
        timestamp_col: Name of timestamp column.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.

    Returns:
        Tuple of (train, val, test) DataFrames.
    """
    n = len(df)
    sorted_df = df.sort_values(timestamp_col).reset_index(drop=True)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return sorted_df[:train_end], sorted_df[train_end:val_end], sorted_df[val_end:]
