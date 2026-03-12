"""Data transforms — normalisation, feature engineering, ID mapping.

Stub for Phase 2+ implementation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def normalize_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Min-max normalise the specified columns.

    Args:
        df: Input DataFrame.
        columns: Columns to normalise.

    Returns:
        DataFrame with normalised columns.
    """
    result = df.copy()
    for col in columns:
        min_val = result[col].min()
        max_val = result[col].max()
        if max_val > min_val:
            result[col] = (result[col] - min_val) / (max_val - min_val)
    return result
