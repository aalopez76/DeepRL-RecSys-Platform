"""Local I/O — CSV and Parquet readers/writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV file into a DataFrame."""
    return pd.read_csv(path, **kwargs)


def read_parquet(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame."""
    return pd.read_parquet(path, **kwargs)


def write_csv(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    """Write a DataFrame to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)


def write_parquet(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    """Write a DataFrame to Parquet."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, **kwargs)
