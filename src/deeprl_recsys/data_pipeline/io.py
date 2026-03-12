"""Unified I/O abstraction — dispatches to local or S3 backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from deeprl_recsys.data_pipeline.loaders.local_io import (
    read_csv,
    read_parquet,
    write_csv,
    write_parquet,
)


def read_dataset(path: str | Path, fmt: str = "csv", **kwargs: Any) -> pd.DataFrame:
    """Read a dataset from the given path.

    Args:
        path: File path (local or s3://).
        fmt: Format — ``"csv"`` or ``"parquet"``.
        **kwargs: Passed to the underlying reader.

    Returns:
        Loaded DataFrame.
    """
    path_str = str(path)
    if path_str.startswith("s3://"):
        from deeprl_recsys.data_pipeline.loaders.s3_io import read_s3
        return read_s3(path_str, **kwargs)
    if fmt == "parquet":
        return read_parquet(path_str, **kwargs)
    return read_csv(path_str, **kwargs)


def write_dataset(
    df: pd.DataFrame, path: str | Path, fmt: str = "csv", **kwargs: Any
) -> None:
    """Write a dataset to the given path.

    Args:
        df: DataFrame to write.
        path: File path (local or s3://).
        fmt: Format — ``"csv"`` or ``"parquet"``.
    """
    path_str = str(path)
    if path_str.startswith("s3://"):
        from deeprl_recsys.data_pipeline.loaders.s3_io import write_s3
        write_s3(df, path_str, **kwargs)
        return
    if fmt == "parquet":
        write_parquet(df, path_str, **kwargs)
    else:
        write_csv(df, path_str, **kwargs)
