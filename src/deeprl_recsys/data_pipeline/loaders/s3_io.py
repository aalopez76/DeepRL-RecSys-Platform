"""S3 I/O — requires extra [aws] (boto3, s3fs).

Do NOT import this module unless the [aws] extra is installed.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def read_s3(path: str, **kwargs: Any) -> pd.DataFrame:
    """Read a file from S3 into a DataFrame.

    Requires: ``pip install deeprl-recsys[aws]``
    """
    try:
        import s3fs  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "S3 I/O requires the [aws] extra. "
            "Install with: pip install deeprl-recsys[aws]"
        ) from exc
    return pd.read_parquet(path, **kwargs)


def write_s3(df: pd.DataFrame, path: str, **kwargs: Any) -> None:
    """Write a DataFrame to S3.

    Requires: ``pip install deeprl-recsys[aws]``
    """
    try:
        import s3fs  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "S3 I/O requires the [aws] extra. "
            "Install with: pip install deeprl-recsys[aws]"
        ) from exc
    df.to_parquet(path, index=False, **kwargs)
