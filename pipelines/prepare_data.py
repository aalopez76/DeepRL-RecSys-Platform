"""Pipeline: prepare_data — load, validate, and persist a clean dataset.

Orchestrates:
1. Load raw data (CSV/Parquet)
2. Validate against a versioned schema
3. Persist clean dataset + validation report
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from deeprl_recsys.core.logging import get_logger
from deeprl_recsys.data_pipeline.validation import ValidationResult, validate_dataset

logger = get_logger(__name__)


def run_prepare(config: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    """Execute the data preparation pipeline.

    Args:
        config: Resolved platform config.  Expected keys:

            - ``dataset.data_path`` — path to raw data (CSV or Parquet)
            - ``dataset.schema_version`` — schema version for validation
            - ``dataset.output_dir`` — directory for cleaned output
            - ``dataset.propensity_policy`` — ``"mark_unreliable"`` or
              ``"block_ope"``

    Returns:
        Dictionary with ``is_valid``, ``n_rows``, ``errors``, ``warnings``,
        and ``output_path``.
    """
    dataset_cfg = config.get("dataset", {})
    data_path = Path(dataset_cfg.get("data_path", ""))
    schema_version = dataset_cfg.get("schema_version", "bandit_v1")
    output_dir = Path(dataset_cfg.get("output_dir", "data/clean"))
    propensity_policy = dataset_cfg.get("propensity_policy", "mark_unreliable")

    logger.info("prepare_start", data_path=str(data_path), schema=schema_version)

    # 1. Load
    df = _load_data(data_path)
    logger.info("prepare_loaded", n_rows=len(df), n_cols=len(df.columns))

    # 2. Validate
    result = validate_dataset(df, schema_version, propensity_policy=propensity_policy)

    if not result.is_valid:
        logger.error("prepare_validation_failed", errors=result.errors)
        return {
            "is_valid": False,
            "n_rows": len(df),
            "errors": result.errors,
            "warnings": result.warnings,
            "output_path": "",
        }

    # 3. Persist
    if dry_run:
        logger.info("prepare_dry_run", msg="Skipping write operations")
        output_path = ""
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "dataset_clean.csv"
        df.to_csv(output_path, index=False)

        # Validation report
        report_path = output_dir / "validation_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "schema_version": schema_version,
                    "n_rows": len(df),
                    "is_valid": True,
                    "errors": result.errors,
                    "warnings": result.warnings,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    logger.info("prepare_done", output=str(output_path), warnings=len(result.warnings))

    return {
        "is_valid": True,
        "n_rows": len(df),
        "errors": result.errors,
        "warnings": result.warnings,
        "output_path": str(output_path),
    }


def _load_data(path: Path) -> pd.DataFrame:
    """Load a DataFrame from CSV or Parquet."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    elif suffix == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


if __name__ == "__main__":
    run_prepare({})
