"""Dataset validation against :mod:`core.schema`.

Validates a :class:`~pandas.DataFrame` against a versioned
:class:`~core.schema.SchemaSpec`, checking required columns,
dtype compatibility, value ranges, and propensity policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from deeprl_recsys.core.logging import get_logger
from deeprl_recsys.core.schema import SchemaSpec, get_schema
from deeprl_recsys.core.validators import (
    check_dtype_compatible,
    check_range,
    check_required_columns,
)

logger = get_logger(__name__)

# Threshold above which a performance warning is emitted
LARGE_DATASET_THRESHOLD: int = 1_000_000


@dataclass
class ValidationResult:
    """Result of validating a dataset against a schema.

    Attributes:
        errors: Hard errors that prevent usage.
        warnings: Soft warnings (e.g. missing optional columns).
        schema_version: The schema version that was validated against.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    schema_version: str = ""

    @property
    def is_valid(self) -> bool:
        """``True`` when there are no errors."""
        return len(self.errors) == 0


def validate_dataset(
    df: pd.DataFrame,
    schema_version: str,
    propensity_policy: Literal["mark_unreliable", "block_ope"] = "mark_unreliable",
) -> ValidationResult:
    """Validate a DataFrame against a versioned schema.

    Performs the following checks in order:

    1. **Required columns** — all must be present.
    2. **Dtype compatibility** — column dtypes must match schema expectations.
    3. **Null checks** — ``action`` column must not contain nulls (bandit_v1).
    4. **Propensity policy** — handles missing propensity per configured policy.
    5. **Range checks** — propensity ∈ (0, 1], reward/rating within declared range.

    Args:
        df: Input DataFrame to validate.
        schema_version: Schema version identifier (e.g. ``"bandit_v1"``).
        propensity_policy: Behaviour when propensity column is absent:

            - ``"mark_unreliable"`` — add a warning but remain valid.
            - ``"block_ope"`` — add an error (invalid for OPE).

    Returns:
        A :class:`ValidationResult` with accumulated errors and warnings.

    Raises:
        SchemaError: If *schema_version* is unknown (propagated from
            :func:`core.schema.get_schema`).
    """
    spec = get_schema(schema_version)
    result = ValidationResult(schema_version=schema_version)

    # ── Large dataset warning ────────────────────────────────────
    if len(df) > LARGE_DATASET_THRESHOLD:
        logger.warning(
            "large_dataset",
            n_rows=len(df),
            threshold=LARGE_DATASET_THRESHOLD,
            msg="Dataset exceeds threshold; validation may be slow",
        )

    # ── 1. Required columns ──────────────────────────────────────
    result.errors.extend(
        check_required_columns(list(df.columns), list(spec.required_columns))
    )

    # ── 2. Dtype compatibility (only for present columns) ────────
    for col, expected_kind in spec.required_columns.items():
        if col in df.columns:
            actual_kind = df[col].dtype.kind
            result.errors.extend(
                check_dtype_compatible(actual_kind, expected_kind, column=col)
            )

    # ── 3. Null checks for critical columns ──────────────────────
    if "action" in df.columns and df["action"].isna().any():
        null_count = int(df["action"].isna().sum())
        result.errors.append(
            f"Column 'action' contains {null_count} null value(s)"
        )

    # ── 4. Propensity policy ─────────────────────────────────────
    if "propensity" in spec.optional_columns and "propensity" not in df.columns:
        if propensity_policy == "block_ope":
            result.errors.append(
                "Propensity column missing — OPE blocked by policy"
            )
        else:
            result.warnings.append(
                "Propensity column missing — OPE results marked unreliable"
            )

    # ── 5. Range checks ──────────────────────────────────────────
    _check_propensity_range(df, result)
    _check_constraint_ranges(df, spec, result)

    return result


# ── Internal helpers ────────────────────────────────────────────


def _check_propensity_range(df: pd.DataFrame, result: ValidationResult) -> None:
    """Validate propensity values are in (0, 1]."""
    if "propensity" not in df.columns:
        return

    prop = df["propensity"].dropna()
    if len(prop) == 0:
        return

    prop_min = float(prop.min())
    prop_max = float(prop.max())

    # propensity must be > 0 (exclusive)
    result.errors.extend(
        check_range(prop_min, name="propensity (min)", low=0.0, low_inclusive=False)
    )
    # propensity must be <= 1 (inclusive)
    result.errors.extend(
        check_range(prop_max, name="propensity (max)", high=1.0, high_inclusive=True)
    )


def _check_constraint_ranges(
    df: pd.DataFrame, spec: SchemaSpec, result: ValidationResult
) -> None:
    """Apply named range constraints from the schema spec."""
    for constraint_name, bounds in spec.constraints.items():
        if not constraint_name.endswith("_range"):
            continue
        col_name = constraint_name.replace("_range", "")
        if col_name not in df.columns:
            continue
        low, high = bounds
        col_min = float(df[col_name].min())
        col_max = float(df[col_name].max())
        if col_min < low:
            result.warnings.append(
                f"Column '{col_name}': minimum ({col_min}) below expected lower bound ({low})"
            )
        if col_max > high:
            result.warnings.append(
                f"Column '{col_name}': maximum ({col_max}) above expected upper bound ({high})"
            )
