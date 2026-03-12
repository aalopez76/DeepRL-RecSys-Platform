"""Shared validators for types, ranges, and constraints.

Used by :mod:`data_pipeline.validation` and other modules that need
reusable validation logic.
"""

from __future__ import annotations

from typing import Any, Sequence


def check_range(
    value: float,
    *,
    name: str,
    low: float | None = None,
    high: float | None = None,
    low_inclusive: bool = True,
    high_inclusive: bool = True,
) -> list[str]:
    """Validate that *value* falls within [low, high].

    Args:
        value: The numeric value to check.
        name: Human-readable field name for error messages.
        low: Lower bound (``None`` = no lower bound).
        high: Upper bound (``None`` = no upper bound).
        low_inclusive: If ``True``, ``value >= low``; else ``value > low``.
        high_inclusive: If ``True``, ``value <= high``; else ``value < high``.

    Returns:
        List of error strings (empty if valid).
    """
    errors: list[str] = []
    if low is not None:
        if low_inclusive and value < low:
            errors.append(f"{name}: expected >= {low}, got {value}")
        elif not low_inclusive and value <= low:
            errors.append(f"{name}: expected > {low}, got {value}")
    if high is not None:
        if high_inclusive and value > high:
            errors.append(f"{name}: expected <= {high}, got {value}")
        elif not high_inclusive and value < high:
            pass  # valid
        elif not high_inclusive and value >= high:
            errors.append(f"{name}: expected < {high}, got {value}")
    return errors


def check_required_columns(
    columns: Sequence[str],
    required: Sequence[str],
) -> list[str]:
    """Check that all *required* column names are present in *columns*.

    Args:
        columns: Column names present in the dataframe.
        required: Column names that must be present.

    Returns:
        List of error strings for missing columns.
    """
    present = set(columns)
    return [f"Missing required column: {col!r}" for col in required if col not in present]


def check_dtype_compatible(
    actual_kind: str,
    expected_kind: str,
    *,
    column: str,
) -> list[str]:
    """Check that a column's dtype kind matches the expected kind.

    Args:
        actual_kind: Single-char NumPy dtype kind (e.g. ``"f"``, ``"i"``).
        expected_kind: Expected dtype kind.
        column: Column name for error messages.

    Returns:
        List of error strings (empty if compatible).
    """
    compatible_groups: dict[str, set[str]] = {
        "f": {"f", "i", "u"},  # float accepts int/uint
        "i": {"i", "u"},
        "u": {"u", "i"},
        "U": {"U", "O"},  # string accepts object
        "O": {"O", "U"},
    }
    allowed = compatible_groups.get(expected_kind, {expected_kind})
    if actual_kind not in allowed:
        return [f"Column {column!r}: expected dtype kind {expected_kind!r}, got {actual_kind!r}"]
    return []
