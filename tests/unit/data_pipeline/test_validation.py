"""Unit tests for data_pipeline/validation.py — dataset validation contract.

Required tests from implementación.txt Phase 2:
- test_bandit_v1_missing_propensity_marks_unreliable_or_blocks
- test_propensity_out_of_range_fails
- test_missing_required_column_fails_with_message
- test_schema_version_unknown_raises
Plus additional coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deeprl_recsys.core.exceptions import SchemaError
from deeprl_recsys.data_pipeline.validation import (
    LARGE_DATASET_THRESHOLD,
    ValidationResult,
    validate_dataset,
)


# ── Helpers ──────────────────────────────────────────


def _make_bandit_df(
    *,
    include_propensity: bool = True,
    propensity_values: list[float] | None = None,
    action_nulls: bool = False,
) -> pd.DataFrame:
    """Build a minimal bandit_v1 DataFrame for testing."""
    n = 5
    data: dict[str, list] = {
        "action": [1, 2, 3, 4, 5],
        "reward": [0.1, 0.0, 1.0, 0.5, 0.3],
        "timestamp": [1000.0, 1001.0, 1002.0, 1003.0, 1004.0],
    }
    if include_propensity:
        data["propensity"] = propensity_values or [0.2, 0.5, 0.3, 0.8, 0.1]
    if action_nulls:
        data["action"][2] = None  # type: ignore[assignment]
    return pd.DataFrame(data)


def _make_sequential_df() -> pd.DataFrame:
    """Build a minimal sequential_v1 DataFrame."""
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "item_id": [10, 20, 30],
            "rating": [4.5, 3.0, 1.5],
            "timestamp": [100.0, 200.0, 300.0],
        }
    )


# ── Contract Tests (from implementación.txt) ─────────


@pytest.mark.unit
class TestPropensityPolicy:
    """Tests for propensity-missing behaviour."""

    def test_bandit_v1_missing_propensity_marks_unreliable(self) -> None:
        """Missing propensity with mark_unreliable → warning, still valid."""
        df = _make_bandit_df(include_propensity=False)
        result = validate_dataset(df, "bandit_v1", propensity_policy="mark_unreliable")
        assert result.is_valid
        assert any("unreliable" in w.lower() for w in result.warnings)

    def test_bandit_v1_missing_propensity_blocks_ope(self) -> None:
        """Missing propensity with block_ope → error, NOT valid."""
        df = _make_bandit_df(include_propensity=False)
        result = validate_dataset(df, "bandit_v1", propensity_policy="block_ope")
        assert not result.is_valid
        assert any("blocked" in e.lower() for e in result.errors)


@pytest.mark.unit
class TestPropensityRange:
    """Tests for propensity range validation."""

    def test_propensity_out_of_range_zero_fails(self) -> None:
        """Propensity == 0.0 must fail (must be > 0)."""
        df = _make_bandit_df(propensity_values=[0.0, 0.5, 0.3, 0.8, 0.1])
        result = validate_dataset(df, "bandit_v1")
        assert not result.is_valid
        assert any("propensity" in e.lower() for e in result.errors)

    def test_propensity_negative_fails(self) -> None:
        """Propensity < 0 must fail."""
        df = _make_bandit_df(propensity_values=[-0.1, 0.5, 0.3, 0.8, 0.1])
        result = validate_dataset(df, "bandit_v1")
        assert not result.is_valid

    def test_propensity_above_one_fails(self) -> None:
        """Propensity > 1.0 must fail."""
        df = _make_bandit_df(propensity_values=[0.2, 0.5, 0.3, 1.5, 0.1])
        result = validate_dataset(df, "bandit_v1")
        assert not result.is_valid

    def test_propensity_valid_range_passes(self) -> None:
        """Propensity in (0, 1] must pass."""
        df = _make_bandit_df(propensity_values=[0.01, 0.5, 1.0, 0.3, 0.99])
        result = validate_dataset(df, "bandit_v1")
        assert result.is_valid


@pytest.mark.unit
class TestRequiredColumns:
    """Tests for required column checks."""

    def test_missing_required_column_fails_with_message(self) -> None:
        """Missing a required column must produce an error naming the column."""
        df = pd.DataFrame({"action": [1], "reward": [0.5]})
        # 'timestamp' is missing
        result = validate_dataset(df, "bandit_v1")
        assert not result.is_valid
        assert any("timestamp" in e for e in result.errors)

    def test_all_required_present_passes(self) -> None:
        """A valid DataFrame with all required columns must pass."""
        df = _make_bandit_df()
        result = validate_dataset(df, "bandit_v1")
        assert result.is_valid

    def test_multiple_missing_columns_all_reported(self) -> None:
        """All missing columns must appear in errors, not just the first."""
        df = pd.DataFrame({"user_id": [1]})
        # sequential_v1 needs: user_id, item_id, rating, timestamp
        result = validate_dataset(df, "sequential_v1")
        assert not result.is_valid
        missing_cols = [e for e in result.errors if "Missing" in e]
        assert len(missing_cols) >= 3  # item_id, rating, timestamp


@pytest.mark.unit
class TestSchemaVersionValidation:
    """Tests for unknown schema versions."""

    def test_schema_version_unknown_raises(self) -> None:
        """Passing an unknown schema_version must raise SchemaError."""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(SchemaError, match="Unknown schema version"):
            validate_dataset(df, "nonexistent_v99")


@pytest.mark.unit
class TestDtypeValidation:
    """Tests for dtype compatibility checking."""

    def test_string_column_where_int_expected_fails(self) -> None:
        """A string column where int is expected must raise a dtype error."""
        df = pd.DataFrame(
            {
                "action": ["a", "b", "c"],
                "reward": [0.1, 0.2, 0.3],
                "timestamp": [1.0, 2.0, 3.0],
            }
        )
        result = validate_dataset(df, "bandit_v1")
        assert not result.is_valid
        assert any("dtype" in e.lower() for e in result.errors)

    def test_int_column_accepted_for_float(self) -> None:
        """An int column where float is expected must be accepted."""
        df = pd.DataFrame(
            {
                "action": [1, 2, 3],
                "reward": [0, 1, 0],       # int, acceptable for float
                "timestamp": [100, 200, 300],  # int, acceptable for float
            }
        )
        result = validate_dataset(df, "bandit_v1")
        assert result.is_valid


@pytest.mark.unit
class TestNullChecks:
    """Tests for null value detection."""

    def test_action_nulls_produce_error(self) -> None:
        """Null values in 'action' column must produce an error."""
        df = _make_bandit_df(action_nulls=True)
        result = validate_dataset(df, "bandit_v1")
        assert not result.is_valid
        assert any("null" in e.lower() for e in result.errors)


@pytest.mark.unit
class TestValidationResult:
    """Tests for the ValidationResult dataclass."""

    def test_empty_result_is_valid(self) -> None:
        """A freshly-created ValidationResult with no errors is valid."""
        r = ValidationResult()
        assert r.is_valid

    def test_result_with_errors_is_invalid(self) -> None:
        """A result with errors is not valid."""
        r = ValidationResult(errors=["something went wrong"])
        assert not r.is_valid

    def test_result_with_only_warnings_is_valid(self) -> None:
        """Warnings alone do not make a result invalid."""
        r = ValidationResult(warnings=["minor issue"])
        assert r.is_valid


@pytest.mark.unit
class TestSequentialV1:
    """Tests for sequential_v1 schema validation."""

    def test_valid_sequential_passes(self) -> None:
        """A valid sequential_v1 DataFrame must pass."""
        df = _make_sequential_df()
        result = validate_dataset(df, "sequential_v1")
        assert result.is_valid

    def test_sequential_missing_item_id(self) -> None:
        """Missing item_id in sequential_v1 must fail."""
        df = pd.DataFrame(
            {"user_id": [1], "rating": [3.0], "timestamp": [100.0]}
        )
        result = validate_dataset(df, "sequential_v1")
        assert not result.is_valid
        assert any("item_id" in e for e in result.errors)
