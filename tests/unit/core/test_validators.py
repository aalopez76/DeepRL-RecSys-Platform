"""Unit tests for core/validators.py — coverage hardening.

Targets uncovered lines: check_range with low_inclusive=True/False,
high_inclusive=True/False, and check_dtype_compatible edge cases.
"""

from __future__ import annotations

import pytest

from deeprl_recsys.core.validators import (
    check_dtype_compatible,
    check_range,
    check_required_columns,
)


@pytest.mark.unit
class TestCheckRangeLowInclusive:
    """Tests for check_range with low_inclusive=True (default)."""

    def test_value_below_low_inclusive_fails(self) -> None:
        """value < low with low_inclusive=True must produce error."""
        errs = check_range(4.9, name="test", low=5.0, low_inclusive=True)
        assert len(errs) == 1
        assert ">=" in errs[0]

    def test_value_at_low_inclusive_passes(self) -> None:
        """value == low with low_inclusive=True must pass."""
        errs = check_range(5.0, name="test", low=5.0, low_inclusive=True)
        assert len(errs) == 0

    def test_value_above_low_inclusive_passes(self) -> None:
        errs = check_range(5.1, name="test", low=5.0, low_inclusive=True)
        assert len(errs) == 0


@pytest.mark.unit
class TestCheckRangeLowExclusive:
    """Tests for check_range with low_inclusive=False."""

    def test_value_at_low_exclusive_fails(self) -> None:
        """value == low with low_inclusive=False must produce error."""
        errs = check_range(5.0, name="test", low=5.0, low_inclusive=False)
        assert len(errs) == 1
        assert ">" in errs[0]

    def test_value_below_low_exclusive_fails(self) -> None:
        errs = check_range(4.9, name="test", low=5.0, low_inclusive=False)
        assert len(errs) == 1

    def test_value_above_low_exclusive_passes(self) -> None:
        errs = check_range(5.1, name="test", low=5.0, low_inclusive=False)
        assert len(errs) == 0


@pytest.mark.unit
class TestCheckRangeHighInclusive:
    """Tests for check_range with high_inclusive=True (default)."""

    def test_value_above_high_inclusive_fails(self) -> None:
        errs = check_range(10.1, name="test", high=10.0, high_inclusive=True)
        assert len(errs) == 1
        assert "<=" in errs[0]

    def test_value_at_high_inclusive_passes(self) -> None:
        errs = check_range(10.0, name="test", high=10.0, high_inclusive=True)
        assert len(errs) == 0


@pytest.mark.unit
class TestCheckRangeHighExclusive:
    """Tests for check_range with high_inclusive=False."""

    def test_value_at_high_exclusive_fails(self) -> None:
        """value == high with high_inclusive=False must produce error."""
        errs = check_range(10.0, name="test", high=10.0, high_inclusive=False)
        assert len(errs) == 1
        assert "<" in errs[0]

    def test_value_above_high_exclusive_fails(self) -> None:
        errs = check_range(10.1, name="test", high=10.0, high_inclusive=False)
        assert len(errs) == 1

    def test_value_below_high_exclusive_passes(self) -> None:
        """value < high with high_inclusive=False must pass."""
        errs = check_range(9.9, name="test", high=10.0, high_inclusive=False)
        assert len(errs) == 0


@pytest.mark.unit
class TestCheckRangeCombined:
    """Tests for check_range with both low and high bounds."""

    def test_value_in_range_passes(self) -> None:
        errs = check_range(5.0, name="test", low=0.0, high=10.0)
        assert len(errs) == 0

    def test_no_bounds_always_passes(self) -> None:
        errs = check_range(999.0, name="test")
        assert len(errs) == 0

    def test_both_exclusive_at_boundary_fails(self) -> None:
        errs = check_range(
            10.0, name="test", low=0.0, high=10.0,
            low_inclusive=False, high_inclusive=False,
        )
        assert len(errs) == 1  # at high exclusive boundary


@pytest.mark.unit
class TestCheckDtypeCompatible:
    """Tests for check_dtype_compatible with diverse type kinds."""

    @pytest.mark.parametrize("actual,expected", [
        ("f", "f"),  # float == float
        ("i", "f"),  # int compatible with float
        ("u", "f"),  # uint compatible with float
        ("i", "i"),  # int == int
        ("u", "i"),  # uint compatible with int
        ("U", "U"),  # string == string
        ("O", "U"),  # object compatible with string
        ("U", "O"),  # string compatible with object
    ])
    def test_compatible_types_pass(self, actual: str, expected: str) -> None:
        errs = check_dtype_compatible(actual, expected, column="col")
        assert len(errs) == 0

    @pytest.mark.parametrize("actual,expected", [
        ("U", "f"),  # string not compatible with float
        ("f", "i"),  # float not compatible with int
        ("O", "i"),  # object not compatible with int
        ("b", "f"),  # boolean kind not in float group
    ])
    def test_incompatible_types_fail(self, actual: str, expected: str) -> None:
        errs = check_dtype_compatible(actual, expected, column="col")
        assert len(errs) == 1
        assert "dtype" in errs[0].lower()

    def test_unknown_expected_kind_uses_exact_match(self) -> None:
        """Unknown expected_kind should only match itself."""
        errs = check_dtype_compatible("x", "x", column="col")
        assert len(errs) == 0
        errs = check_dtype_compatible("y", "x", column="col")
        assert len(errs) == 1


@pytest.mark.unit
class TestCheckRequiredColumns:
    """Tests for check_required_columns."""

    def test_all_present_returns_empty(self) -> None:
        errs = check_required_columns(["a", "b", "c"], ["a", "b"])
        assert len(errs) == 0

    def test_missing_returns_errors(self) -> None:
        errs = check_required_columns(["a"], ["a", "b", "c"])
        assert len(errs) == 2
        assert all("Missing" in e for e in errs)

    def test_empty_required_passes(self) -> None:
        errs = check_required_columns(["a", "b"], [])
        assert len(errs) == 0
