"""Unit tests for core/schema.py — SchemaSpec, BANDIT_V1, SEQUENTIAL_V1, get_schema."""

from __future__ import annotations

import pytest

from deeprl_recsys.core.exceptions import SchemaError
from deeprl_recsys.core.schema import (
    BANDIT_V1,
    SEQUENTIAL_V1,
    SchemaSpec,
    get_schema,
    list_schemas,
)


@pytest.mark.unit
class TestSchemaSpec:
    """Tests for the SchemaSpec dataclass."""

    def test_schema_spec_is_frozen(self) -> None:
        """SchemaSpec instances must be immutable."""
        spec = SchemaSpec(schema_version="test_v1")
        with pytest.raises(AttributeError):
            spec.schema_version = "changed"  # type: ignore[misc]

    def test_bandit_v1_has_required_columns(self) -> None:
        """BANDIT_V1 must define action, reward, timestamp as required."""
        assert "action" in BANDIT_V1.required_columns
        assert "reward" in BANDIT_V1.required_columns
        assert "timestamp" in BANDIT_V1.required_columns

    def test_bandit_v1_propensity_is_optional(self) -> None:
        """Propensity must be optional in BANDIT_V1."""
        assert "propensity" in BANDIT_V1.optional_columns

    def test_sequential_v1_has_required_columns(self) -> None:
        """SEQUENTIAL_V1 must define user_id, item_id, rating, timestamp."""
        for col in ("user_id", "item_id", "rating", "timestamp"):
            assert col in SEQUENTIAL_V1.required_columns

    def test_bandit_v1_has_propensity_constraint(self) -> None:
        """BANDIT_V1 should define a propensity_range constraint."""
        assert "propensity_range" in BANDIT_V1.constraints
        lo, hi = BANDIT_V1.constraints["propensity_range"]
        assert lo == 0.0
        assert hi == 1.0


@pytest.mark.unit
class TestGetSchema:
    """Tests for the get_schema() lookup function."""

    def test_get_bandit_v1(self) -> None:
        """get_schema('bandit_v1') returns the correct spec."""
        spec = get_schema("bandit_v1")
        assert spec is BANDIT_V1

    def test_get_sequential_v1(self) -> None:
        """get_schema('sequential_v1') returns the correct spec."""
        spec = get_schema("sequential_v1")
        assert spec is SEQUENTIAL_V1

    def test_schema_version_unknown_raises(self) -> None:
        """Unknown schema version must raise SchemaError with available list."""
        with pytest.raises(SchemaError, match="Unknown schema version.*'nonexistent'"):
            get_schema("nonexistent")

    def test_unknown_schema_error_lists_available(self) -> None:
        """The error message should list all available schemas."""
        with pytest.raises(SchemaError, match="bandit_v1") as exc_info:
            get_schema("bad_version")
        assert "sequential_v1" in str(exc_info.value)


@pytest.mark.unit
class TestListSchemas:
    """Tests for list_schemas()."""

    def test_list_schemas_returns_sorted(self) -> None:
        """list_schemas() should return sorted version identifiers."""
        schemas = list_schemas()
        assert schemas == sorted(schemas)
        assert "bandit_v1" in schemas
        assert "sequential_v1" in schemas
