"""Versioned dataset schemas — ``bandit_v1``, ``sequential_v1``.

Defines :class:`SchemaSpec` as a frozen declarative specification and
provides :func:`get_schema` for version-based lookup.  These schemas are
the **single source of truth** consumed by :mod:`data_pipeline.validation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from deeprl_recsys.core.exceptions import SchemaError


@dataclass(frozen=True)
class SchemaSpec:
    """Declarative schema specification for a dataset version.

    Attributes:
        schema_version: Unique version identifier (e.g. ``"bandit_v1"``).
        required_columns: Mapping of ``column_name → dtype_kind``
            where *dtype_kind* is the single-char NumPy dtype kind
            (``"i"`` = int, ``"f"`` = float, ``"O"`` = object, etc.).
        optional_columns: Columns that may appear but are not required.
        constraints: Named validation rules such as ``"propensity_range"``
            or ``"reward_range"`` expressed as ``(low, high)`` tuples.
    """

    schema_version: str
    required_columns: dict[str, str] = field(default_factory=dict)
    optional_columns: dict[str, str] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)


# ── Schema Definitions ──────────────────────────────────────────

BANDIT_V1 = SchemaSpec(
    schema_version="bandit_v1",
    required_columns={
        "action": "i",
        "reward": "f",
        "timestamp": "f",
    },
    optional_columns={
        "propensity": "f",
        "user_id": "i",
        "context": "O",
    },
    constraints={
        "propensity_range": (0.0, 1.0),  # exclusive low, inclusive high
        "reward_range": (0.0, 1.0),
    },
)

SEQUENTIAL_V1 = SchemaSpec(
    schema_version="sequential_v1",
    required_columns={
        "user_id": "i",
        "item_id": "i",
        "rating": "f",
        "timestamp": "f",
    },
    optional_columns={
        "context": "O",
    },
    constraints={
        "rating_range": (0.0, 5.0),
    },
)

# ── Registry ────────────────────────────────────────────────────

_SCHEMAS: dict[str, SchemaSpec] = {
    "bandit_v1": BANDIT_V1,
    "sequential_v1": SEQUENTIAL_V1,
}


def get_schema(schema_version: str) -> SchemaSpec:
    """Retrieve a :class:`SchemaSpec` by version string.

    Args:
        schema_version: Version identifier (e.g. ``"bandit_v1"``).

    Returns:
        The corresponding :class:`SchemaSpec`.

    Raises:
        SchemaError: If the version is not recognised.
    """
    spec = _SCHEMAS.get(schema_version)
    if spec is None:
        available = ", ".join(sorted(_SCHEMAS))
        raise SchemaError(
            f"Unknown schema version: {schema_version!r}. Available: [{available}]"
        )
    return spec


def list_schemas() -> list[str]:
    """Return sorted list of available schema version identifiers.

    Returns:
        Sorted list of schema version strings.
    """
    return sorted(_SCHEMAS)
