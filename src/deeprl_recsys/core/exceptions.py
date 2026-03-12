"""Custom exceptions for the DeepRL-RecSys platform.

All exceptions inherit from ``PlatformError`` to allow catching any
platform-specific error with a single except clause.
"""

from __future__ import annotations


class PlatformError(Exception):
    """Base exception for all DeepRL-RecSys errors."""


class ConfigError(PlatformError):
    """Raised when configuration loading, merging, or validation fails.

    Attributes:
        field: The config field that caused the error (if known).
        source: The file path or source that triggered the error.
    """

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        source: str | None = None,
    ) -> None:
        parts: list[str] = []
        if source:
            parts.append(f"[source={source}]")
        if field:
            parts.append(f"[field={field}]")
        parts.append(message)
        super().__init__(" ".join(parts))
        self.field = field
        self.source = source


class SchemaError(PlatformError):
    """Raised when a dataset does not conform to its declared schema."""


class ArtifactError(PlatformError):
    """Raised on artifact save/load failures (missing files, checksum mismatch, etc.)."""


class OPEError(PlatformError):
    """Raised when Off-Policy Evaluation cannot proceed reliably."""
