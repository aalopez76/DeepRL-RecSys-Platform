"""Serving Pydantic schemas — Request/Response models.

Defines the API contract for the recommendation serving endpoint.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


# ── Constants ────────────────────────────────────────────────────
MAX_CANDIDATES: int = 1000


class RecommendRequest(BaseModel):
    """Request body for ``POST /recommend``.

    Attributes:
        request_id: Unique request identifier for tracing.
        context: User/session context dictionary.
        candidates: Candidate item IDs to rank.
        k: Number of items to return (must be > 0).
    """

    request_id: str
    context: dict = Field(default_factory=dict)
    candidates: list[int] = Field(default_factory=list)
    k: int = Field(default=10, gt=0, description="Number of items to return")

    @field_validator("candidates")
    @classmethod
    def _validate_candidates(cls, v: list[int]) -> list[int]:
        if len(v) > MAX_CANDIDATES:
            raise ValueError(
                f"Too many candidates ({len(v)}), maximum is {MAX_CANDIDATES}"
            )
        return v


class RecommendItem(BaseModel):
    """A single recommended item with its score."""

    item_id: int
    score: float


class RecommendResponse(BaseModel):
    """Response body for ``POST /recommend``."""

    request_id: str
    items: list[RecommendItem]
    model_version: str = ""
    schema_version: str = ""
    latency_ms: float = 0.0


class InfoResponse(BaseModel):
    """Response body for ``GET /info``."""

    artifact_version: str = ""
    model_version: str = ""
    schema_version: str = ""
    agent_name: str = ""
    git_sha: str = ""
    checksums: dict[str, str] = Field(default_factory=dict)
    config_fingerprint: str = ""
