"""Core data types for the platform.

Defines canonical Pydantic models used across every layer:
``LoggedEvent``, ``Observation``, and ``Action``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """Represents the context/observation for making a recommendation."""

    user_id: str | int
    features: dict[str, Any] = Field(default_factory=dict)
    timestamp: float | None = None


class Action(BaseModel):
    """Represents a recommendation action."""

    item_id: str | int
    score: float | None = None


class LoggedEvent(BaseModel):
    """A single logged interaction from a behaviour policy.

    Attributes:
        observation: The context at recommendation time.
        action: The action taken by the behaviour policy.
        reward: The observed reward (e.g., click, purchase).
        propensity: The probability the behaviour policy assigned to this action.
        timestamp: Unix timestamp of the event.
    """

    observation: Observation
    action: Action
    reward: float
    propensity: float | None = None
    timestamp: float | None = None
