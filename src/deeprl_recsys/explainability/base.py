"""Base Explainer interface (stub)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseExplainer(ABC):
    """Abstract base for model explainers."""

    @abstractmethod
    def explain(self, model: Any, inputs: Any) -> dict[str, Any]:
        """Generate explanations for model predictions."""
