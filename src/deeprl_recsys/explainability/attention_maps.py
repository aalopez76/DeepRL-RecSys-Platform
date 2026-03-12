"""Attention maps explainer — requires [explain] or [llm] (stub)."""

from __future__ import annotations

from typing import Any

from deeprl_recsys.explainability.base import BaseExplainer


class AttentionMapsExplainer(BaseExplainer):
    """Attention-based explainer (stub)."""

    def explain(self, model: Any, inputs: Any) -> dict[str, Any]:
        return {}  # Stub
