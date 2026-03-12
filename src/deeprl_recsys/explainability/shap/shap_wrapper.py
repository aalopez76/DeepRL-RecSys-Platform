"""SHAP wrapper — requires extra [explain] (stub)."""

from __future__ import annotations

from typing import Any

from deeprl_recsys.explainability.base import BaseExplainer


class SHAPExplainer(BaseExplainer):
    """SHAP-based model explainer (stub).

    Requires: ``pip install deeprl-recsys[explain]``
    """

    def __init__(self, **kwargs: Any) -> None:
        try:
            import shap  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "SHAPExplainer requires the [explain] extra. "
                "Install with: pip install deeprl-recsys[explain]"
            ) from exc

    def explain(self, model: Any, inputs: Any) -> dict[str, Any]:
        return {}  # Stub
