"""Gymnasium wrappers for RecSys environments (stub)."""

from __future__ import annotations

from typing import Any

from deeprl_recsys.environment.base import BaseEnvironment


class RecEnv(BaseEnvironment):
    """Gymnasium-compatible RecSys environment (stub)."""

    def __init__(self, **kwargs: Any) -> None:
        self._config = kwargs

    def reset(self) -> dict[str, Any]:
        return {"user_id": 0, "features": {}}

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        return {"user_id": 0, "features": {}}, 0.0, True, {}
