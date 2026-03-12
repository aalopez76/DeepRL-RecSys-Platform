"""TF-Agents adapter — requires extra [tf].

Do NOT import this module unless the [tf] extra is installed.
"""

from __future__ import annotations

from typing import Any

from deeprl_recsys.agents.base import BaseAgent


class TFAgentsAdapter(BaseAgent):
    """Adapter wrapping a TF-Agents policy as a BaseAgent (stub).

    Requires: ``pip install deeprl-recsys[tf]``
    """

    def __init__(self, **kwargs: Any) -> None:
        try:
            import tf_agents  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "TFAgentsAdapter requires the [tf] extra. "
                "Install with: pip install deeprl-recsys[tf]"
            ) from exc
        self._config = kwargs

    def act(self, observation: dict[str, Any], candidates: list[int]) -> list[int]:
        return candidates

    @property
    def name(self) -> str:
        return "tf_agents"
