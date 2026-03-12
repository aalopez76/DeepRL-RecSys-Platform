"""Functional DQN Agent using PyTorch (if available).

If the [torch] extra is not installed, every method falls back gracefully
so tests and pipelines won't break without it.

**Lazy-loading contract:** importing this module does NOT import ``torch``.
Torch is resolved inside each method that requires it so that
``import deeprl_recsys.agents.dqn`` never pollutes ``sys.modules`` with
heavy optional dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deeprl_recsys.agents.base import BaseAgent
from deeprl_recsys.core.logging import get_logger

logger = get_logger(__name__)


def _try_import_torch() -> tuple[bool, Any]:
    """Return (has_torch, torch_module_or_None) without caching at module level."""
    try:
        import torch  # noqa: PLC0415

        return True, torch
    except (ImportError, OSError):
        return False, None


class DQNAgent(BaseAgent):
    """Deep Q-Network agent for item recommendation.

    Uses a simple embedding-based Q-network to rank candidate items.
    Falls back to passthrough ordering when PyTorch is not installed.
    """

    def __init__(
        self,
        num_items: int = 10000,
        embedding_dim: int = 32,
        lr: float = 1e-3,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initialise the DQN agent.

        Args:
            num_items: Size of the item catalogue (embedding table rows).
            embedding_dim: Dimension of each item embedding.
            lr: Adam learning rate.
            seed: Random seed for reproducibility.
            **kwargs: Additional config forwarded to ``_config``.
        """
        self._seed = seed
        self._num_items = num_items
        self._embedding_dim = embedding_dim
        self._lr = lr
        self._config = kwargs

        has_torch, torch = _try_import_torch()
        if has_torch:
            import torch.nn as nn  # noqa: PLC0415
            import torch.optim as optim  # noqa: PLC0415

            torch.manual_seed(seed)

            class _QNetwork(nn.Module):
                """Simple Q-Network representing item values."""

                def __init__(self, num_items: int, embedding_dim: int) -> None:
                    super().__init__()
                    self.item_emb = nn.Embedding(num_items, embedding_dim)
                    self.fc = nn.Linear(embedding_dim, 1)

                def forward(self, item_ids: Any) -> Any:
                    emb = self.item_emb(item_ids)
                    return self.fc(emb).squeeze(-1)

            self.q_net: Any = _QNetwork(num_items, embedding_dim)
            self.optimizer: Any = optim.Adam(self.q_net.parameters(), lr=lr)
        else:
            logger.warning("torch_missing", agent="dqn", msg="Falling back to stub")
            self.q_net = None
            self.optimizer = None

    def act(self, observation: dict[str, Any], candidates: list[int]) -> list[int]:
        """Select actions by greedy Q-value ranking.

        Args:
            observation: Context dictionary (unused in current implementation).
            candidates: List of candidate item IDs.

        Returns:
            Candidates sorted by descending Q-value, or unchanged if no torch.
        """
        has_torch, torch = _try_import_torch()
        if not has_torch or not candidates or self.q_net is None:
            return candidates

        self.q_net.eval()
        with torch.no_grad():
            c_tensor = torch.tensor(candidates, dtype=torch.long)
            q_values = self.q_net(c_tensor)
            sorted_idx = torch.argsort(q_values, descending=True)
            return [candidates[i] for i in sorted_idx.tolist()]

    def get_action_probabilities(
        self, observation: dict[str, Any], candidates: list[int]
    ) -> dict[int, float]:
        """Return a softmax distribution over candidates based on Q-values.

        Args:
            observation: Context dictionary.
            candidates: List of candidate item IDs.

        Returns:
            Dict mapping each candidate ID to its selection probability.
        """
        has_torch, torch = _try_import_torch()
        if not has_torch or not candidates or self.q_net is None:
            return super().get_action_probabilities(observation, candidates)

        import torch.nn.functional as F  # noqa: PLC0415

        self.q_net.eval()
        with torch.no_grad():
            c_tensor = torch.tensor(candidates, dtype=torch.long)
            q_values = self.q_net(c_tensor)
            probs = F.softmax(q_values, dim=0).tolist()
            return {c: p for c, p in zip(candidates, probs)}

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """Run one gradient-descent step on a dummy TD loss.

        Args:
            batch: Experience dictionary (keys unused in current stub).

        Returns:
            Dict with ``"loss"`` key.
        """
        has_torch, torch = _try_import_torch()
        if not has_torch or self.q_net is None:
            return {"loss": 0.0}

        self.q_net.train()
        self.optimizer.zero_grad()
        dummy_input = torch.tensor([0, 1, 2], dtype=torch.long)
        q_vals = self.q_net(dummy_input)
        loss = q_vals.mean() ** 2  # dummy TD loss
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def save(self, path: str) -> None:
        """Serialize Q-network weights to *path*.

        Args:
            path: Destination file path (parent dirs created automatically).
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        has_torch, torch = _try_import_torch()
        if has_torch and self.q_net is not None:
            torch.save(self.q_net.state_dict(), p)
        else:
            p.write_text(json.dumps({"agent": "dqn", "stub": True}))

    def load(self, path: str) -> None:
        """Load Q-network weights from *path*.

        Args:
            path: Source file path.
        """
        p = Path(path)
        has_torch, torch = _try_import_torch()
        if has_torch and self.q_net is not None and p.exists():
            try:
                self.q_net.load_state_dict(torch.load(p, weights_only=True))
            except Exception as exc:
                logger.warning("Agent load failed", error=str(exc))

    @property
    def name(self) -> str:
        """Registered agent name."""
        return "dqn"
