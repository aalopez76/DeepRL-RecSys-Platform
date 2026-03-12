"""Functional SAC Agent using PyTorch (if available).

If the [torch] extra is not installed, every method falls back gracefully.

**Lazy-loading contract:** importing this module does NOT import ``torch``.
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


class SACAgent(BaseAgent):
    """Soft Actor-Critic agent for discrete item recommendation.

    Uses twin Q-networks plus a stochastic policy over item embeddings.
    Falls back to passthrough ordering when PyTorch is not installed.
    """

    def __init__(
        self,
        num_items: int = 10000,
        embedding_dim: int = 32,
        lr: float = 3e-4,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initialise the SAC agent.

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
            import torch.nn.functional as F  # noqa: PLC0415
            import torch.optim as optim  # noqa: PLC0415

            torch.manual_seed(seed)

            class _SACActorCritic(nn.Module):
                """Discrete Soft Actor-Critic networks."""

                def __init__(self, num_items: int, embedding_dim: int) -> None:
                    super().__init__()
                    self.item_emb = nn.Embedding(num_items, embedding_dim)
                    self.q1 = nn.Linear(embedding_dim, 1)
                    self.q2 = nn.Linear(embedding_dim, 1)
                    self.actor = nn.Linear(embedding_dim, 1)

                def get_q_values(self, item_ids: Any) -> tuple[Any, Any]:
                    emb = self.item_emb(item_ids)
                    return self.q1(emb).squeeze(-1), self.q2(emb).squeeze(-1)

                def get_action_probs(self, item_ids: Any) -> Any:
                    emb = self.item_emb(item_ids)
                    logits = self.actor(emb).squeeze(-1)
                    return F.softmax(logits, dim=0)

            self.network: Any = _SACActorCritic(num_items, embedding_dim)
            self.optimizer: Any = optim.Adam(self.network.parameters(), lr=lr)
        else:
            logger.warning("torch_missing", agent="sac", msg="Falling back to stub")
            self.network = None
            self.optimizer = None

    def act(self, observation: dict[str, Any], candidates: list[int]) -> list[int]:
        """Select actions by greedy policy-probability ranking.

        Args:
            observation: Context dictionary (unused in current implementation).
            candidates: List of candidate item IDs.

        Returns:
            Candidates sorted by descending policy probability, or unchanged.
        """
        has_torch, torch = _try_import_torch()
        if not has_torch or not candidates or self.network is None:
            return candidates

        self.network.eval()
        with torch.no_grad():
            c_tensor = torch.tensor(candidates, dtype=torch.long)
            probs = self.network.get_action_probs(c_tensor)
            sorted_idx = torch.argsort(probs, descending=True)
            return [candidates[i] for i in sorted_idx.tolist()]

    def get_action_probabilities(
        self, observation: dict[str, Any], candidates: list[int]
    ) -> dict[int, float]:
        """Return a softmax distribution over candidates from the stochastic actor.

        Args:
            observation: Context dictionary.
            candidates: List of candidate item IDs.

        Returns:
            Dict mapping each candidate ID to its selection probability.
        """
        has_torch, torch = _try_import_torch()
        if not has_torch or not candidates or self.network is None:
            return super().get_action_probabilities(observation, candidates)

        self.network.eval()
        with torch.no_grad():
            c_tensor = torch.tensor(candidates, dtype=torch.long)
            probs = self.network.get_action_probs(c_tensor).tolist()
            return {c: p for c, p in zip(candidates, probs)}

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """Run one gradient-descent step on a dummy SAC loss.

        Loss combines twin Q-value MSE with a policy entropy term.

        Args:
            batch: Experience dictionary (keys unused in current stub).

        Returns:
            Dict with ``"loss"`` key.
        """
        has_torch, torch = _try_import_torch()
        if not has_torch or self.network is None:
            return {"loss": 0.0}

        self.network.train()
        self.optimizer.zero_grad()
        c_tensor = torch.tensor([0, 1, 2], dtype=torch.long)
        q1, q2 = self.network.get_q_values(c_tensor)
        probs = self.network.get_action_probs(c_tensor)
        loss = (q1.mean() ** 2) + (q2.mean() ** 2) - (probs * torch.log(probs + 1e-8)).mean()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def save(self, path: str) -> None:
        """Serialize SAC network weights to *path*.

        Args:
            path: Destination file path (parent dirs created automatically).
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        has_torch, torch = _try_import_torch()
        if has_torch and self.network is not None:
            torch.save(self.network.state_dict(), p)
        else:
            p.write_text(json.dumps({"agent": "sac", "stub": True}))

    def load(self, path: str) -> None:
        """Load SAC network weights from *path*.

        Args:
            path: Source file path.
        """
        p = Path(path)
        has_torch, torch = _try_import_torch()
        if has_torch and self.network is not None and p.exists():
            try:
                self.network.load_state_dict(torch.load(p, weights_only=True))
            except Exception as exc:
                logger.warning("Agent load failed", error=str(exc))

    @property
    def name(self) -> str:
        """Registered agent name."""
        return "sac"
