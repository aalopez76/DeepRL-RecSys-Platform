"""Baseline agents — Random, Greedy, Top-K.

These are fully functional and serve as benchmarks for RL agents.
They can be exported and served via the serving layer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from deeprl_recsys.agents.base import BaseAgent


class RandomAgent(BaseAgent):
    """Selects actions uniformly at random from candidates."""

    def __init__(self, seed: int = 42, **kwargs: Any) -> None:
        self._rng = np.random.default_rng(seed)
        self._seed = seed

    def act(self, observation: dict[str, Any], candidates: list[int]) -> list[int]:
        """Return candidates in a random permutation."""
        indices = self._rng.permutation(len(candidates))
        return [candidates[i] for i in indices]

    def save(self, path: str) -> None:
        """Save seed to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"agent": "random", "seed": self._seed}))

    def load(self, path: str) -> None:
        """Load seed from a JSON file."""
        data = json.loads(Path(path).read_text())
        self._seed = data.get("seed", 42)
        self._rng = np.random.default_rng(self._seed)

    @property
    def name(self) -> str:
        return "random"


class GreedyAgent(BaseAgent):
    """Selects top items by score (or by index order if no scores)."""

    def __init__(self, **kwargs: Any) -> None:
        self._scores: dict[int, float] = {}

    def act(self, observation: dict[str, Any], candidates: list[int]) -> list[int]:
        """Return candidates sorted by precomputed scores (desc)."""
        return sorted(candidates, key=lambda x: self._scores.get(x, 0.0), reverse=True)

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"agent": "greedy", "scores": self._scores}))

    def load(self, path: str) -> None:
        data = json.loads(Path(path).read_text())
        raw = data.get("scores", {})
        self._scores = {int(k): float(v) for k, v in raw.items()}

    @property
    def name(self) -> str:
        return "greedy"


class TopKAgent(BaseAgent):
    """Returns the first K items from candidates (popularity baseline)."""

    def __init__(self, k: int = 10, **kwargs: Any) -> None:
        self._k = k

    def act(self, observation: dict[str, Any], candidates: list[int]) -> list[int]:
        """Return the first *k* candidates."""
        return candidates[: self._k]

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"agent": "topk", "k": self._k}))

    def load(self, path: str) -> None:
        data = json.loads(Path(path).read_text())
        self._k = data.get("k", 10)

    @property
    def name(self) -> str:
        return "topk"
