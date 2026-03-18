"""Serving runtime — artifact loading, warm-up, and inference.

:class:`ServingRuntime` is the core serving component.  It loads a
canonical artifact via :func:`core.artifacts.load_artifact`, exposes
its metadata, and provides a :meth:`predict` method for inference.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from deeprl_recsys.core.logging import get_logger

logger = get_logger(__name__)


class ServingRuntime:
    """Manages the loaded artifact and agent for serving.

    Args:
        artifact_dir: Path to the canonical artifact directory (optional).
    """

    def __init__(self, artifact_dir: str | Path | None = None) -> None:
        self.artifact_dir: Path | None = Path(artifact_dir) if artifact_dir else None
        self.metadata: dict[str, Any] = {}
        self._loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        """Whether an artifact has been loaded."""
        return self._loaded

    def load(self, artifact_dir: str | Path | None = None) -> None:
        """Load an artifact and prepare for inference.

        Reads ``metadata.json`` from the artifact directory and populates
        :attr:`metadata`.  Computes a ``config_fingerprint`` from
        ``config.yaml`` for traceability.

        Args:
            artifact_dir: Override the artifact directory set at init.
        """
        if artifact_dir is not None:
            self.artifact_dir = Path(artifact_dir)

        if self.artifact_dir is None:
            logger.warning("runtime_load_skipped", reason="no artifact_dir")
            return

        adir = self.artifact_dir.resolve()
        logger.info("runtime_load_start", artifact_dir=str(adir))
        print(f"LOADING ARTIFACT: {adir.absolute()}")

        # Load metadata
        meta_path = adir / "metadata.json"
        if meta_path.exists():
            self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            self.metadata = {}
            logger.warning("metadata_missing", path=str(meta_path))

        # Config fingerprint and object
        config_path = adir / "config.yaml"
        self.config: dict[str, Any] = {}
        if config_path.exists():
            config_hash = hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest()[:12]
            self.metadata["config_fingerprint"] = config_hash
            import yaml
            with config_path.open("r", encoding="utf-8") as fh:
                self.config = yaml.safe_load(fh) or {}

        # Instantiate agent
        agent_name = self.metadata.get("agent_name", "")
        self.agent = None
        if agent_name:
            from deeprl_recsys.core.registry import create
            import copy
            agent_hp = self.config.get("agent", {}).get("hyperparams", {})
            try:
                self.agent = create("agents", agent_name, **copy.deepcopy(agent_hp))
                model_pt = adir / "model.pt"
                if model_pt.exists():
                    self.agent.load(str(model_pt))
                    logger.info("runtime_agent_loaded", model="model.pt")
            except Exception as e:
                logger.error("runtime_agent_init_failed", error=str(e))

        self._loaded = True
        logger.info(
            "runtime_load_done",
            agent_name=self.metadata.get("agent_name", ""),
            schema_version=self.metadata.get("schema_version", ""),
        )

    def predict(
        self,
        context: dict[str, Any],
        candidates: list[int],
        k: int = 10,
    ) -> list[dict[str, Any]]:
        """Run inference — rank candidates and return top-k.

        Args:
            context: Request context.
            candidates: Candidate item IDs.
            k: Number of items to return.

        Returns:
            List of ``{"item_id": int, "score": float}`` dicts.
        """
        if not getattr(self, "agent", None):
            return [{"item_id": c, "score": 0.0} for c in candidates[:k]]

        raw_scores = self.agent.get_action_probabilities(context, candidates)
        print(f"RAW PREDICT SCORES (ServingRuntime): {raw_scores}")
        
        # Ensure we have scores for all candidates
        scores = {c: float(raw_scores.get(c, 0.0)) for c in candidates}
        
        # Check if they are valid probabilities (sum to 1, range [0,1])
        vals = list(scores.values())
        total = sum(vals)
        all_in_range = all(0.0 <= v <= 1.0 for v in vals)
        
        import math
        is_prob = all_in_range and math.isclose(total, 1.0, rel_tol=1e-5)

        if not is_prob and total > 0:
            # If they don't sum to 1 but are positive, they might be logits or unnormalized scores
            # Apply Softmax for better distribution if not already probabilities
            import numpy as np
            exp_scores = np.exp(vals - np.max(vals)) # stable softmax
            probs = exp_scores / exp_scores.sum()
            scores = {c: float(p) for c, p in zip(candidates, probs)}
        elif not is_prob and total <= 0:
            # Fallback for all zero or negative scores
            prob = 1.0 / len(candidates) if candidates else 0.0
            scores = {c: prob for c in candidates}

        sorted_candidates = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)[:k]
        return [{"item_id": c, "score": scores[c]} for c in sorted_candidates]
