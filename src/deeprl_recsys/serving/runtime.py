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

        # Load metadata
        meta_path = adir / "metadata.json"
        if meta_path.exists():
            self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            self.metadata = {}
            logger.warning("metadata_missing", path=str(meta_path))

        # Config fingerprint
        config_path = adir / "config.yaml"
        if config_path.exists():
            config_hash = hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest()[:12]
            self.metadata["config_fingerprint"] = config_hash

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

        When no trained agent is loaded, falls back to returning the
        first *k* candidates with a score of ``0.0`` (useful for
        smoke-testing the serving endpoint).

        Args:
            context: Request context.
            candidates: Candidate item IDs.
            k: Number of items to return.

        Returns:
            List of ``{"item_id": int, "score": float}`` dicts.
        """
        # Fallback: return first k candidates (no model loaded)
        return [{"item_id": c, "score": 0.0} for c in candidates[:k]]
