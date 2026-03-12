"""Unit tests for serving/runtime.py — coverage hardening.

Targets uncovered lines: is_loaded property, load(None) warning,
and load() with missing metadata.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from deeprl_recsys.serving.runtime import ServingRuntime


@pytest.mark.unit
class TestServingRuntimeInit:
    """Tests for ServingRuntime initialization and is_loaded property."""

    def test_is_loaded_false_before_load(self) -> None:
        """ServingRuntime.is_loaded must be False before any load()."""
        rt = ServingRuntime()
        assert rt.is_loaded is False

    def test_is_loaded_true_after_load(self, tmp_path: Path) -> None:
        """ServingRuntime.is_loaded must be True after successful load()."""
        _create_minimal_runtime_artifact(tmp_path)
        rt = ServingRuntime()
        rt.load(tmp_path)
        assert rt.is_loaded is True

    def test_init_with_artifact_dir(self, tmp_path: Path) -> None:
        """ServingRuntime accepts artifact_dir at init."""
        rt = ServingRuntime(artifact_dir=tmp_path)
        assert rt.artifact_dir == tmp_path


@pytest.mark.unit
class TestServingRuntimeLoad:
    """Tests for load() edge cases."""

    def test_load_none_without_artifact_dir_logs_warning(self) -> None:
        """load(None) with no artifact_dir should return without error."""
        rt = ServingRuntime()
        rt.load(None)
        # Should NOT set _loaded since there's no artifact_dir
        assert rt.is_loaded is False
        assert rt.metadata == {}

    def test_load_no_args_without_artifact_dir(self) -> None:
        """load() with no argument and no artifact_dir should log warning."""
        rt = ServingRuntime()
        rt.load()
        assert rt.is_loaded is False

    def test_load_missing_metadata_json(self, tmp_path: Path) -> None:
        """load() with directory lacking metadata.json should still load."""
        # Create a config.yaml but no metadata.json
        (tmp_path / "config.yaml").write_text("seed: 42\n", encoding="utf-8")
        rt = ServingRuntime()
        rt.load(tmp_path)
        assert rt.is_loaded is True
        assert rt.metadata.get("config_fingerprint") is not None
        # metadata should be empty dict (no agent_name etc.)
        assert "agent_name" not in rt.metadata

    def test_load_missing_config_yaml(self, tmp_path: Path) -> None:
        """load() with directory lacking config.yaml should still load."""
        meta = {"agent_name": "test", "schema_version": "bandit_v1"}
        (tmp_path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        rt = ServingRuntime()
        rt.load(tmp_path)
        assert rt.is_loaded is True
        assert rt.metadata["agent_name"] == "test"
        # No config_fingerprint since config.yaml doesn't exist
        assert "config_fingerprint" not in rt.metadata

    def test_load_full_artifact(self, tmp_path: Path) -> None:
        """load() with valid artifact populates metadata and fingerprint."""
        _create_minimal_runtime_artifact(tmp_path)
        rt = ServingRuntime()
        rt.load(tmp_path)
        assert rt.is_loaded is True
        assert rt.metadata["agent_name"] == "random"
        assert "config_fingerprint" in rt.metadata

    def test_load_override_artifact_dir(self, tmp_path: Path) -> None:
        """load(new_dir) overrides the init artifact_dir."""
        rt = ServingRuntime(artifact_dir=Path("/nonexistent"))
        _create_minimal_runtime_artifact(tmp_path)
        rt.load(tmp_path)
        assert rt.artifact_dir == tmp_path
        assert rt.is_loaded is True


@pytest.mark.unit
class TestServingRuntimePredict:
    """Tests for predict() fallback behavior."""

    def test_predict_returns_k_items(self) -> None:
        """predict() should return k items from candidates."""
        rt = ServingRuntime()
        result = rt.predict({}, [1, 2, 3, 4, 5], k=3)
        assert len(result) == 3
        assert all(r["score"] == 0.0 for r in result)

    def test_predict_fewer_than_k(self) -> None:
        """predict() with fewer candidates than k returns all."""
        rt = ServingRuntime()
        result = rt.predict({}, [1, 2], k=5)
        assert len(result) == 2


# ── Helper ──────────────────────────────────────────

def _create_minimal_runtime_artifact(adir: Path) -> None:
    """Create minimal artifact for runtime load testing."""
    adir.mkdir(parents=True, exist_ok=True)
    meta = {
        "agent_name": "random",
        "schema_version": "bandit_v1",
        "model_version": "0.0.1",
    }
    (adir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    with (adir / "config.yaml").open("w") as fh:
        yaml.safe_dump({"seed": 42}, fh)
