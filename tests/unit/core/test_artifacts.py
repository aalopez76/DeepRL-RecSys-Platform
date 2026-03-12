"""Unit tests for core/artifacts.py — save/load, checksums, schema guards.

Required tests from implementación.txt Phase 3:
- test_load_tiny_artifact_reads_metadata()
- test_schema_version_guard_mismatch_raises()
- test_checksum_mismatch_raises()
- test_missing_required_file_raises()
Plus additional round-trip and edge-case coverage.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from deeprl_recsys.core.artifacts import (
    ArtifactMetadata,
    LoadedArtifact,
    compute_sha256,
    load_artifact,
    save_artifact,
)
from deeprl_recsys.core.exceptions import ArtifactError

# ── Path to the committed fixture ────────────────────
TINY_ARTIFACT = Path(__file__).resolve().parents[2] / "fixtures" / "tiny_artifact"


# ── Contract Tests (from implementación.txt) ─────────


@pytest.mark.unit
class TestLoadTinyArtifact:
    """Tests that use the committed tiny_artifact fixture."""

    @pytest.mark.skipif(
        not TINY_ARTIFACT.exists(),
        reason="tiny_artifact fixture not found",
    )
    def test_load_tiny_artifact_reads_metadata(self) -> None:
        """Loading tiny_artifact must correctly parse metadata fields."""
        loaded = load_artifact(TINY_ARTIFACT)

        assert isinstance(loaded, LoadedArtifact)
        assert isinstance(loaded.metadata, ArtifactMetadata)
        assert loaded.metadata.artifact_version == "1.0.0"
        assert loaded.metadata.schema_version == "bandit_v1"
        assert loaded.metadata.agent_name == "random"
        assert loaded.metadata.model_version == "0.1.0-tiny"
        assert loaded.metadata.git_sha == "abc1234"
        assert "model.pt" in loaded.metadata.checksums
        assert "config.yaml" in loaded.metadata.checksums

    @pytest.mark.skipif(
        not TINY_ARTIFACT.exists(),
        reason="tiny_artifact fixture not found",
    )
    def test_load_tiny_artifact_reads_config(self) -> None:
        """Loading tiny_artifact must parse config.yaml."""
        loaded = load_artifact(TINY_ARTIFACT)
        assert loaded.config["seed"] == 42
        assert loaded.config["agent"]["name"] == "random"


@pytest.mark.unit
class TestSchemaGuard:
    """Tests for schema_version.txt guard validation."""

    def test_schema_version_guard_mismatch_raises(self, tmp_path: Path) -> None:
        """Schema guard mismatch must raise ArtifactError."""
        # Set up artifact with mismatched schema
        _create_minimal_artifact(tmp_path, schema_version="bandit_v1")
        # Tamper guard to different version
        (tmp_path / "schema_version.txt").write_text("sequential_v1\n")

        with pytest.raises(ArtifactError, match="Schema version mismatch"):
            load_artifact(tmp_path)

    def test_schema_version_guard_matches_passes(self, tmp_path: Path) -> None:
        """Matching schema guard must not raise."""
        _create_minimal_artifact(tmp_path, schema_version="bandit_v1")
        loaded = load_artifact(tmp_path)
        assert loaded.metadata.schema_version == "bandit_v1"

    def test_missing_schema_guard_is_ok(self, tmp_path: Path) -> None:
        """Missing schema_version.txt should not raise (it's optional)."""
        _create_minimal_artifact(tmp_path, schema_version="bandit_v1")
        (tmp_path / "schema_version.txt").unlink()
        # Should load without error
        loaded = load_artifact(tmp_path)
        assert loaded.metadata.schema_version == "bandit_v1"


@pytest.mark.unit
class TestChecksumValidation:
    """Tests for checksum integrity validation."""

    def test_checksum_mismatch_raises(self, tmp_path: Path) -> None:
        """Tampered file must trigger ArtifactError with 'Checksum mismatch'."""
        _create_minimal_artifact(tmp_path, schema_version="bandit_v1")
        # Tamper model file
        (tmp_path / "model.pt").write_bytes(b"corrupted data!")
        with pytest.raises(ArtifactError, match="Checksum mismatch.*model\\.pt"):
            load_artifact(tmp_path)

    def test_checksum_file_missing_raises(self, tmp_path: Path) -> None:
        """File declared in checksums but physically missing must raise."""
        _create_minimal_artifact(tmp_path, schema_version="bandit_v1")
        # Remove model file but keep it in metadata checksums
        (tmp_path / "model.pt").unlink()
        with pytest.raises(ArtifactError, match="model\\.pt.*missing"):
            load_artifact(tmp_path)

    def test_no_checksums_is_valid(self, tmp_path: Path) -> None:
        """An artifact without checksums in metadata should still load."""
        _create_minimal_artifact(tmp_path, schema_version="bandit_v1", include_checksums=False)
        loaded = load_artifact(tmp_path)
        assert loaded.metadata.checksums == {}


@pytest.mark.unit
class TestMissingFiles:
    """Tests for required file presence."""

    def test_missing_required_file_raises(self, tmp_path: Path) -> None:
        """Missing metadata.json must raise ArtifactError."""
        _create_minimal_artifact(tmp_path, schema_version="bandit_v1")
        (tmp_path / "metadata.json").unlink()
        with pytest.raises(ArtifactError, match="metadata\\.json"):
            load_artifact(tmp_path)

    def test_missing_config_yaml_raises(self, tmp_path: Path) -> None:
        """Missing config.yaml must raise ArtifactError."""
        _create_minimal_artifact(tmp_path, schema_version="bandit_v1")
        (tmp_path / "config.yaml").unlink()
        with pytest.raises(ArtifactError, match="config\\.yaml"):
            load_artifact(tmp_path)


@pytest.mark.unit
class TestSaveArtifact:
    """Tests for save_artifact()."""

    def test_save_creates_all_required_files(self, tmp_path: Path) -> None:
        """save_artifact must create all canonical files."""
        out = save_artifact(
            output_dir=tmp_path / "my_artifact",
            model_bytes=b"fake model bytes",
            config_dict={"seed": 42, "agent": {"name": "dqn"}},
            schema_version="bandit_v1",
            agent_name="dqn",
        )
        assert (out / "metadata.json").exists()
        assert (out / "config.yaml").exists()
        assert (out / "schema_version.txt").exists()
        assert (out / "checksums.txt").exists()
        assert (out / "model.pt").exists()

    def test_save_metadata_fields(self, tmp_path: Path) -> None:
        """Metadata must contain the expected fields."""
        out = save_artifact(
            output_dir=tmp_path / "art",
            model_bytes=b"data",
            config_dict={"x": 1},
            schema_version="sequential_v1",
            agent_name="ppo",
            model_version="1.2.3",
        )
        meta = json.loads((out / "metadata.json").read_text())
        assert meta["schema_version"] == "sequential_v1"
        assert meta["agent_name"] == "ppo"
        assert meta["model_version"] == "1.2.3"
        assert "model.pt" in meta["checksums"]

    def test_schema_guard_matches_metadata(self, tmp_path: Path) -> None:
        """schema_version.txt must match metadata.schema_version."""
        out = save_artifact(
            output_dir=tmp_path / "art",
            model_bytes=b"",
            config_dict={},
            schema_version="bandit_v1",
        )
        guard = (out / "schema_version.txt").read_text().strip()
        meta = json.loads((out / "metadata.json").read_text())
        assert guard == meta["schema_version"]


@pytest.mark.unit
class TestRoundTrip:
    """Tests that save → load preserves all data."""

    def test_save_then_load_roundtrip(self, tmp_path: Path) -> None:
        """An artifact saved and immediately loaded must be identical."""
        config = {"seed": 42, "agent": {"name": "sac"}, "training": {"max_steps": 100}}
        model_data = b"binary model content"
        out = save_artifact(
            output_dir=tmp_path / "roundtrip",
            model_bytes=model_data,
            config_dict=config,
            schema_version="bandit_v1",
            agent_name="sac",
            model_version="0.5.0",
        )
        loaded = load_artifact(out)

        assert loaded.metadata.schema_version == "bandit_v1"
        assert loaded.metadata.agent_name == "sac"
        assert loaded.metadata.model_version == "0.5.0"
        assert loaded.config == config
        assert (loaded.artifact_dir / "model.pt").read_bytes() == model_data


@pytest.mark.unit
class TestComputeSha256:
    """Tests for the compute_sha256() helper."""

    def test_empty_file_hash(self, tmp_path: Path) -> None:
        """SHA-256 of an empty file is the well-known hash."""
        f = tmp_path / "empty"
        f.write_bytes(b"")
        result = compute_sha256(f)
        assert result == "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_known_content_hash(self, tmp_path: Path) -> None:
        """SHA-256 of known content must match precomputed hash."""
        f = tmp_path / "hello"
        f.write_bytes(b"hello")
        result = compute_sha256(f)
        import hashlib
        expected = "sha256:" + hashlib.sha256(b"hello").hexdigest()
        assert result == expected


@pytest.mark.unit
class TestExpectedSchemaVersion:
    """Tests for the expected_schema_version parameter."""

    def test_expected_schema_version_matches_passes(self, tmp_path: Path) -> None:
        """load_artifact with matching expected_schema_version must pass."""
        _create_minimal_artifact(tmp_path, schema_version="bandit_v1")
        loaded = load_artifact(tmp_path, expected_schema_version="bandit_v1")
        assert loaded.metadata.schema_version == "bandit_v1"

    def test_expected_schema_version_mismatch_raises(self, tmp_path: Path) -> None:
        """load_artifact with mismatched expected_schema_version must raise."""
        _create_minimal_artifact(tmp_path, schema_version="bandit_v1")
        with pytest.raises(ArtifactError, match="Expected schema version"):
            load_artifact(tmp_path, expected_schema_version="sequential_v1")

    def test_expected_schema_version_none_skips_check(self, tmp_path: Path) -> None:
        """load_artifact with expected_schema_version=None must not check."""
        _create_minimal_artifact(tmp_path, schema_version="bandit_v1")
        loaded = load_artifact(tmp_path, expected_schema_version=None)
        assert loaded.metadata.schema_version == "bandit_v1"


# ── Helper ──────────────────────────────────────────


def _create_minimal_artifact(
    adir: Path,
    schema_version: str = "bandit_v1",
    include_checksums: bool = True,
) -> None:
    """Create a minimal valid artifact directory for testing."""
    adir.mkdir(parents=True, exist_ok=True)

    # model
    model = adir / "model.pt"
    model.write_bytes(b"test model")

    # config
    config = adir / "config.yaml"
    with config.open("w") as fh:
        yaml.safe_dump({"seed": 1}, fh)

    # schema guard
    guard = adir / "schema_version.txt"
    guard.write_text(schema_version + "\n")

    # checksums
    checksums: dict[str, str] = {}
    if include_checksums:
        checksums = {
            "model.pt": compute_sha256(model),
            "config.yaml": compute_sha256(config),
        }

    # metadata
    meta = {
        "artifact_version": "1.0.0",
        "schema_version": schema_version,
        "agent_name": "test",
        "model_version": "0.0.1",
        "git_sha": "test123",
        "created_at": "2026-01-01T00:00:00Z",
        "checksums": checksums,
    }
    with (adir / "metadata.json").open("w") as fh:
        json.dump(meta, fh, indent=4)
