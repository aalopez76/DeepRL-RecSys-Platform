"""Unit tests for core/config.py — config inheritance, merge, path resolution, errors.

Tests the contract defined in implementación.txt Phase 1:
- test_inheritance_two_overrides_is_deterministic
- test_cli_overrides_take_precedence
- test_path_resolution_makes_absolute
- test_invalid_config_shows_useful_error
Plus additional coverage tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from deeprl_recsys.core.config import (
    PlatformConfig,
    deep_merge,
    load_config,
    load_yaml,
    resolve_config,
    resolve_paths,
)
from deeprl_recsys.core.exceptions import ConfigError


# ── Helpers ──────────────────────────────────────────


def _write_yaml(path: Path, data: dict) -> Path:
    """Write a dict as YAML and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh)
    return path


# ── Contract Tests (from implementación.txt) ─────────


@pytest.mark.unit
class TestConfigInheritance:
    """Tests for deterministic config inheritance and merging."""

    def test_inheritance_two_overrides_is_deterministic(self, tmp_path: Path) -> None:
        """default defines 5 fields, experiment overrides 2.

        The merged config must be exactly the expected dict, every time.
        """
        default_data = {
            "seed": 42,
            "agent": {"name": "random", "hyperparams": {}},
            "dataset": {"schema_version": "bandit_v1", "path": "data/default/"},
            "training": {"max_steps": 1000},
            "ope": {"estimator": "ips"},
        }
        exp_data = {
            "seed": 123,
            "agent": {"name": "dqn", "hyperparams": {"learning_rate": 0.001}},
        }
        expected = {
            "seed": 123,
            "agent": {"name": "dqn", "hyperparams": {"learning_rate": 0.001}},
            "dataset": {"schema_version": "bandit_v1", "path": "data/default/"},
            "training": {"max_steps": 1000},
            "ope": {"estimator": "ips"},
        }

        default_path = _write_yaml(tmp_path / "default.yaml", default_data)
        exp_path = _write_yaml(tmp_path / "exp.yaml", exp_data)

        # Run multiple times to verify determinism
        for _ in range(5):
            merged = resolve_config(default_path, exp_path)
            assert merged == expected, f"Non-deterministic merge: {merged}"

    def test_cli_overrides_take_precedence(self, tmp_path: Path) -> None:
        """CLI overrides must always win, regardless of merge order."""
        default_data = {"seed": 42, "agent": {"name": "random"}}
        exp_data = {"agent": {"name": "dqn"}}
        cli_overrides = {"agent": {"name": "ppo"}, "seed": 999}

        default_path = _write_yaml(tmp_path / "default.yaml", default_data)
        exp_path = _write_yaml(tmp_path / "exp.yaml", exp_data)

        merged = resolve_config(default_path, exp_path, cli_overrides)
        assert merged["agent"]["name"] == "ppo"
        assert merged["seed"] == 999

    def test_deep_merge_preserves_nested_keys(self) -> None:
        """Keys in base not overridden in override must survive."""
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 10}}
        result = deep_merge(base, override)
        assert result == {"a": {"b": 10, "c": 2}, "d": 3}

    def test_deep_merge_does_not_mutate_inputs(self) -> None:
        """deep_merge must be pure — no side effects on inputs."""
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        base_copy = {"a": {"b": 1}}
        override_copy = {"a": {"c": 2}}

        deep_merge(base, override)

        assert base == base_copy
        assert override == override_copy


@pytest.mark.unit
class TestPathResolution:
    """Tests for path resolution to absolute paths."""

    def test_path_resolution_makes_absolute(self, tmp_path: Path) -> None:
        """Relative paths must become absolute, resolved against base_dir."""
        cfg = {
            "paths": {
                "data_dir": "data/ml100k",
                "artifact_dir": "artifacts/exp1",
            },
            "dataset": {
                "path": "data/train.csv",
            },
        }
        resolved = resolve_paths(cfg, base_dir=tmp_path)

        # All path fields must be absolute
        assert Path(resolved["paths"]["data_dir"]).is_absolute()
        assert Path(resolved["paths"]["artifact_dir"]).is_absolute()
        assert Path(resolved["dataset"]["path"]).is_absolute()

        # Must be resolved relative to base_dir
        assert resolved["paths"]["data_dir"] == str(tmp_path / "data" / "ml100k")

    def test_absolute_paths_unchanged(self, tmp_path: Path) -> None:
        """Already-absolute paths must not be modified."""
        abs_path = str(tmp_path / "already" / "absolute")
        cfg = {"paths": {"data_dir": abs_path}}
        resolved = resolve_paths(cfg, base_dir=tmp_path)
        assert resolved["paths"]["data_dir"] == abs_path

    def test_non_path_fields_unchanged(self, tmp_path: Path) -> None:
        """Fields not in _PATH_FIELDS must not be touched."""
        cfg = {"agent": {"name": "relative/path/like/string"}}
        resolved = resolve_paths(cfg, base_dir=tmp_path)
        assert resolved["agent"]["name"] == "relative/path/like/string"


@pytest.mark.unit
class TestLoadYAML:
    """Tests for YAML loading edge cases."""

    def test_load_valid_yaml(self, tmp_path: Path) -> None:
        """A valid YAML file loads correctly."""
        data = {"key": "value", "nested": {"a": 1}}
        path = _write_yaml(tmp_path / "test.yaml", data)
        loaded = load_yaml(path)
        assert loaded == data

    def test_load_missing_file_raises_config_error(self, tmp_path: Path) -> None:
        """Missing file must raise ConfigError."""
        with pytest.raises(ConfigError, match="not found"):
            load_yaml(tmp_path / "nonexistent.yaml")

    def test_load_empty_yaml_returns_empty_dict(self, tmp_path: Path) -> None:
        """An empty YAML file should return {}."""
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        assert load_yaml(path) == {}

    def test_load_non_dict_yaml_raises(self, tmp_path: Path) -> None:
        """A YAML file with a list at top level should raise ConfigError."""
        path = tmp_path / "list.yaml"
        path.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="Expected a mapping"):
            load_yaml(path)


@pytest.mark.unit
class TestConfigValidation:
    """Tests for Pydantic validation and error messages."""

    def test_invalid_config_shows_useful_error(self, tmp_path: Path) -> None:
        """Pydantic validation errors must include field names and source."""
        default_data = {
            "seed": "not_a_number",  # seed must be int
        }
        default_path = _write_yaml(tmp_path / "bad.yaml", default_data)

        with pytest.raises(ConfigError) as exc_info:
            load_config(default_path)

        error_msg = str(exc_info.value)
        assert "seed" in error_msg
        assert "bad.yaml" in error_msg

    def test_valid_full_config_produces_platform_config(self, tmp_path: Path) -> None:
        """A valid config pipeline produces a PlatformConfig instance."""
        default_data = {
            "seed": 42,
            "paths": {"data_dir": "data/"},
            "agent": {"name": "random"},
        }
        default_path = _write_yaml(tmp_path / "default.yaml", default_data)
        config = load_config(default_path)

        assert isinstance(config, PlatformConfig)
        assert config.seed == 42
        assert config.agent.name == "random"

    def test_defaults_applied_when_keys_missing(self) -> None:
        """PlatformConfig must fill in defaults for missing keys."""
        config = PlatformConfig()
        assert config.seed == 42
        assert config.ope.estimator == "ips"
        assert config.serving.port == 8000

    def test_full_pipeline_with_experiment_overrides(self, tmp_path: Path) -> None:
        """Full load_config pipeline with default + experiment."""
        default_data = {
            "seed": 1,
            "agent": {"name": "random"},
            "training": {"max_steps": 100},
        }
        exp_data = {
            "seed": 42,
            "agent": {"name": "dqn", "hyperparams": {"lr": 0.001}},
        }
        default_path = _write_yaml(tmp_path / "default.yaml", default_data)
        exp_path = _write_yaml(tmp_path / "exp.yaml", exp_data)

        config = load_config(default_path, exp_path)
        assert config.seed == 42
        assert config.agent.name == "dqn"
        assert config.agent.hyperparams == {"lr": 0.001}
        assert config.training.max_steps == 100  # preserved from default


@pytest.mark.unit
class TestResolveConfig:
    """Tests for the resolve_config function edge cases."""

    def test_resolve_with_only_defaults(self, tmp_path: Path) -> None:
        """resolve_config with only a default file should work."""
        data = {"seed": 7}
        path = _write_yaml(tmp_path / "default.yaml", data)
        cfg = resolve_config(default_path=path)
        assert cfg["seed"] == 7

    def test_resolve_with_only_overrides(self) -> None:
        """resolve_config with only CLI overrides should work."""
        cfg = resolve_config(overrides={"seed": 99})
        assert cfg["seed"] == 99

    def test_resolve_with_no_inputs(self) -> None:
        """resolve_config with no arguments returns empty dict."""
        cfg = resolve_config()
        assert cfg == {}


@pytest.mark.unit
class TestLoadConfigWithRealFiles:
    """Integration-style tests using the real project config files."""

    CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parents[3] / "configs" / "default.yaml").exists(),
        reason="Project configs not available",
    )
    def test_load_project_default_yaml(self) -> None:
        """The project's own default.yaml must parse into PlatformConfig."""
        config = load_config(self.CONFIGS_DIR / "default.yaml")
        assert isinstance(config, PlatformConfig)
        assert config.seed == 42

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parents[3] / "configs" / "experiments" / "exp1_dqn_movielens.yaml").exists(),
        reason="Project configs not available",
    )
    def test_load_project_experiment_yaml(self) -> None:
        """default.yaml + exp1 must merge correctly."""
        config = load_config(
            self.CONFIGS_DIR / "default.yaml",
            self.CONFIGS_DIR / "experiments" / "exp1_dqn_movielens.yaml",
        )
        assert config.agent.name == "dqn"
        assert config.seed == 42  # exp1 has seed: 42
