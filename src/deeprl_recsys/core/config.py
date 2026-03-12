"""Configuration system — YAML loading, deterministic merge, and Pydantic validation.

Implements the config inheritance contract:

    default.yaml  <  experiment.yaml  <  CLI overrides

All merges are deterministic deep-merges.  Relative paths in designated
fields are resolved to absolute paths relative to the experiment file's
directory.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

from deeprl_recsys.core.exceptions import ConfigError

# ── Path fields that should be resolved to absolute paths ────────
_PATH_FIELDS: set[str] = {"data_dir", "artifact_dir", "log_dir", "plot_dir", "path"}


# ── Pydantic sub-models ─────────────────────────────────────────
class PathsConfig(BaseModel):
    """Resolved filesystem paths."""

    data_dir: str = "data/"
    artifact_dir: str = "artifacts/"
    log_dir: str = "artifacts/logs/"
    plot_dir: str = "artifacts/plots/"


class LoggingConfig(BaseModel):
    """Logging settings."""

    level: str = "INFO"
    format: str = "json"


class SplitsConfig(BaseModel):
    """Dataset split ratios."""

    strategy: str = "temporal"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    path: str = ""
    format: str = "csv"
    schema_version: str = "bandit_v1"
    propensity_policy: Literal["mark_unreliable", "block_ope"] = "mark_unreliable"
    splits: SplitsConfig = Field(default_factory=SplitsConfig)


class AgentConfig(BaseModel):
    """Agent configuration."""

    name: str = "random"
    hyperparams: dict[str, Any] = Field(default_factory=dict)


class TrainingConfig(BaseModel):
    """Training loop configuration."""

    max_steps: int = 1000
    eval_interval: int = 100
    checkpoint_interval: int = 500
    callbacks: list[str] = Field(default_factory=list)


class OPEConfig(BaseModel):
    """Off-Policy Evaluation configuration."""

    estimator: str = "ips"
    clip_epsilon: float = 0.01
    fail_on_severity: Literal["error", "warning", "none"] = "error"


class ServingConfig(BaseModel):
    """Serving endpoint configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    auth_enabled: bool = False
    rate_limit_rpm: int = 60
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class PlatformConfig(BaseModel):
    """Top-level platform configuration (combines all sub-configs)."""

    seed: int = 42
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    ope: OPEConfig = Field(default_factory=OPEConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)


# ── Pure functions ──────────────────────────────────────────────


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed dictionary.

    Raises:
        ConfigError: If the file cannot be read or parsed.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {p}", source=str(p))
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML parse error: {exc}", source=str(p)) from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Expected a mapping at top level, got {type(data).__name__}", source=str(p))
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively deep-merge *override* into a copy of *base*.

    - Dict values are merged recursively.
    - Non-dict values in *override* replace *base*.
    - Keys in *base* not present in *override* are preserved.

    This is a **pure function** — neither input is mutated.

    Args:
        base: Base dictionary.
        override: Dictionary whose values take priority.

    Returns:
        A new merged dictionary.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_config(
    default_path: str | Path | None = None,
    experiment_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge configs with deterministic priority: default < experiment < CLI.

    Args:
        default_path: Path to ``default.yaml`` (or ``None`` to start empty).
        experiment_path: Path to experiment YAML (optional).
        overrides: CLI-level key-value overrides (optional).

    Returns:
        The fully-merged configuration dictionary.
    """
    cfg: dict[str, Any] = {}
    if default_path is not None:
        cfg = load_yaml(default_path)
    if experiment_path is not None:
        exp = load_yaml(experiment_path)
        cfg = deep_merge(cfg, exp)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def resolve_paths(cfg: dict[str, Any], base_dir: str | Path) -> dict[str, Any]:
    """Convert relative path values to absolute paths.

    Only fields whose keys are in ``_PATH_FIELDS`` and whose values are
    **relative** path strings are resolved.  Resolution is relative to
    *base_dir* (typically the experiment file's parent directory).

    Args:
        cfg: Configuration dictionary (may be nested).
        base_dir: Base directory for resolving relative paths.

    Returns:
        A **new** dictionary with paths resolved.
    """
    base = Path(base_dir).resolve()
    result = copy.deepcopy(cfg)

    def _resolve(d: dict[str, Any]) -> None:
        for key, value in d.items():
            if isinstance(value, dict):
                _resolve(value)
            elif isinstance(value, str) and key in _PATH_FIELDS:
                p = Path(value)
                if not p.is_absolute():
                    d[key] = str(base / p)

    _resolve(result)
    return result


def load_config(
    default_path: str | Path | None = None,
    experiment_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    resolve_paths_base: str | Path | None = None,
) -> PlatformConfig:
    """Full config pipeline: load → merge → resolve paths → validate.

    Args:
        default_path: Path to ``default.yaml``.
        experiment_path: Path to experiment YAML.
        overrides: CLI overrides.
        resolve_paths_base: Base directory for relative path resolution.
            Defaults to the experiment file's parent or CWD.

    Returns:
        A validated ``PlatformConfig`` instance.

    Raises:
        ConfigError: On any validation failure, with field and source context.
    """
    cfg = resolve_config(default_path, experiment_path, overrides)

    # Determine base dir for path resolution
    if resolve_paths_base is not None:
        base = Path(resolve_paths_base)
    elif experiment_path is not None:
        base = Path(experiment_path).resolve().parent
    elif default_path is not None:
        base = Path(default_path).resolve().parent
    else:
        base = Path.cwd()

    cfg = resolve_paths(cfg, base)

    # Validate with Pydantic
    try:
        return PlatformConfig(**cfg)
    except ValidationError as exc:
        # Enhance error with source context
        source = str(experiment_path or default_path or "overrides")
        errors = exc.errors()
        messages: list[str] = []
        for err in errors:
            loc = ".".join(str(x) for x in err["loc"])
            messages.append(f"  {loc}: {err['msg']}")
        detail = "\n".join(messages)
        raise ConfigError(
            f"Config validation failed:\n{detail}",
            source=source,
        ) from exc
