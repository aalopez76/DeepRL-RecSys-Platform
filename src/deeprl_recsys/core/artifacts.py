"""Canonical artifact save/load with metadata, checksums, and schema guards.

An *artifact* is a directory containing everything needed to reproduce or
serve a trained model:

.. code-block:: text

    my_artifact/
        metadata.json       # ArtifactMetadata (Pydantic)
        config.yaml         # Training configuration snapshot
        schema_version.txt  # Guard — must match metadata.schema_version
        checksums.txt       # "sha256:<hex>  <filename>" per line
        model.*             # Serialised model file(s)

Use :func:`save_artifact` to create an artifact and
:func:`load_artifact` to load + validate one.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from deeprl_recsys.core.exceptions import ArtifactError

# ── Required files inside an artifact ────────────────────────────
REQUIRED_FILES: tuple[str, ...] = ("metadata.json", "config.yaml")


# ── Pydantic model ──────────────────────────────────────────────


class ArtifactMetadata(BaseModel):
    """Metadata stored alongside a model artifact.

    Attributes:
        artifact_version: Version of the artifact format.
        schema_version: Dataset schema version used during training.
        agent_name: Name of the agent that produced this artifact.
        model_version: Semantic version of the model.
        git_sha: Git commit SHA at artifact creation time.
        created_at: ISO-8601 timestamp of creation.
        checksums: Mapping of ``filename → "sha256:<hex>"`` for integrity.
    """

    artifact_version: str = "1.0.0"
    schema_version: str = ""
    agent_name: str = ""
    model_version: str = ""
    git_sha: str = ""
    created_at: str = ""
    checksums: dict[str, str] = Field(default_factory=dict)


# ── Checksum helpers ────────────────────────────────────────────


def compute_sha256(path: Path) -> str:
    """Compute ``sha256:<hex>`` for the given file.

    Args:
        path: Path to the file.

    Returns:
        String in the format ``"sha256:<hex>"``.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _get_git_sha() -> str:
    """Try to obtain the current git HEAD SHA.  Returns ``""`` on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


# ── Save ────────────────────────────────────────────────────────


def save_artifact(
    output_dir: str | Path,
    model_bytes: bytes,
    config_dict: dict[str, Any],
    schema_version: str,
    agent_name: str = "",
    model_filename: str = "model.pt",
    **extra_metadata: Any,
) -> Path:
    """Save a model artifact with metadata and checksums.

    Creates the canonical artifact layout under *output_dir*.

    Args:
        output_dir: Destination directory (created if absent).
        model_bytes: Raw bytes for the model file.
        config_dict: Training configuration dictionary.
        schema_version: Dataset schema version used during training.
        agent_name: Agent that produced the artifact.
        model_filename: Filename for the model blob (default ``"model.pt"``).
        **extra_metadata: Additional fields for :class:`ArtifactMetadata`.

    Returns:
        Path to the saved artifact directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Write model file
    model_path = out / model_filename
    model_path.write_bytes(model_bytes)

    # 2. Write config.yaml
    config_path = out / "config.yaml"
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config_dict, fh, default_flow_style=False)

    # 3. Write schema_version.txt (guard)
    schema_guard = out / "schema_version.txt"
    schema_guard.write_text(schema_version + "\n", encoding="utf-8")

    # 4. Compute checksums
    checksums: dict[str, str] = {}
    for file in out.iterdir():
        if file.is_file() and file.name not in ("metadata.json", "checksums.txt"):
            checksums[file.name] = compute_sha256(file)

    # 5. Write checksums.txt
    checksums_path = out / "checksums.txt"
    lines = [f"{chk}  {fname}" for fname, chk in sorted(checksums.items())]
    checksums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 6. Build and write metadata.json
    metadata = ArtifactMetadata(
        schema_version=schema_version,
        agent_name=agent_name,
        git_sha=extra_metadata.pop("git_sha", _get_git_sha()),
        created_at=extra_metadata.pop(
            "created_at", datetime.now(timezone.utc).isoformat()
        ),
        model_version=extra_metadata.pop("model_version", ""),
        artifact_version=extra_metadata.pop("artifact_version", "1.0.0"),
        checksums=checksums,
    )
    meta_path = out / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata.model_dump(), fh, indent=4)
        fh.write("\n")

    return out


# ── Load ────────────────────────────────────────────────────────


class LoadedArtifact:
    """Container for a loaded and validated artifact.

    Attributes:
        metadata: Parsed :class:`ArtifactMetadata`.
        config: Training configuration dict.
        artifact_dir: Absolute path to the artifact directory.
    """

    def __init__(
        self,
        metadata: ArtifactMetadata,
        config: dict[str, Any],
        artifact_dir: Path,
    ) -> None:
        self.metadata = metadata
        self.config = config
        self.artifact_dir = artifact_dir


def load_artifact(
    artifact_dir: str | Path,
    expected_schema_version: str | None = None,
) -> LoadedArtifact:
    """Load and validate a model artifact.

    Validation steps:

    1. Presence of required files (``metadata.json``, ``config.yaml``).
    2. **Schema guard** — if ``schema_version.txt`` exists its content must
       match ``metadata.schema_version``.
    3. **Checksum validation** — if ``checksums`` are declared in metadata,
       each listed file is verified.
    4. **Expected schema** — if *expected_schema_version* is given, it must
       match ``metadata.schema_version``.

    Args:
        artifact_dir: Path to the artifact directory.
        expected_schema_version: If set, validate that the artifact's
            schema version matches this value.

    Returns:
        A :class:`LoadedArtifact` with parsed metadata and config.

    Raises:
        ArtifactError: On any validation failure.
    """
    adir = Path(artifact_dir).resolve()

    # ── 1. Required file presence ─────────────────────────────
    _check_required_files(adir)

    # ── 2. Parse metadata ─────────────────────────────────────
    meta_path = adir / "metadata.json"
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata = ArtifactMetadata(**raw)
    except Exception as exc:
        raise ArtifactError(
            f"Cannot parse metadata.json in {adir}: {exc}"
        ) from exc

    # ── 3. Parse config ───────────────────────────────────────
    config_path = adir / "config.yaml"
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
    except Exception as exc:
        raise ArtifactError(
            f"Cannot parse config.yaml in {adir}: {exc}"
        ) from exc

    # ── 4. Schema version guard ───────────────────────────────
    _check_schema_guard(adir, metadata)

    # ── 5. Expected schema version ────────────────────────────
    if expected_schema_version is not None:
        if metadata.schema_version != expected_schema_version:
            raise ArtifactError(
                f"Expected schema version {expected_schema_version!r} "
                f"but artifact has {metadata.schema_version!r}"
            )

    # ── 6. Checksum validation ────────────────────────────────
    _check_checksums(adir, metadata)

    return LoadedArtifact(metadata=metadata, config=config, artifact_dir=adir)


# ── Internal validators ─────────────────────────────────────────


def _check_required_files(adir: Path) -> None:
    """Raise ``ArtifactError`` if any required files are missing."""
    for fname in REQUIRED_FILES:
        if not (adir / fname).exists():
            raise ArtifactError(
                f"Missing required file '{fname}' in artifact: {adir}"
            )


def _check_schema_guard(adir: Path, metadata: ArtifactMetadata) -> None:
    """Validate schema_version.txt matches metadata, if present."""
    guard = adir / "schema_version.txt"
    if not guard.exists():
        return
    guard_version = guard.read_text(encoding="utf-8").strip()
    if guard_version != metadata.schema_version:
        raise ArtifactError(
            f"Schema version mismatch: schema_version.txt says "
            f"{guard_version!r} but metadata says "
            f"{metadata.schema_version!r}"
        )


def _check_checksums(adir: Path, metadata: ArtifactMetadata) -> None:
    """Validate checksums declared in metadata."""
    if not metadata.checksums:
        return
    for fname, expected_hash in metadata.checksums.items():
        file_path = adir / fname
        if not file_path.exists():
            raise ArtifactError(
                f"Checksum declared for '{fname}' but file is missing in {adir}"
            )
        actual_hash = compute_sha256(file_path)
        if actual_hash != expected_hash:
            raise ArtifactError(
                f"Checksum mismatch for '{fname}': "
                f"expected {expected_hash}, got {actual_hash}"
            )
