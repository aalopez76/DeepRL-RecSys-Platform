"""Pipeline: export — package a canonical artifact.

Orchestrates:
1. Read training checkpoint + config
2. Call :func:`core.artifacts.save_artifact`
3. Write artifact with metadata, checksums, schema guard
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deeprl_recsys.core.artifacts import save_artifact
from deeprl_recsys.core.logging import get_logger

logger = get_logger(__name__)


def run_export(config: dict[str, Any], *, dry_run: bool = False) -> str:
    """Execute the export pipeline.

    Args:
        config: Resolved platform config.  Expected keys:

            - ``training.checkpoint_dir`` — path to training checkpoint
            - ``export.output_dir`` — destination for the artifact
            - ``dataset.schema_version`` — schema version
            - ``agent.name`` — agent name for metadata
            - ``model_version`` — semantic version for the artifact

    Returns:
        Absolute path to the exported artifact directory.
    """
    training_cfg = config.get("training", {})
    export_cfg = config.get("export", {})
    dataset_cfg = config.get("dataset", {})
    agent_cfg = config.get("agent", {})

    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "artifacts/checkpoints"))
    output_dir = Path(export_cfg.get("output_dir", "artifacts/models/latest"))
    schema_version = dataset_cfg.get("schema_version", "bandit_v1")
    agent_name = agent_cfg.get("name", "")
    model_version = config.get("model_version", "0.1.0")

    logger.info("export_start", checkpoint=str(checkpoint_dir), output=str(output_dir))

    # Read model bytes from checkpoint
    model_path = checkpoint_dir / "model.pt"
    if model_path.exists():
        model_bytes = model_path.read_bytes()
    else:
        logger.warning("export_no_model", msg=f"No model.pt at {model_path}, exporting empty")
        model_bytes = b""

    # Build config dict for artifact
    config_dict = {
        "seed": config.get("seed", 42),
        "agent": agent_cfg,
        "training": training_cfg,
    }

    # Save artifact
    if dry_run:
        logger.info("export_dry_run", msg="Skipping artifact save")
        return str(output_dir.resolve())

    artifact_dir = save_artifact(
        output_dir=output_dir,
        model_bytes=model_bytes,
        config_dict=config_dict,
        schema_version=schema_version,
        agent_name=agent_name,
        model_version=model_version,
    )

    logger.info("export_done", artifact_dir=str(artifact_dir))
    return str(artifact_dir.resolve())


if __name__ == "__main__":
    run_export({"agent": {"name": "random"}, "dataset": {"schema_version": "bandit_v1"}})
