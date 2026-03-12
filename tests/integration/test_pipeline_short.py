"""Integration test: short pipeline — dataset → train → export (no serving).

From implementación.txt Phase 6:
  integration: dataset tiny sintético → train 2-5 iter → export (sin serving)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pipelines.evaluate import run_evaluate
from pipelines.export import run_export
from pipelines.prepare_data import run_prepare
from pipelines.train import run_train


@pytest.fixture()
def tiny_csv(tmp_path: Path) -> Path:
    """Create a tiny CSV dataset for integration testing."""
    df = pd.DataFrame(
        {
            "action": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "reward": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "timestamp": [float(i) for i in range(10)],
            "propensity": [0.2, 0.3, 0.5, 0.1, 0.8, 0.4, 0.6, 0.2, 0.9, 0.3],
        }
    )
    csv_path = tmp_path / "tiny_bandit.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.mark.integration
class TestShortPipeline:
    """Integration test: prepare → train → evaluate → export."""

    def test_prepare_validates_tiny_dataset(self, tiny_csv: Path, tmp_path: Path) -> None:
        """prepare_data must validate a tiny dataset successfully."""
        config = {
            "dataset": {
                "data_path": str(tiny_csv),
                "schema_version": "bandit_v1",
                "output_dir": str(tmp_path / "clean"),
            }
        }
        result = run_prepare(config)
        assert result["is_valid"]
        assert result["n_rows"] == 10
        assert Path(result["output_path"]).exists()

    def test_train_produces_checkpoint(self, tmp_path: Path) -> None:
        """train must produce a checkpoint file."""
        config = {
            "seed": 42,
            "agent": {"name": "random", "params": {}},
            "training": {"max_steps": 3, "checkpoint_dir": str(tmp_path / "ckpt")},
        }
        result = run_train(config)
        assert result["agent_name"] == "random"
        assert result["steps_completed"] == 3
        assert Path(result["model_path"]).exists()
        assert len(result["metrics"]) == 3

    def test_evaluate_produces_verdict(self) -> None:
        """evaluate must produce estimates and a verdict."""
        config = {
            "seed": 42,
            "ope": {
                "estimators": ["ips", "dr"],
                "clip_epsilon": 0.01,
                "n_samples": 50,
            },
        }
        result = run_evaluate(config)
        assert "ips" in result["estimates"]
        assert "dr" in result["estimates"]
        assert result["severity"] in ("ok", "warning", "error")

    def test_export_creates_artifact(self, tmp_path: Path) -> None:
        """export must create a canonical artifact directory."""
        # First train
        ckpt_dir = tmp_path / "ckpt"
        train_config = {
            "seed": 42,
            "agent": {"name": "random", "params": {}},
            "training": {"max_steps": 2, "checkpoint_dir": str(ckpt_dir)},
        }
        run_train(train_config)

        # Then export
        export_config = {
            "seed": 42,
            "agent": {"name": "random"},
            "dataset": {"schema_version": "bandit_v1"},
            "training": {"checkpoint_dir": str(ckpt_dir)},
            "export": {"output_dir": str(tmp_path / "artifact")},
            "model_version": "0.0.1-test",
        }
        artifact_path = run_export(export_config)
        adir = Path(artifact_path)

        assert (adir / "metadata.json").exists()
        assert (adir / "config.yaml").exists()
        assert (adir / "schema_version.txt").exists()
        assert (adir / "checksums.txt").exists()
        assert (adir / "model.pt").exists()

        meta = json.loads((adir / "metadata.json").read_text())
        assert meta["agent_name"] == "random"
        assert meta["schema_version"] == "bandit_v1"

    def test_full_short_pipeline(self, tiny_csv: Path, tmp_path: Path) -> None:
        """End-to-end: prepare → train → evaluate → export."""
        # 1. Prepare
        prep_result = run_prepare(
            {
                "dataset": {
                    "data_path": str(tiny_csv),
                    "schema_version": "bandit_v1",
                    "output_dir": str(tmp_path / "clean"),
                }
            }
        )
        assert prep_result["is_valid"]

        # 2. Train
        ckpt_dir = tmp_path / "ckpt"
        train_result = run_train(
            {
                "seed": 42,
                "agent": {"name": "random", "params": {}},
                "training": {"max_steps": 3, "checkpoint_dir": str(ckpt_dir)},
            }
        )
        assert train_result["steps_completed"] == 3

        # 3. Evaluate
        eval_result = run_evaluate(
            {
                "seed": 42,
                "ope": {"estimators": ["ips"], "n_samples": 30},
            }
        )
        assert "ips" in eval_result["estimates"]

        # 4. Export
        artifact_path = run_export(
            {
                "seed": 42,
                "agent": {"name": "random"},
                "dataset": {"schema_version": "bandit_v1"},
                "training": {"checkpoint_dir": str(ckpt_dir)},
                "export": {"output_dir": str(tmp_path / "artifact")},
            }
        )
        assert Path(artifact_path).exists()
        assert (Path(artifact_path) / "metadata.json").exists()
