"""E2E test: full pipeline — dataset → train → export → serve.

From implementación.txt Phase 6:
  e2e: dataset tiny → train → export → levantar server (testclient) → /info + /recommend
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pipelines.export import run_export
from pipelines.train import run_train


@pytest.fixture()
def artifact_dir(tmp_path: Path) -> Path:
    """Create a full artifact via train → export."""
    ckpt_dir = tmp_path / "ckpt"
    run_train(
        {
            "seed": 42,
            "agent": {"name": "random", "params": {}},
            "training": {"max_steps": 3, "checkpoint_dir": str(ckpt_dir)},
        }
    )
    artifact = run_export(
        {
            "seed": 42,
            "agent": {"name": "random"},
            "dataset": {"schema_version": "bandit_v1"},
            "training": {"checkpoint_dir": str(ckpt_dir)},
            "export": {"output_dir": str(tmp_path / "artifact")},
            "model_version": "0.0.1-e2e",
        }
    )
    return Path(artifact)


@pytest.fixture()
def e2e_client(artifact_dir: Path) -> TestClient:
    """Create a TestClient with runtime loaded from a fresh artifact."""
    from deeprl_recsys.serving.app import app, runtime

    runtime.load(artifact_dir)
    return TestClient(app)


@pytest.mark.e2e
class TestFullPipeline:
    """E2E: train → export → serve → /info + /recommend."""

    def test_info_returns_exported_metadata(self, e2e_client: TestClient) -> None:
        """GET /info must return metadata from the exported artifact."""
        resp = e2e_client.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["schema_version"] == "bandit_v1"
        assert data["agent_name"] == "random"
        assert data["model_version"] == "0.0.1-e2e"
        assert "config_fingerprint" in data
        assert len(data["config_fingerprint"]) > 0

    def test_recommend_returns_items(self, e2e_client: TestClient) -> None:
        """POST /recommend must return ranked items from the served artifact."""
        payload = {
            "request_id": "e2e-001",
            "context": {"user_id": 1},
            "candidates": [10, 20, 30, 40, 50],
            "k": 3,
        }
        resp = e2e_client.post("/recommend", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["request_id"] == "e2e-001"
        assert len(data["items"]) == 3
        assert data["latency_ms"] >= 0

    def test_health_still_works(self, e2e_client: TestClient) -> None:
        """GET /health must return ok even after loading an artifact."""
        resp = e2e_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


@pytest.mark.e2e
class TestRealisticPipeline:
    """E2E: MovieLens stub -> prepare -> train -> export -> serve."""

    def test_full_pipeline_with_movielens_stub(self, tmp_path: Path) -> None:
        """Complete pipeline using synthetic MovieLens-like data."""
        from tests.fixtures.movielens_stub import generate_movielens_stub
        from pipelines.prepare_data import run_prepare
        from pipelines.evaluate import run_evaluate
        from deeprl_recsys.serving.app import app, runtime

        # 1. Generate stub dataset
        csv_path = generate_movielens_stub(tmp_path / "raw", n_interactions=100)
        assert csv_path.exists()

        # 2. Prepare
        prep_result = run_prepare({
            "dataset": {
                "data_path": str(csv_path),
                "schema_version": "bandit_v1",
                "output_dir": str(tmp_path / "prepared"),
            },
        })
        assert prep_result["is_valid"]
        assert prep_result["n_rows"] == 100

        # 3. Train
        ckpt_dir = tmp_path / "ckpt"
        train_result = run_train({
            "seed": 42,
            "agent": {"name": "random", "params": {}},
            "training": {"max_steps": 5, "checkpoint_dir": str(ckpt_dir)},
        })
        assert train_result["steps_completed"] == 5

        # 4. Export
        artifact_path = run_export({
            "seed": 42,
            "agent": {"name": "random"},
            "dataset": {"schema_version": "bandit_v1"},
            "training": {"checkpoint_dir": str(ckpt_dir)},
            "export": {"output_dir": str(tmp_path / "artifact")},
            "model_version": "0.0.1-stub",
        })
        assert Path(artifact_path).exists()

        # 5. Serve and test
        runtime.load(Path(artifact_path))
        client = TestClient(app)

        resp = client.get("/info")
        assert resp.status_code == 200
        assert resp.json()["agent_name"] == "random"

        resp = client.post("/recommend", json={
            "request_id": "stub-001",
            "context": {"user_id": 1},
            "candidates": list(range(1, 21)),
            "k": 5,
        })
        assert resp.status_code == 200
        assert len(resp.json()["items"]) == 5
