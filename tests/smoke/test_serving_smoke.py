"""Smoke tests for the serving layer — /health, /info, /recommend.

Required tests from implementación.txt Phase 5:
- test_info_endpoint_returns_metadata()
- test_recommend_endpoint_returns_k_items()
Plus health check, error handling, and edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient


# ── Fixture: tiny_artifact path ──────────────────────
TINY_ARTIFACT = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_artifact"


@pytest.fixture()
def client() -> TestClient:
    """Create a TestClient with runtime loaded from tiny_artifact."""
    from deeprl_recsys.serving.app import app, runtime

    if TINY_ARTIFACT.exists():
        runtime.load(TINY_ARTIFACT)
    return TestClient(app)


@pytest.fixture()
def unloaded_client() -> TestClient:
    """Create a TestClient without any artifact loaded."""
    from deeprl_recsys.serving.app import app, runtime

    runtime.metadata = {}
    runtime._loaded = False
    return TestClient(app)


# ── Contract Tests (from implementación.txt) ─────────


@pytest.mark.smoke
class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_ok(self, client: TestClient) -> None:
        """Health check must return 200 with status=ok."""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


@pytest.mark.smoke
class TestInfoEndpoint:
    """Tests for GET /info."""

    @pytest.mark.skipif(
        not TINY_ARTIFACT.exists(),
        reason="tiny_artifact fixture not found",
    )
    def test_info_endpoint_returns_metadata(self, client: TestClient) -> None:
        """GET /info must return artifact metadata fields."""
        resp = client.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["artifact_version"] == "1.0.0"
        assert data["schema_version"] == "bandit_v1"
        assert data["agent_name"] == "random"
        assert "checksums" in data

    @pytest.mark.skipif(
        not TINY_ARTIFACT.exists(),
        reason="tiny_artifact fixture not found",
    )
    def test_info_has_config_fingerprint(self, client: TestClient) -> None:
        """GET /info must include a config_fingerprint."""
        resp = client.get("/info")
        data = resp.json()
        assert "config_fingerprint" in data
        assert len(data["config_fingerprint"]) > 0

    def test_info_unloaded_returns_defaults(self, unloaded_client: TestClient) -> None:
        """GET /info without loaded artifact returns empty defaults."""
        resp = unloaded_client.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["artifact_version"] == ""


@pytest.mark.smoke
class TestRecommendEndpoint:
    """Tests for POST /recommend."""

    def test_recommend_endpoint_returns_k_items(self, client: TestClient) -> None:
        """POST /recommend must return exactly k items (or fewer if candidates < k)."""
        payload = {
            "request_id": "test-001",
            "context": {"user_id": 42},
            "candidates": [101, 102, 103, 104, 105],
            "k": 3,
        }
        resp = client.post("/recommend", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["request_id"] == "test-001"
        assert len(data["items"]) == 3
        assert all("item_id" in it and "score" in it for it in data["items"])

    def test_recommend_returns_fewer_if_candidates_less_than_k(
        self, client: TestClient
    ) -> None:
        """If candidates < k, return all candidates."""
        payload = {
            "request_id": "test-002",
            "context": {},
            "candidates": [10, 20],
            "k": 5,
        }
        resp = client.post("/recommend", json=payload)
        assert resp.status_code == 200
        assert len(resp.json()["items"]) == 2

    def test_recommend_includes_latency(self, client: TestClient) -> None:
        """Response must include latency_ms."""
        payload = {
            "request_id": "test-003",
            "context": {},
            "candidates": [1, 2, 3],
            "k": 2,
        }
        resp = client.post("/recommend", json=payload)
        data = resp.json()
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0

    def test_recommend_empty_candidates_returns_400(self, client: TestClient) -> None:
        """Empty candidates list must return HTTP 400."""
        payload = {
            "request_id": "test-004",
            "context": {},
            "candidates": [],
            "k": 5,
        }
        resp = client.post("/recommend", json=payload)
        assert resp.status_code == 400

    def test_recommend_k_zero_returns_422(self, client: TestClient) -> None:
        """k=0 must be rejected by Pydantic validation (HTTP 422)."""
        payload = {
            "request_id": "test-005",
            "context": {},
            "candidates": [1, 2],
            "k": 0,
        }
        resp = client.post("/recommend", json=payload)
        assert resp.status_code == 422

    def test_recommend_negative_k_returns_422(self, client: TestClient) -> None:
        """k<0 must be rejected by Pydantic validation (HTTP 422)."""
        payload = {
            "request_id": "test-006",
            "context": {},
            "candidates": [1],
            "k": -1,
        }
        resp = client.post("/recommend", json=payload)
        assert resp.status_code == 422

    def test_recommend_missing_request_id_returns_422(self, client: TestClient) -> None:
        """Missing required field must return HTTP 422."""
        payload = {"candidates": [1, 2, 3], "k": 2}
        resp = client.post("/recommend", json=payload)
        assert resp.status_code == 422
