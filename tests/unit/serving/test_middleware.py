"""Tests for serving middleware — request_id and API key auth."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from deeprl_recsys.serving.middleware import (
    ApiKeyMiddleware,
    RequestIdMiddleware,
    request_id_ctx,
)


def _make_app(*, auth_enabled: bool = False, api_keys: list[str] | None = None) -> FastAPI:
    """Create a minimal FastAPI app with middleware for testing."""
    test_app = FastAPI()
    test_app.add_middleware(RequestIdMiddleware)
    test_app.add_middleware(
        ApiKeyMiddleware,
        enabled=auth_enabled,
        api_keys=api_keys or [],
    )

    @test_app.get("/test")
    async def _test_endpoint() -> dict:
        return {"rid": request_id_ctx.get()}

    @test_app.get("/health")
    async def _health() -> dict:
        return {"status": "ok"}

    return test_app


@pytest.mark.unit
class TestRequestIdMiddleware:
    """Tests for X-Request-ID propagation."""

    def test_generates_request_id(self) -> None:
        """Middleware generates X-Request-ID if not provided."""
        client = TestClient(_make_app())
        resp = client.get("/test")
        assert resp.status_code == 200
        assert "x-request-id" in resp.headers
        assert len(resp.headers["x-request-id"]) > 0

    def test_propagates_incoming_request_id(self) -> None:
        """Middleware reuses X-Request-ID if sent by client."""
        client = TestClient(_make_app())
        resp = client.get("/test", headers={"x-request-id": "custom-123"})
        assert resp.headers["x-request-id"] == "custom-123"
        assert resp.json()["rid"] == "custom-123"

    def test_context_var_available(self) -> None:
        """request_id is accessible in the endpoint via contextvars."""
        client = TestClient(_make_app())
        resp = client.get("/test", headers={"x-request-id": "ctx-test"})
        assert resp.json()["rid"] == "ctx-test"


@pytest.mark.unit
class TestApiKeyMiddleware:
    """Tests for API key authentication middleware."""

    def test_disabled_auth_allows_all(self) -> None:
        """When auth is disabled, all requests pass."""
        client = TestClient(_make_app(auth_enabled=False))
        resp = client.get("/test")
        assert resp.status_code == 200

    def test_enabled_auth_without_key_returns_401(self) -> None:
        """Enabled auth without valid key returns 401."""
        client = TestClient(_make_app(auth_enabled=True, api_keys=["secret"]))
        resp = client.get("/test")
        assert resp.status_code == 401
        assert "Invalid" in resp.json()["detail"]

    def test_enabled_auth_with_valid_key_passes(self) -> None:
        """Enabled auth with valid X-API-Key passes."""
        client = TestClient(_make_app(auth_enabled=True, api_keys=["my-key"]))
        resp = client.get("/test", headers={"x-api-key": "my-key"})
        assert resp.status_code == 200

    def test_enabled_auth_with_wrong_key_returns_401(self) -> None:
        """Wrong key returns 401."""
        client = TestClient(_make_app(auth_enabled=True, api_keys=["correct"]))
        resp = client.get("/test", headers={"x-api-key": "wrong"})
        assert resp.status_code == 401

    def test_health_exempt_from_auth(self) -> None:
        """/health is exempt from auth even when enabled."""
        client = TestClient(_make_app(auth_enabled=True, api_keys=["secret"]))
        resp = client.get("/health")
        assert resp.status_code == 200
