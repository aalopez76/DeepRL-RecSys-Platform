"""Serving middleware — request_id propagation and API key authentication.

Middleware is applied to the FastAPI app to:

1. **Request ID** — generate or propagate a ``X-Request-ID`` header
   via ``contextvars`` for structured log correlation.
2. **API Key Authentication** — verify ``X-API-Key`` header against a
   configured allowlist (opt-in via ``configs/serving.yaml``).
"""

from __future__ import annotations

import contextvars
import uuid
from typing import Any, Sequence

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from deeprl_recsys.core.logging import get_logger

logger = get_logger(__name__)

# Context variable for request_id propagation across async call chain
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Generate or propagate a ``X-Request-ID`` for every request.

    If the client sends an ``X-Request-ID`` header, its value is reused.
    Otherwise a UUID4 is generated.  The ID is stored in
    :data:`request_id_ctx` and echoed back in the response header.
    """

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        incoming_id = request.headers.get("x-request-id")
        rid = incoming_id or str(uuid.uuid4())

        # Set context variable for downstream use
        token = request_id_ctx.set(rid)
        try:
            logger.info("request_start", request_id=rid, path=request.url.path)
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            request_id_ctx.reset(token)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Verify ``X-API-Key`` header against an allowlist.

    If authentication is disabled (``enabled=False``) or the list of
    valid keys is empty, all requests are allowed through.

    Args:
        app: ASGI application.
        enabled: Whether to enforce authentication.
        api_keys: List of valid API keys.
        exempt_paths: Paths that bypass auth (default: ``/health``).
    """

    def __init__(
        self,
        app: Any,
        *,
        enabled: bool = False,
        api_keys: Sequence[str] = (),
        exempt_paths: Sequence[str] = ("/health", "/docs", "/openapi.json"),
    ) -> None:
        super().__init__(app)
        self.enabled = enabled
        self.api_keys = set(api_keys)
        self.exempt_paths = set(exempt_paths)

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if not self.enabled or request.url.path in self.exempt_paths:
            return await call_next(request)

        api_key = request.headers.get("x-api-key", "")
        if api_key not in self.api_keys:
            logger.warning(
                "auth_failed",
                path=request.url.path,
                reason="invalid_api_key",
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )

        return await call_next(request)
