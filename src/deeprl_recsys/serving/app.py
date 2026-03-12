"""FastAPI application — ``/health``, ``/info``, ``/recommend`` endpoints.

The application is created at module level so that ``uvicorn`` can
reference it as ``deeprl_recsys.serving.app:app``.  The global
:data:`runtime` is loaded via a lifespan event or by calling
:meth:`runtime.load` directly before starting the server.
"""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException

from deeprl_recsys.serving.runtime import ServingRuntime
from deeprl_recsys.serving.schemas import (
    InfoResponse,
    RecommendItem,
    RecommendRequest,
    RecommendResponse,
)
from deeprl_recsys.serving.middleware import ApiKeyMiddleware, RequestIdMiddleware

app = FastAPI(
    title="DeepRL-RecSys",
    version="0.1.0",
    description="Deep RL-based Recommendation Serving API",
)

# Middleware (order matters: outermost first)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(ApiKeyMiddleware, enabled=False, api_keys=[])

# Global runtime instance — loaded on startup or via CLI
runtime = ServingRuntime()


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check — always returns ``{"status": "ok"}``."""
    return {"status": "ok"}


@app.get("/info", response_model=InfoResponse)
async def info() -> InfoResponse:
    """Return loaded artifact metadata.

    Returns metadata from the currently loaded artifact, including
    version, agent name, schema version, checksums, and config
    fingerprint.
    """
    return InfoResponse(**runtime.metadata)


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest) -> RecommendResponse:
    """Generate ranked recommendations.

    Accepts a list of candidate item IDs and returns the top-k
    ranked by the loaded model.  If no model is loaded, returns
    candidates in their original order with score 0.0.

    Raises:
        HTTPException 400: If ``k`` ≤ 0 or candidates list is empty.
    """
    if not request.candidates:
        raise HTTPException(status_code=400, detail="candidates list must not be empty")

    start = time.perf_counter()
    results = runtime.predict(request.context, request.candidates, request.k)
    latency = (time.perf_counter() - start) * 1000

    items = [RecommendItem(item_id=r["item_id"], score=r["score"]) for r in results]

    return RecommendResponse(
        request_id=request.request_id,
        items=items,
        model_version=runtime.metadata.get("model_version", ""),
        schema_version=runtime.metadata.get("schema_version", ""),
        latency_ms=latency,
    )
