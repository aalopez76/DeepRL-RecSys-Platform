"""Pipeline: serve — launch uvicorn with artifact."""

from __future__ import annotations

from typing import Any


def run_serve(
    artifact_dir: str,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Launch the serving endpoint.

    Args:
        artifact_dir: Path to the canonical artifact directory.
        host: Server host.
        port: Server port.
    """
    import uvicorn

    from deeprl_recsys.serving.app import app, runtime

    runtime.load(artifact_dir)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_serve("artifacts/models/latest")
