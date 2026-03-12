"""Structured JSON logging for the platform.

Provides a pre-configured ``structlog`` logger that outputs JSON by default
and supports ``request_id`` binding for serving traceability.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Configure structured logging for the platform.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        fmt: Output format — ``"json"`` for machine-readable, ``"text"`` for human-readable.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if fmt == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(level=log_level, format="%(message)s", stream=sys.stderr)


def get_logger(name: str | None = None, **initial_bindings: Any) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (typically ``__name__``).
        **initial_bindings: Key-value pairs to bind to every log message.

    Returns:
        A bound structured logger.
    """
    logger = structlog.get_logger(name)
    if initial_bindings:
        logger = logger.bind(**initial_bindings)
    return logger
