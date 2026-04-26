"""
Structured JSON logging with correlation IDs.
"""

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

import structlog


# ─── Correlation ID for request tracing ───
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    cid = _correlation_id.get()
    if not cid:
        cid = str(uuid.uuid4())[:8]
        _correlation_id.set(cid)
    return cid


def set_correlation_id(correlation_id: str) -> None:
    _correlation_id.set(correlation_id)


def new_correlation_id() -> str:
    cid = str(uuid.uuid4())[:8]
    _correlation_id.set(cid)
    return cid


def _add_correlation_id(logger: Any, method_name: str, event_dict: dict) -> dict:
    event_dict["correlation_id"] = get_correlation_id()
    return event_dict


def setup_logging(log_level: str = "INFO", log_format: str = "console") -> None:
    """Configure structlog. Call once at application startup."""
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        _add_correlation_id,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Suppress noisy loggers
    for name in ["httpx", "httpcore", "openai", "anthropic", "chromadb"]:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)
