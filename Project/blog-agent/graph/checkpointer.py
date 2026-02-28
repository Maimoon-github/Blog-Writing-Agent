# """
# graph/checkpointer.py
# ---------------------
# Sets up the LangGraph SQLite checkpointer for crash recovery and
# mid-run state persistence. Consumed by graph_builder.py when compiling
# the graph with persistence enabled.

# Dependency:
#     pip install langgraph-checkpoint-sqlite
# """

# from __future__ import annotations

# import pathlib  # noqa: F401 – available for callers that import this module

# import structlog
# from langgraph.checkpoint.sqlite import SqliteSaver

# from app.config import LOGS_DIR

# logger = structlog.get_logger(__name__)


# def get_checkpointer() -> SqliteSaver:
#     """Create and return a :class:`SqliteSaver` backed by a local SQLite file.

#     The database is stored at ``outputs/logs/checkpoints.db`` (resolved via
#     ``LOGS_DIR`` from *app.config*).  The parent directory is created
#     automatically if it does not exist yet.

#     Returns
#     -------
#     SqliteSaver
#         A ready-to-use checkpointer instance for compiling the LangGraph graph.

#     Example
#     -------
#     >>> checkpointer = get_checkpointer()
#     >>> graph = workflow.compile(checkpointer=checkpointer)
#     """
#     db_path: pathlib.Path = LOGS_DIR / "checkpoints.db"

#     # Guarantee the directory tree exists before SQLite tries to open the file.
#     db_path.parent.mkdir(parents=True, exist_ok=True)

#     saver: SqliteSaver = SqliteSaver.from_conn_string(str(db_path))

#     logger.info(
#         "checkpointer_initialised",
#         db_path=str(db_path),
#     )

#     return saver





















"""LangGraph SQLite checkpointer setup for state persistence with crash recovery.

This module provides factory functions to create LangGraph checkpointers backed by
a local SQLite database. It handles filesystem operations, error recovery, and
structured logging for both synchronous and asynchronous execution contexts.

The checkpointers enable:
    - Mid-run state persistence for crash recovery
    - Conversation memory across multiple interactions
    - Human-in-the-loop workflows via state checkpointing
    - Thread-based conversation isolation

Example:
    >>> from graph.checkpointer import get_checkpointer
    >>> checkpointer = get_checkpointer()
    >>> graph = workflow.compile(checkpointer=checkpointer)
"""

from __future__ import annotations

import asyncio
import pathlib
from typing import Optional

import structlog
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import LOGS_DIR

logger = structlog.get_logger(__name__)

# Database filename constant for consistency
CHECKPOINT_DB_FILENAME = "checkpoints.db"


class CheckpointerError(RuntimeError):
    """Custom exception for checkpointer initialization failures."""
    pass


def get_checkpointer() -> SqliteSaver:
    """Create and return a synchronous SQLite checkpointer.

    The checkpointer uses a SQLite database file located at `LOGS_DIR / "checkpoints.db"`.
    The parent directory is created automatically if it doesn't exist.

    Returns:
        SqliteSaver: A configured checkpointer instance ready for graph compilation.

    Raises:
        CheckpointerError: If directory creation or database connection fails,
                          with detailed context about the failure.

    Example:
        >>> checkpointer = get_checkpointer()
        >>> graph = workflow.compile(checkpointer=checkpointer)
        >>> # Use with thread_id for conversation isolation
        >>> config = {"configurable": {"thread_id": "user-123"}}
        >>> result = graph.invoke(input_data, config)
    """
    db_path: pathlib.Path = LOGS_DIR / CHECKPOINT_DB_FILENAME

    try:
        # Ensure the directory tree exists with proper permissions
        db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("Database directory ensured", path=str(db_path.parent))
    except OSError as e:
        logger.error(
            "Failed to create checkpointer directory",
            path=str(db_path.parent),
            error=str(e),
            error_type=e.__class__.__name__,
        )
        raise CheckpointerError(
            f"Cannot create checkpointer directory at {db_path.parent}: {e}"
        ) from e

    try:
        # Initialize the SQLite checkpointer with the database path
        # from_conn_string handles both new and existing databases
        saver: SqliteSaver = SqliteSaver.from_conn_string(str(db_path))

        logger.info(
            "Checkpointer initialized successfully",
            db_path=str(db_path),
            checkpointer_type="sync",
        )

        return saver

    except Exception as e:
        # Catch-all for database-related errors (sqlite3.OperationalError, etc.)
        logger.error(
            "Failed to initialize checkpointer",
            db_path=str(db_path),
            error=str(e),
            error_type=e.__class__.__name__,
        )
        raise CheckpointerError(
            f"Failed to connect to checkpointer database at {db_path}: {e}"
        ) from e


async def get_async_checkpointer() -> AsyncSqliteSaver:
    """Create and return an asynchronous SQLite checkpointer.

    This version is designed for use with async graph execution (.ainvoke, .astream).
    Directory creation is offloaded to a thread pool to avoid blocking the event loop.

    Returns:
        AsyncSqliteSaver: A configured async checkpointer instance.

    Raises:
        CheckpointerError: If directory creation or database connection fails.

    Example:
        >>> checkpointer = await get_async_checkpointer()
        >>> graph = workflow.compile(checkpointer=checkpointer)
        >>> config = {"configurable": {"thread_id": "user-123"}}
        >>> result = await graph.ainvoke(input_data, config)

    Note:
        Directory creation uses asyncio.to_thread for non-blocking filesystem I/O.
        The database connection itself is handled by aiosqlite via AsyncSqliteSaver.
    """
    db_path: pathlib.Path = LOGS_DIR / CHECKPOINT_DB_FILENAME

    try:
        # Offload blocking filesystem operation to thread pool
        await asyncio.to_thread(db_path.parent.mkdir, parents=True, exist_ok=True)
        logger.debug("Database directory ensured (async)", path=str(db_path.parent))
    except OSError as e:
        logger.error(
            "Failed to create checkpointer directory (async)",
            path=str(db_path.parent),
            error=str(e),
            error_type=e.__class__.__name__,
        )
        raise CheckpointerError(
            f"Cannot create checkpointer directory at {db_path.parent}: {e}"
        ) from e

    try:
        # Initialize async checkpointer (uses aiosqlite internally)
        saver: AsyncSqliteSaver = await AsyncSqliteSaver.from_conn_string(str(db_path))

        logger.info(
            "Async checkpointer initialized successfully",
            db_path=str(db_path),
            checkpointer_type="async",
        )

        return saver

    except Exception as e:
        logger.error(
            "Failed to initialize async checkpointer",
            db_path=str(db_path),
            error=str(e),
            error_type=e.__class__.__name__,
        )
        raise CheckpointerError(
            f"Failed to connect to async checkpointer database at {db_path}: {e}"
        ) from e


# Convenience exports for common use cases
__all__ = [
    "get_checkpointer",
    "get_async_checkpointer",
    "CheckpointerError",
    "CHECKPOINT_DB_FILENAME",
]