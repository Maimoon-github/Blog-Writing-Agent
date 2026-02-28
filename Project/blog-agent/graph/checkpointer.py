"""
graph/checkpointer.py
---------------------
Sets up the LangGraph SQLite checkpointer for crash recovery and
mid-run state persistence. Consumed by graph_builder.py when compiling
the graph with persistence enabled.

Dependency:
    pip install langgraph-checkpoint-sqlite
"""

from __future__ import annotations

import pathlib  # noqa: F401 – available for callers that import this module

import structlog
from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import LOGS_DIR

logger = structlog.get_logger(__name__)


def get_checkpointer() -> SqliteSaver:
    """Create and return a :class:`SqliteSaver` backed by a local SQLite file.

    The database is stored at ``outputs/logs/checkpoints.db`` (resolved via
    ``LOGS_DIR`` from *app.config*).  The parent directory is created
    automatically if it does not exist yet.

    Returns
    -------
    SqliteSaver
        A ready-to-use checkpointer instance for compiling the LangGraph graph.

    Example
    -------
    >>> checkpointer = get_checkpointer()
    >>> graph = workflow.compile(checkpointer=checkpointer)
    """
    db_path: pathlib.Path = LOGS_DIR / "checkpoints.db"

    # Guarantee the directory tree exists before SQLite tries to open the file.
    db_path.parent.mkdir(parents=True, exist_ok=True)

    saver: SqliteSaver = SqliteSaver.from_conn_string(str(db_path))

    logger.info(
        "checkpointer_initialised",
        db_path=str(db_path),
    )

    return saver