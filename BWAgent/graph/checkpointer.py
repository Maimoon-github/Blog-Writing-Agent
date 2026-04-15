"""SQLite checkpointer setup for BWAgent LangGraph persistence."""

import asyncio
import pathlib

import structlog
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from config.settings import LOGS_DIR

logger = structlog.get_logger(__name__)
CHECKPOINT_DB_FILENAME = "checkpoints.db"


class CheckpointerError(RuntimeError):
    pass


def get_checkpointer() -> SqliteSaver:
    db_path = LOGS_DIR / CHECKPOINT_DB_FILENAME
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("checkpointer.directory_ready", path=str(db_path.parent))
    except OSError as exc:
        logger.error("checkpointer.directory_failed", error=str(exc), path=str(db_path.parent))
        raise CheckpointerError(f"Cannot create checkpointer directory: {exc}") from exc

    try:
        saver = SqliteSaver.from_conn_string(str(db_path))
        logger.info("checkpointer.initialized", db_path=str(db_path))
        return saver
    except Exception as exc:
        logger.error("checkpointer.init_failed", error=str(exc), db_path=str(db_path))
        raise CheckpointerError(f"Failed to initialize checkpointer: {exc}") from exc


async def get_async_checkpointer() -> AsyncSqliteSaver:
    db_path = LOGS_DIR / CHECKPOINT_DB_FILENAME
    try:
        await asyncio.to_thread(db_path.parent.mkdir, parents=True, exist_ok=True)
        logger.debug("checkpointer.async_directory_ready", path=str(db_path.parent))
    except OSError as exc:
        logger.error("checkpointer.async_directory_failed", error=str(exc), path=str(db_path.parent))
        raise CheckpointerError(f"Cannot create checkpointer directory: {exc}") from exc

    try:
        saver = await AsyncSqliteSaver.from_conn_string(str(db_path))
        logger.info("checkpointer.async_initialized", db_path=str(db_path))
        return saver
    except Exception as exc:
        logger.error("checkpointer.async_init_failed", error=str(exc), db_path=str(db_path))
        raise CheckpointerError(f"Failed to initialize async checkpointer: {exc}") from exc


__all__ = ["get_checkpointer", "get_async_checkpointer", "CheckpointerError"]
