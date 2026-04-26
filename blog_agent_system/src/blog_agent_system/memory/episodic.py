"""
Episodic memory — PostgreSQL conversation history for audit and replay.
"""

from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session

from blog_agent_system.utils.logging import get_logger

logger = get_logger(__name__)


class EpisodicMemory:
    """Thread-scoped conversation history stored in PostgreSQL."""

    def __init__(self, db_session: Session):
        self.db = db_session

    def record_turn(
        self,
        thread_id: str,
        agent: str,
        role: str,
        content: str,
        tokens_used: int,
    ) -> None:
        """Record a single conversation turn to the database."""
        from blog_agent_system.persistence.models.agent_run import ConversationTurn

        turn = ConversationTurn(
            thread_id=thread_id,
            agent_name=agent,
            role=role,
            content=content,
            tokens_used=tokens_used,
            timestamp=datetime.now(timezone.utc),
        )
        self.db.add(turn)
        self.db.commit()
        logger.debug("episodic.recorded", thread_id=thread_id, agent=agent)

    def get_thread_history(self, thread_id: str, limit: int = 100) -> list:
        """Retrieve conversation history for a thread."""
        from blog_agent_system.persistence.models.agent_run import ConversationTurn

        return (
            self.db.query(ConversationTurn)
            .filter(ConversationTurn.thread_id == thread_id)
            .order_by(ConversationTurn.timestamp.desc())
            .limit(limit)
            .all()
        )

    def archive_old_threads(self, days: int = 90) -> int:
        """Soft-archive threads older than the specified number of days."""
        from blog_agent_system.persistence.models.agent_run import ConversationTurn

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        count = (
            self.db.query(ConversationTurn)
            .filter(ConversationTurn.timestamp < cutoff)
            .update({"archived": True})
        )
        self.db.commit()
        logger.info("episodic.archived", count=count, days=days)
        return count