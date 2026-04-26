# src/blog_agent_system/memory/episodic.py
from sqlalchemy.orm import Session
from blog_agent_system.persistence.models import ConversationTurn
from datetime import datetime, timedelta

class EpisodicMemory:
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def record_turn(self, thread_id: str, agent: str, role: str, content: str, tokens_used: int):
        turn = ConversationTurn(
            thread_id=thread_id,
            agent=agent,
            role=role,
            content=content,
            tokens_used=tokens_used,
            timestamp=datetime.utcnow()
        )
        self.db.add(turn)
        self.db.commit()
    
    def get_thread_history(self, thread_id: str, limit: int = 100) -> list[ConversationTurn]:
        return self.db.query(ConversationTurn).filter(
            ConversationTurn.thread_id == thread_id
        ).order_by(ConversationTurn.timestamp.desc()).limit(limit).all()
    
    def archive_old_threads(self, days: int = 90):
        cutoff = datetime.utcnow() - timedelta(days=days)
        self.db.query(ConversationTurn).filter(
            ConversationTurn.timestamp < cutoff
        ).update({"archived": True})
        self.db.commit()