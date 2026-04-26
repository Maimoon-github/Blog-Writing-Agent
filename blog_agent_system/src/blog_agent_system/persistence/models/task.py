# src/blog_agent_system/persistence/models/task.py
from sqlalchemy import Column, String, Integer, DateTime, DECIMAL, JSON, Text
from sqlalchemy.dialects.postgresql import UUID
from blog_agent_system.persistence.database import Base
import uuid

class Task(Base):
    __tablename__ = "tasks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(String(255), unique=True, nullable=False, index=True)
    topic = Column(Text, nullable=False)
    target_audience = Column(String(100))
    tone = Column(String(50))
    word_count_target = Column(Integer, default=1500)
    status = Column(String(50), default="pending", index=True)
    quality_score = Column(DECIMAL(3, 2))
    revision_count = Column(Integer, default=0)
    final_document_id = Column(UUID(as_uuid=True))
    created_at = Column(DateTime(timezone=True), default=func.now())
    completed_at = Column(DateTime(timezone=True))
    error_message = Column(Text)