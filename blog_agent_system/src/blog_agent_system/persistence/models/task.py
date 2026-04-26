"""
ORM model: Task — blog generation task tracking.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from blog_agent_system.persistence.database import Base


class Task(Base):
    __tablename__ = "tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(String(255), unique=True, nullable=False, index=True)
    topic = Column(Text, nullable=False)
    target_audience = Column(String(100), default="technical professionals")
    tone = Column(String(50), default="informative")
    word_count_target = Column(Integer, default=1500)
    status = Column(String(50), default="pending", index=True)
    quality_score = Column(Numeric(3, 2))
    revision_count = Column(Integer, default=0)
    final_document_id = Column(UUID(as_uuid=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    agent_runs = relationship("AgentRun", back_populates="task", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="task", cascade="all, delete-orphan")