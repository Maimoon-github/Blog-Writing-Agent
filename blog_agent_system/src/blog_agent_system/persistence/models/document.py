"""
ORM models: Document and Feedback — versioned blog outputs and RLHF data.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, CheckConstraint
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import relationship

from blog_agent_system.persistence.database import Base


class Document(Base):
    """Versioned blog post output."""

    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id", ondelete="CASCADE"))
    version = Column(Integer, nullable=False, default=1)
    title = Column(String(500), nullable=True)
    content = Column(Text, nullable=False)
    format = Column(String(20), default="markdown")
    seo_metadata = Column(JSON, nullable=True)
    fact_check_summary = Column(JSON, nullable=True)
    word_count = Column(Integer, default=0)
    reading_time_minutes = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    task = relationship("Task", back_populates="documents")
    feedback = relationship("Feedback", back_populates="document", cascade="all, delete-orphan")


class Feedback(Base):
    """Human-in-the-loop feedback for RLHF data collection."""

    __tablename__ = "feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"))
    rating = Column(Integer, CheckConstraint("rating BETWEEN 1 AND 5"))
    comments = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    document = relationship("Document", back_populates="feedback")