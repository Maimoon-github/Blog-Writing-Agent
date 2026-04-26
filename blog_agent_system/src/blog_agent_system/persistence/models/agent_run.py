"""
ORM models: AgentRun and ConversationTurn — execution audit and episodic memory.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import relationship

from blog_agent_system.persistence.database import Base


class AgentRun(Base):
    """Per-agent execution log for audit and debugging."""

    __tablename__ = "agent_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id", ondelete="CASCADE"))
    agent_name = Column(String(50), nullable=False)
    input_state = Column(JSON, nullable=True)
    output_state = Column(JSON, nullable=True)
    tokens_input = Column(Integer, default=0)
    tokens_output = Column(Integer, default=0)
    latency_ms = Column(Integer, default=0)
    status = Column(String(20), default="running")
    error = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True), nullable=True)

    task = relationship("Task", back_populates="agent_runs")


class ConversationTurn(Base):
    """Episodic memory — conversation history per thread."""

    __tablename__ = "conversation_turns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(String(255), nullable=False, index=True)
    agent_name = Column(String(50), nullable=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    tokens_used = Column(Integer, default=0)
    tool_calls = Column(JSON, nullable=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    archived = Column(Boolean, default=False)
