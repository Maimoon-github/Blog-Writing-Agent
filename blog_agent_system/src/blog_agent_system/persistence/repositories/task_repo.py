"""
Repository: Task — data access abstraction for tasks table.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from blog_agent_system.persistence.models.task import Task


class TaskRepository:
    """Data access layer for blog generation tasks."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, task: Task) -> Task:
        self.db.add(task)
        await self.db.commit()
        await self.db.refresh(task)
        return task

    async def get_by_thread(self, thread_id: str) -> Optional[Task]:
        result = await self.db.execute(
            select(Task).where(Task.thread_id == thread_id)
        )
        return result.scalar_one_or_none()

    async def get_by_id(self, task_id: uuid.UUID) -> Optional[Task]:
        result = await self.db.execute(
            select(Task).where(Task.id == task_id)
        )
        return result.scalar_one_or_none()

    async def update_status(self, task_id: uuid.UUID, status: str) -> None:
        task = await self.get_by_id(task_id)
        if task:
            task.status = status
            if status == "complete":
                task.completed_at = datetime.now(timezone.utc)
            await self.db.commit()

    async def list_recent(self, limit: int = 20) -> list[Task]:
        result = await self.db.execute(
            select(Task).order_by(Task.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())