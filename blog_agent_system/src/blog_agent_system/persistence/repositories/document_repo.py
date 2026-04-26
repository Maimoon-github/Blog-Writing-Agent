"""
Repository: Document — data access for blog post versions.
"""

import uuid
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from blog_agent_system.persistence.models.document import Document


class DocumentRepository:
    """Data access layer for generated blog documents."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, document: Document) -> Document:
        self.db.add(document)
        await self.db.commit()
        await self.db.refresh(document)
        return document

    async def get_by_task(self, task_id: uuid.UUID) -> list[Document]:
        result = await self.db.execute(
            select(Document)
            .where(Document.task_id == task_id)
            .order_by(Document.version.desc())
        )
        return list(result.scalars().all())

    async def get_latest_by_task(self, task_id: uuid.UUID) -> Optional[Document]:
        result = await self.db.execute(
            select(Document)
            .where(Document.task_id == task_id)
            .order_by(Document.version.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()