"""
Database seed script — initial data for development.
"""

import asyncio

from blog_agent_system.persistence.database import init_db
from blog_agent_system.utils.logging import setup_logging, get_logger


async def main():
    setup_logging()
    logger = get_logger(__name__)

    logger.info("seed.start")

    # Create tables
    await init_db()
    logger.info("seed.tables_created")

    # TODO: Add seed data (style guides, example blog posts for RAG)
    logger.info("seed.complete")


if __name__ == "__main__":
    asyncio.run(main())
