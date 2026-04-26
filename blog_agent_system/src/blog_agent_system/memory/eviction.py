"""
Memory eviction — TTL policies, LRU eviction, and memory compaction.
"""

from blog_agent_system.utils.logging import get_logger

logger = get_logger(__name__)


class EvictionManager:
    """
    Manages memory eviction across tiers:
    - Short-term: Automatic via Redis TTL (no code needed)
    - Episodic: Batch archive job for old threads
    - Long-term: Manual review (no automatic eviction to prevent knowledge loss)
    """

    def __init__(self, episodic_memory=None):
        self.episodic = episodic_memory

    async def run_nightly_cleanup(self, archive_days: int = 90) -> dict:
        """Run nightly cleanup tasks across memory tiers."""
        results = {}

        if self.episodic:
            archived = self.episodic.archive_old_threads(days=archive_days)
            results["episodic_archived"] = archived

        logger.info("eviction.nightly_complete", results=results)
        return results
