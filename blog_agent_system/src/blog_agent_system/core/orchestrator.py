"""Blog Orchestrator — central supervisor."""

import uuid
from typing import Any, Optional

from blog_agent_system.core.graph import create_blog_graph
from blog_agent_system.core.checkpoint import create_checkpointer
from blog_agent_system.utils.logging import get_logger, new_correlation_id

logger = get_logger(__name__)


class BlogOrchestrator:
    """Main supervisor for the blog generation pipeline."""

    def __init__(self):
        self.checkpointer = None
        self.graph = None

    async def initialize(self):
        """Lazy initialization with checkpointer."""
        if self.checkpointer is None:
            self.checkpointer = await create_checkpointer()
            self.graph = create_blog_graph(checkpointer=self.checkpointer)
        return self

    async def generate_blog(
        self,
        topic: str,
        target_audience: str = "technical professionals",
        tone: str = "informative yet conversational",
        word_count_target: int = 1500,
        include_images: bool = True,
        style_guide: str = "AP",
        thread_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Execute the full multi-agent pipeline."""
        await self.initialize()

        thread_id = thread_id or f"blog-{uuid.uuid4().hex[:12]}"
        correlation_id = new_correlation_id()

        logger.info("orchestrator.start", thread_id=thread_id, topic=topic)

        initial_state = {
            "topic": topic,
            "target_audience": target_audience,
            "tone": tone,
            "word_count_target": word_count_target,
            "include_images": include_images,
            "style_guide": style_guide,
            "status": "pending",
            "messages": [],
        }

        config = {"configurable": {"thread_id": thread_id}}

        try:
            final_state = await self.graph.ainvoke(initial_state, config=config)

            logger.info(
                "orchestrator.complete",
                thread_id=thread_id,
                quality_score=final_state.get("quality_score", 0.0),
                revisions=final_state.get("revision_count", 0),
            )

            return {
                "thread_id": thread_id,
                "correlation_id": correlation_id,
                "final_blog": final_state.get("final_blog", ""),
                "draft": final_state.get("draft", ""),
                "quality_score": final_state.get("quality_score", 0.0),
                "revision_count": final_state.get("revision_count", 0),
                "seo_metadata": final_state.get("seo_metadata"),
                "fact_check_results": final_state.get("fact_check_results", []),
                "status": "complete",
            }

        except Exception as e:
            logger.error("orchestrator.error", thread_id=thread_id, error=str(e), exc_info=True)
            return {
                "thread_id": thread_id,
                "correlation_id": correlation_id,
                "status": "error",
                "error": str(e),
            }

    async def get_status(self, thread_id: str) -> dict[str, Any]:
        """Get current workflow status."""
        if not self.checkpointer:
            return {"thread_id": thread_id, "status": "unknown"}

        config = {"configurable": {"thread_id": thread_id}}
        state = await self.graph.aget_state(config)

        if state and state.values:
            return {
                "thread_id": thread_id,
                "status": state.values.get("status", "unknown"),
                "current_step": state.values.get("current_step", "unknown"),
                "quality_score": state.values.get("quality_score", 0.0),
                "revision_count": state.values.get("revision_count", 0),
            }

        return {"thread_id": thread_id, "status": "not_found"}