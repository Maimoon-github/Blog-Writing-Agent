"""
Image Agent — generates cover images and section illustrations (placeholder for now).
"""

from typing import Any

from blog_agent_system.agents.base import BaseAgent
from blog_agent_system.core.state import BlogState
from blog_agent_system.utils.logging import get_logger

logger = get_logger(__name__)


class ImageAgent(BaseAgent):
    """Generates blog illustrations based on content themes."""

    def __init__(self):
        super().__init__(role="image")

    async def execute(self, state: BlogState) -> dict[str, Any]:
        logger.info("image_agent.start")

        # TODO: Integrate real image generation (DALL-E / Flux via tool)
        # Placeholder for now
        cover_image_url = None
        section_images: dict[str, str] = {}

        logger.info("image_agent.complete")

        return {
            "cover_image_url": cover_image_url,
            "section_images": section_images,
            "current_step": "image_gen",
        }


async def image_node(state: BlogState) -> dict[str, Any]:
    """LangGraph node wrapper."""
    agent = ImageAgent()
    return await agent.execute(state)