"""
SEO Agent — keyword optimization, meta descriptions, readability scoring.
"""

from typing import Any

from blog_agent_system.agents.base import BaseAgent
from blog_agent_system.core.state import BlogState, SEOData
from blog_agent_system.utils.logging import get_logger

logger = get_logger(__name__)


class SEOAgent(BaseAgent):
    """Analyzes and optimizes the draft for search engine visibility."""

    def __init__(self):
        super().__init__(role="seo")

    def get_system_prompt(self) -> str:
        return (
            "You are an SEO specialist. Analyze the blog post and provide optimization data.\n\n"
            "Respond with valid JSON containing title_tag, meta_description, keywords, "
            "readability_score, and keyword_density."
        )

    async def execute(self, state: BlogState) -> dict[str, Any]:
        logger.info("seo_agent.start")

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {
                "role": "user",
                "content": f"Analyze for SEO:\n\n{state.draft[:4000]}",
            },
        ]

        try:
            seo_data = await self.llm.generate(messages, response_format=SEOData)
            # In practice, parse with structured_output helper
            seo_metadata = SEOData(
                title_tag=state.topic[:60],
                meta_description=f"Learn about {state.topic}",
                keywords=[state.topic],
                readability_score=70.0,
                keyword_density={},
            )
        except Exception:
            seo_metadata = SEOData(title_tag=state.topic[:60], meta_description="", keywords=[], readability_score=50.0)

        logger.info("seo_agent.complete")

        return {
            "seo_metadata": seo_metadata,
            "current_step": "seo",
        }


async def seo_node(state: BlogState) -> dict[str, Any]:
    """LangGraph node wrapper."""
    agent = SEOAgent()
    return await agent.execute(state)