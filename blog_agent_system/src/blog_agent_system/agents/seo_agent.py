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
            "You are an SEO specialist. Analyze the blog post and provide "
            "optimization recommendations.\n\n"
            "You must respond with valid JSON containing:\n"
            '- "title_tag": SEO-optimized title (50-60 chars)\n'
            '- "meta_description": Compelling meta description (150-160 chars)\n'
            '- "keywords": List of 5-8 target keywords\n'
            '- "readability_score": Flesch-Kincaid readability score (0-100)\n'
            '- "keyword_density": Dict of keyword to density percentage\n\n'
            "Respond ONLY with valid JSON."
        )

    async def execute(self, state: BlogState) -> dict[str, Any]:
        logger.info("seo_agent.start", draft_words=len(state.draft.split()))

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Analyze the following blog post for SEO:\n\n"
                    f"TOPIC: {state.topic}\n"
                    f"TARGET AUDIENCE: {state.target_audience}\n\n"
                    f"---BLOG POST---\n{state.draft[:4000]}\n---END---\n\n"
                    f"Provide SEO analysis as JSON."
                ),
            },
        ]

        response = await self._generate(messages, response_format=SEOData)

        # Try to parse structured output
        try:
            import json

            # Strip markdown fences if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            data = json.loads(cleaned)
            seo_metadata = SEOData(
                title_tag=data.get("title_tag", state.topic[:60]),
                meta_description=data.get("meta_description", ""),
                keywords=data.get("keywords", []),
                readability_score=float(data.get("readability_score", 50.0)),
                keyword_density=data.get("keyword_density", {}),
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("seo_agent.parse_fallback", error=str(e))
            seo_metadata = SEOData(
                title_tag=state.topic[:60],
                meta_description=f"Read about {state.topic}",
                keywords=[state.topic],
                readability_score=50.0,
                keyword_density={},
            )

        logger.info("seo_agent.complete", keywords=seo_metadata.keywords)

        return {
            "seo_metadata": seo_metadata,
            "current_step": "seo",
        }


async def seo_node(state: BlogState) -> dict[str, Any]:
    """LangGraph node wrapper for SEOAgent."""
    agent = SEOAgent()
    return await agent.execute(state)
