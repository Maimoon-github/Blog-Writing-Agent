"""
Research Agent — gathers sources, summarizes findings, extracts key facts.

Uses web search and RAG retrieval to build a knowledge base for the blog topic.
"""

from typing import Any

from blog_agent_system.agents.base import BaseAgent
from blog_agent_system.core.state import BlogState, Source
from blog_agent_system.utils.logging import get_logger

logger = get_logger(__name__)


class ResearchAgent(BaseAgent):
    """Gathers and synthesizes research for the blog topic."""

    def __init__(self):
        super().__init__(role="research")

    def get_system_prompt(self) -> str:
        return (
            "You are an expert research analyst. Your job is to gather comprehensive, "
            "accurate information on the given topic for a blog post.\n\n"
            "INSTRUCTIONS:\n"
            "1. Identify the key subtopics and angles to cover\n"
            "2. Find authoritative sources and data points\n"
            "3. Note any conflicting viewpoints or debates\n"
            "4. Extract quotable facts and statistics\n"
            "5. Assess source credibility\n\n"
            "OUTPUT: Provide a structured research summary with key findings, "
            "sources, and recommended angles for the blog post."
        )

    async def execute(self, state: BlogState) -> dict[str, Any]:
        logger.info("research_agent.start", topic=state.topic, audience=state.target_audience)

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Research the following topic for a blog post:\n\n"
                    f"TOPIC: {state.topic}\n"
                    f"TARGET AUDIENCE: {state.target_audience}\n"
                    f"TONE: {state.tone}\n"
                    f"WORD COUNT TARGET: {state.word_count_target}\n\n"
                    f"Provide comprehensive research findings including key facts, "
                    f"statistics, expert opinions, and recommended angles."
                ),
            },
        ]

        response = await self._generate(messages)

        # Parse response into Source objects
        # For now, create a single synthesized source from the LLM research
        research_findings = [
            Source(
                url="llm-synthesized",
                title=f"Research Summary: {state.topic}",
                snippet=response[:500],
                credibility_score=0.7,
            )
        ]

        logger.info("research_agent.complete", finding_count=len(research_findings))

        return {
            "research_findings": research_findings,
            "status": "researching",
            "current_step": "research",
        }


async def research_node(state: BlogState) -> dict[str, Any]:
    """LangGraph node wrapper for ResearchAgent."""
    agent = ResearchAgent()
    return await agent.execute(state)
