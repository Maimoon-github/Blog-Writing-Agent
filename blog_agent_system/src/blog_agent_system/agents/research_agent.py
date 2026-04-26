"""
Research Agent — gathers sources, summarizes findings, extracts key facts.
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
            "You are an expert research analyst. Gather comprehensive, accurate information "
            "on the given topic for a blog post.\n\n"
            "INSTRUCTIONS:\n"
            "1. Identify key subtopics and angles\n"
            "2. Find authoritative sources and data points\n"
            "3. Note conflicting viewpoints\n"
            "4. Extract quotable facts and statistics\n"
            "5. Assess source credibility\n\n"
            "OUTPUT: Structured research summary with key findings, sources, and recommended angles."
        )

    async def execute(self, state: BlogState) -> dict[str, Any]:
        logger.info("research_agent.start", topic=state.topic)

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Research this topic:\n\n"
                    f"TOPIC: {state.topic}\n"
                    f"AUDIENCE: {state.target_audience}\n"
                    f"TONE: {state.tone}\n"
                    f"WORD COUNT: {state.word_count_target}\n\n"
                    "Provide comprehensive research findings."
                ),
            },
        ]

        response = await self.llm.generate(messages)  # Uses _generate in real BaseAgent

        # Synthesize into Source objects
        research_findings = [
            Source(
                url="llm-synthesized-research",
                title=f"Research Summary: {state.topic}",
                snippet=response[:500],
                credibility_score=0.75,
            )
        ]

        logger.info("research_agent.complete", findings=len(research_findings))

        return {
            "research_findings": research_findings,
            "status": "researching",
            "current_step": "research",
        }


async def research_node(state: BlogState) -> dict[str, Any]:
    """LangGraph node wrapper."""
    agent = ResearchAgent()
    return await agent.execute(state)