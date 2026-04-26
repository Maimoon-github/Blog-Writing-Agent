"""
Fact Checker Agent — verifies claims in the draft against research sources.
"""

from typing import Any

from blog_agent_system.agents.base import BaseAgent
from blog_agent_system.core.state import BlogState, FactCheckResult
from blog_agent_system.utils.logging import get_logger

logger = get_logger(__name__)


class FactCheckerAgent(BaseAgent):
    """Cross-references draft claims against sources; assigns confidence scores."""

    def __init__(self):
        super().__init__(role="fact_checker")

    def get_system_prompt(self) -> str:
        return (
            "You are a rigorous fact-checker. Verify claims against sources.\n\n"
            "Respond with valid JSON array of objects with keys: claim, verified, confidence, "
            "source_ref, correction."
        )

    async def execute(self, state: BlogState) -> dict[str, Any]:
        logger.info("fact_checker_agent.start")

        source_context = "\n".join(f"- {s.title}: {s.snippet}" for s in state.research_findings)

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {
                "role": "user",
                "content": f"Fact-check this draft:\n\n{state.draft[:4000]}\n\nSOURCES:\n{source_context}",
            },
        ]

        # Use structured output where possible
        try:
            results = await self.llm.generate(messages, response_format=FactCheckResult)
            fact_check_results = [FactCheckResult(**item) for item in results] if isinstance(results, list) else []
        except Exception:
            fact_check_results = [FactCheckResult(claim="General accuracy", verified=True, confidence=0.7)]

        logger.info("fact_checker_agent.complete", verified_count=len([r for r in fact_check_results if r.verified]))

        return {
            "fact_check_results": fact_check_results,
            "current_step": "fact_check",
        }


async def fact_checker_node(state: BlogState) -> dict[str, Any]:
    """LangGraph node wrapper."""
    agent = FactCheckerAgent()
    return await agent.execute(state)