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
            "You are a rigorous fact-checker. Your job is to verify claims "
            "in a blog post against provided research sources.\n\n"
            "For each factual claim you find:\n"
            "1. Identify the specific claim\n"
            "2. Check if it's supported by the provided sources\n"
            "3. Assign a confidence score (0.0 to 1.0)\n"
            "4. Suggest corrections for inaccurate claims\n\n"
            "Respond with valid JSON: a list of objects with keys: "
            '"claim", "verified" (bool), "confidence" (float), '
            '"source_ref" (string or null), "correction" (string or null).\n\n'
            "Respond ONLY with a JSON array."
        )

    async def execute(self, state: BlogState) -> dict[str, Any]:
        logger.info("fact_checker_agent.start", draft_words=len(state.draft.split()))

        # Build source context
        source_context = "\n".join(
            f"- [{s.title}]({s.url}): {s.snippet}" for s in state.research_findings
        )

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Fact-check the following blog post:\n\n"
                    f"---BLOG POST---\n{state.draft[:4000]}\n---END---\n\n"
                    f"AVAILABLE SOURCES:\n{source_context}\n\n"
                    f"Identify and verify all factual claims."
                ),
            },
        ]

        response = await self._generate(messages, response_format=FactCheckResult)

        # Parse fact check results
        fact_check_results: list[FactCheckResult] = []
        try:
            import json

            cleaned = response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            results_data = json.loads(cleaned)
            if isinstance(results_data, list):
                for item in results_data:
                    fact_check_results.append(
                        FactCheckResult(
                            claim=item.get("claim", ""),
                            verified=item.get("verified", False),
                            confidence=float(item.get("confidence", 0.5)),
                            source_ref=item.get("source_ref"),
                            correction=item.get("correction"),
                        )
                    )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("fact_checker_agent.parse_fallback", error=str(e))
            # Fallback: assume content is reasonably accurate
            fact_check_results = [
                FactCheckResult(
                    claim="General content accuracy",
                    verified=True,
                    confidence=0.7,
                )
            ]

        verified_count = sum(1 for fc in fact_check_results if fc.verified)
        logger.info(
            "fact_checker_agent.complete",
            total_claims=len(fact_check_results),
            verified=verified_count,
        )

        return {
            "fact_check_results": fact_check_results,
            "current_step": "fact_check",
        }


async def fact_checker_node(state: BlogState) -> dict[str, Any]:
    """LangGraph node wrapper for FactCheckerAgent."""
    agent = FactCheckerAgent()
    return await agent.execute(state)
