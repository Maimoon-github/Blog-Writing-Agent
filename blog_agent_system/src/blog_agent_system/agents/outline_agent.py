"""
Outline Agent — transforms research into a structured blog outline.
"""

from typing import Any

from blog_agent_system.agents.base import BaseAgent
from blog_agent_system.core.state import BlogState, Section
from blog_agent_system.llm.structured_output import generate_structured, OutlineOutput
from blog_agent_system.utils.logging import get_logger

logger = get_logger(__name__)


class OutlineAgent(BaseAgent):
    """Designs the structural outline for the blog post."""

    def __init__(self):
        super().__init__(role="outline")

    def get_system_prompt(self) -> str:
        return (
            "You are an expert blog structure designer. Create a compelling, logical outline.\n\n"
            "INSTRUCTIONS:\n"
            "1. Create a compelling title\n"
            "2. Design 4-6 sections with clear H2 headings\n"
            "3. Include 2-4 key points per section\n"
            "4. Distribute word count logically\n"
            "5. Ensure narrative flow\n"
            "6. Include introduction and conclusion\n\n"
            "OUTPUT valid JSON matching the OutlineOutput schema."
        )

    async def execute(self, state: BlogState) -> dict[str, Any]:
        logger.info("outline_agent.start", topic=state.topic)

        research_context = "\n".join(
            f"- {s.title}: {s.snippet}" for s in state.research_findings
        )

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Create outline for:\n\n"
                    f"TOPIC: {state.topic}\n"
                    f"AUDIENCE: {state.target_audience}\n"
                    f"TONE: {state.tone}\n"
                    f"WORD COUNT: {state.word_count_target}\n\n"
                    f"RESEARCH:\n{research_context}"
                ),
            },
        ]

        try:
            outline_data = await generate_structured(self.llm, messages, OutlineOutput)
            sections = [
                Section(
                    heading=s.get("heading", f"Section {i+1}"),
                    key_points=s.get("key_points", []),
                    word_count=s.get("word_count", state.word_count_target // 5),
                )
                for i, s in enumerate(outline_data.sections)
            ]
        except Exception as e:
            logger.warning("outline_agent.fallback", error=str(e))
            sections = [Section(heading="Introduction", word_count=300)]

        logger.info("outline_agent.complete", sections=len(sections))

        return {
            "outline": sections,
            "status": "outlining",
            "current_step": "outline",
        }


async def outline_node(state: BlogState) -> dict[str, Any]:
    """LangGraph node wrapper."""
    agent = OutlineAgent()
    return await agent.execute(state)