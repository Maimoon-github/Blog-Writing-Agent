"""
Outline Agent — transforms research into a structured blog outline.

Produces hierarchical headings, key points per section, and word count estimates.
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
            "You are an expert blog structure designer. Your job is to create "
            "a compelling, well-organized outline for a blog post.\n\n"
            "INSTRUCTIONS:\n"
            "1. Create a compelling title\n"
            "2. Design 4-6 sections with clear H2 headings\n"
            "3. Include 2-4 key points per section\n"
            "4. Distribute the word count logically across sections\n"
            "5. Ensure narrative flow — each section transitions to the next\n"
            "6. Include an engaging introduction and a strong conclusion\n\n"
            "OUTPUT: Respond with valid JSON matching the provided schema."
        )

    async def execute(self, state: BlogState) -> dict[str, Any]:
        logger.info("outline_agent.start", topic=state.topic)

        # Build research context
        research_context = ""
        for source in state.research_findings:
            research_context += f"- {source.title}: {source.snippet}\n"

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Create a blog post outline for:\n\n"
                    f"TOPIC: {state.topic}\n"
                    f"TARGET AUDIENCE: {state.target_audience}\n"
                    f"TONE: {state.tone}\n"
                    f"TOTAL WORD COUNT: {state.word_count_target}\n"
                    f"STYLE GUIDE: {state.style_guide}\n\n"
                    f"RESEARCH FINDINGS:\n{research_context}\n\n"
                    f"Create a structured outline with title, sections (each with "
                    f"heading, key_points list, and estimated word count), "
                    f"and estimated_read_time in minutes."
                ),
            },
        ]

        try:
            outline_data = await generate_structured(self.llm, messages, OutlineOutput)

            sections = [
                Section(
                    heading=s.get("heading", f"Section {i+1}"),
                    key_points=s.get("key_points", []),
                    word_count=s.get("word_count", state.word_count_target // len(outline_data.sections)),
                )
                for i, s in enumerate(outline_data.sections)
            ]

            logger.info("outline_agent.complete", section_count=len(sections), title=outline_data.title)

        except Exception as e:
            logger.warning("outline_agent.structured_fallback", error=str(e))
            # Fallback: generate plain text and create basic sections
            response = await self._generate(messages)
            sections = [
                Section(heading="Introduction", key_points=[], word_count=300),
                Section(heading="Main Discussion", key_points=[], word_count=state.word_count_target - 500),
                Section(heading="Conclusion", key_points=[], word_count=200),
            ]

        return {
            "outline": sections,
            "status": "outlining",
            "current_step": "outline",
        }


async def outline_node(state: BlogState) -> dict[str, Any]:
    """LangGraph node wrapper for OutlineAgent."""
    agent = OutlineAgent()
    return await agent.execute(state)
