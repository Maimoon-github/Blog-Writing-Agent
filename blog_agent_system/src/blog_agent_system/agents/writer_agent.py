"""
Writer Agent — generates prose section-by-section.
"""

from typing import Any

from blog_agent_system.agents.base import BaseAgent
from blog_agent_system.core.state import BlogState, Section
from blog_agent_system.utils.logging import get_logger

logger = get_logger(__name__)


class WriterAgent(BaseAgent):
    """Generates blog post content section by section."""

    def __init__(self):
        super().__init__(role="writer")

    def get_system_prompt(self) -> str:
        return (
            "You are an expert blog writer. Write engaging, original prose that incorporates "
            "research naturally and maintains consistent tone.\n\n"
            "INSTRUCTIONS:\n"
            "1. Write engaging prose\n"
            "2. Incorporate sources naturally\n"
            "3. Use smooth transitions\n"
            "4. Hit target word count per section\n"
            "5. Follow style guide"
        )

    async def execute(self, state: BlogState) -> dict[str, Any]:
        is_revision = state.revision_count > 0
        logger.info("writer_agent.start", topic=state.topic, revision=is_revision)

        research_context = "\n".join(f"- {s.title}: {s.snippet}" for s in state.research_findings)

        draft_sections: list[Section] = []
        previous_summary = ""

        for i, section in enumerate(state.outline):
            revision_instruction = (
                f"\n\nREVISION FEEDBACK:\n{state.revision_feedback}" if is_revision and state.revision_feedback else ""
            )

            messages = [
                {"role": "system", "content": self.get_system_prompt()},
                {
                    "role": "user",
                    "content": (
                        f"Write section {i+1}/{len(state.outline)}:\n\n"
                        f"HEADING: {section.heading}\n"
                        f"WORD COUNT: {section.word_count}\n"
                        f"RESEARCH:\n{research_context}\n"
                        f"PREVIOUS: {previous_summary}\n"
                        f"{revision_instruction}\n\n"
                        "OUTPUT ONLY THE SECTION CONTENT."
                    ),
                },
            ]

            content = await self.llm.generate(messages)

            written = Section(
                heading=section.heading,
                content=content,
                word_count=len(content.split()),
                sources=[s.url for s in state.research_findings[:3]],
            )
            draft_sections.append(written)
            previous_summary = content[:200] + "..." if len(content) > 200 else content

        full_draft = "\n\n".join(f"## {s.heading}\n\n{s.content}" for s in draft_sections)

        logger.info("writer_agent.complete", total_words=sum(s.word_count for s in draft_sections))

        return {
            "draft_sections": draft_sections,
            "draft": full_draft,
            "status": "drafting",
            "current_step": "write",
        }


async def writer_node(state: BlogState) -> dict[str, Any]:
    """LangGraph node wrapper."""
    agent = WriterAgent()
    return await agent.execute(state)