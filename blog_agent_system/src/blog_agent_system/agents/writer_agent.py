"""
Writer Agent — generates prose for each blog section.

Writes section-by-section, maintaining voice consistency and incorporating
research sources. Handles revision feedback from the quality gate.
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
            "You are an expert blog writer. You write engaging, well-researched "
            "content that resonates with the target audience.\n\n"
            "WRITING PRINCIPLES:\n"
            "1. Write original, engaging prose — not a list of facts\n"
            "2. Incorporate sources naturally (don't just cite, weave them in)\n"
            "3. Use clear transitions between paragraphs\n"
            "4. Match the specified tone consistently\n"
            "5. Follow the style guide conventions\n"
            "6. Hit the target word count for each section\n"
            "7. End each section with a bridge to the next\n"
        )

    async def execute(self, state: BlogState) -> dict[str, Any]:
        is_revision = state.revision_count > 0
        logger.info(
            "writer_agent.start",
            topic=state.topic,
            section_count=len(state.outline),
            is_revision=is_revision,
            revision_count=state.revision_count,
        )

        # Build research context
        research_context = "\n".join(
            f"- {s.title}: {s.snippet}" for s in state.research_findings
        )

        draft_sections: list[Section] = []
        previous_summary = ""

        for i, section in enumerate(state.outline):
            # Build section-specific prompt
            revision_instruction = ""
            if is_revision and state.revision_feedback:
                revision_instruction = (
                    f"\n\nREVISION FEEDBACK (address these issues):\n"
                    f"{state.revision_feedback}\n"
                )

            messages = [
                {"role": "system", "content": self.get_system_prompt()},
                {
                    "role": "user",
                    "content": (
                        f"Write section {i + 1} of {len(state.outline)} for the blog post.\n\n"
                        f"TOPIC: {state.topic}\n"
                        f"SECTION HEADING: {section.heading}\n"
                        f"TARGET AUDIENCE: {state.target_audience}\n"
                        f"TONE: {state.tone}\n"
                        f"TARGET WORD COUNT: {section.word_count}\n"
                        f"STYLE GUIDE: {state.style_guide}\n\n"
                        f"KEY POINTS TO COVER:\n"
                        + "\n".join(f"- {kp}" for kp in section.key_points)
                        + f"\n\nRESEARCH CONTEXT:\n{research_context}\n\n"
                        f"PREVIOUS SECTION SUMMARY:\n{previous_summary or 'This is the first section.'}\n"
                        f"{revision_instruction}\n"
                        f"OUTPUT ONLY THE SECTION CONTENT. No meta-commentary."
                    ),
                },
            ]

            content = await self._generate(messages)

            written_section = Section(
                heading=section.heading,
                content=content,
                key_points=section.key_points,
                word_count=len(content.split()),
                sources=[s.url for s in state.research_findings[:3]],
            )
            draft_sections.append(written_section)

            # Summary for next section's context
            previous_summary = content[:200] + "..." if len(content) > 200 else content

            logger.debug(
                "writer_agent.section_complete",
                section=i + 1,
                heading=section.heading,
                words=written_section.word_count,
            )

        # Assemble full draft
        full_draft = "\n\n".join(
            f"## {s.heading}\n\n{s.content}" for s in draft_sections
        )

        total_words = sum(s.word_count for s in draft_sections)
        logger.info("writer_agent.complete", total_words=total_words, section_count=len(draft_sections))

        return {
            "draft_sections": draft_sections,
            "draft": full_draft,
            "status": "drafting",
            "current_step": "write",
        }


async def writer_node(state: BlogState) -> dict[str, Any]:
    """LangGraph node wrapper for WriterAgent."""
    agent = WriterAgent()
    return await agent.execute(state)
