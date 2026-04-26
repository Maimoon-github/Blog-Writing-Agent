"""
Editor Agent — refines the draft for clarity, grammar, and style adherence.
"""

from typing import Any

from blog_agent_system.agents.base import BaseAgent
from blog_agent_system.core.state import BlogState
from blog_agent_system.utils.logging import get_logger

logger = get_logger(__name__)


class EditorAgent(BaseAgent):
    """Line-edits the draft for grammar, clarity, and style guide adherence."""

    def __init__(self):
        super().__init__(role="editor")

    def get_system_prompt(self) -> str:
        return (
            "You are a meticulous professional editor. Your job is to refine "
            "a blog post draft for publication quality.\n\n"
            "EDITING CHECKLIST:\n"
            "1. Fix grammar, spelling, and punctuation errors\n"
            "2. Improve sentence clarity and conciseness\n"
            "3. Ensure consistent tone throughout\n"
            "4. Strengthen transitions between paragraphs and sections\n"
            "5. Remove redundancy and filler words\n"
            "6. Verify logical flow of arguments\n"
            "7. Adhere to the specified style guide\n"
            "8. Ensure headings are parallel in structure\n\n"
            "OUTPUT the complete edited blog post. Do not add commentary."
        )

    async def execute(self, state: BlogState) -> dict[str, Any]:
        logger.info("editor_agent.start", draft_length=len(state.draft.split()))

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Edit the following blog post draft:\n\n"
                    f"STYLE GUIDE: {state.style_guide}\n"
                    f"TONE: {state.tone}\n"
                    f"TARGET AUDIENCE: {state.target_audience}\n\n"
                    f"---DRAFT---\n{state.draft}\n---END DRAFT---\n\n"
                    f"Return the complete edited blog post."
                ),
            },
        ]

        edited = await self._generate(messages)

        logger.info(
            "editor_agent.complete",
            original_words=len(state.draft.split()),
            edited_words=len(edited.split()),
        )

        return {
            "draft": edited,
            "edited_draft": edited,
            "status": "editing",
            "current_step": "edit",
        }


async def editor_node(state: BlogState) -> dict[str, Any]:
    """LangGraph node wrapper for EditorAgent."""
    agent = EditorAgent()
    return await agent.execute(state)
