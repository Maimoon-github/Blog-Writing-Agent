"""
Editor Agent — refines the draft for clarity, grammar, and style.
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
            "You are a meticulous professional editor. Refine the draft for publication quality.\n\n"
            "CHECKLIST:\n"
            "1. Fix grammar and punctuation\n"
            "2. Improve clarity and conciseness\n"
            "3. Ensure consistent tone\n"
            "4. Strengthen transitions\n"
            "5. Remove redundancy\n"
            "6. Follow style guide"
        )

    async def execute(self, state: BlogState) -> dict[str, Any]:
        logger.info("editor_agent.start")

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Edit this draft:\n\n"
                    f"STYLE: {state.style_guide}\n"
                    f"TONE: {state.tone}\n\n"
                    f"---DRAFT---\n{state.draft}\n---END---\n\n"
                    "Return the complete edited blog post."
                ),
            },
        ]

        edited = await self.llm.generate(messages)

        logger.info("editor_agent.complete")

        return {
            "draft": edited,
            "edited_draft": edited,
            "status": "editing",
            "current_step": "edit",
        }


async def editor_node(state: BlogState) -> dict[str, Any]:
    """LangGraph node wrapper."""
    agent = EditorAgent()
    return await agent.execute(state)