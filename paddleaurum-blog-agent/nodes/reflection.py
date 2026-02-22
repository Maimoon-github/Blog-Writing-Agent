import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

from graph.state import AgentState, SEOIssue

logger = logging.getLogger(__name__)

_REFLECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a senior content editor and SEO specialist reviewing a pickleball article draft. "
        "Your job is to produce clear, actionable revision instructions for the writer. "
        "Be specific and concrete — reference exact sections, headings, or sentences where possible. "
        "Output a numbered list of targeted instructions only. No preamble, no summary."
    )),
    ("human", (
        "Primary keyword: {primary_kw}\n"
        "Word count goal: {word_count_goal}\n"
        "Current SEO score: {seo_score}/100\n\n"
        "SEO issues to fix:\n{issues_text}\n\n"
        "Current draft (first 800 words shown):\n{draft_preview}\n\n"
        "Write specific revision instructions that will fix every SEO issue above "
        "and bring the article to a score of 85 or higher. "
        "Reference the exact problem and the exact fix for each instruction. "
        "Output numbered instructions only."
    )),
])


def _format_issues(issues: List[SEOIssue]) -> str:
    if not issues:
        return "No specific issues recorded."
    lines = []
    for i, issue in enumerate(issues, 1):
        lines.append(
            f"{i}. [{issue['severity'].upper()}] {issue['field']}: {issue['message']}\n"
            f"   Fix: {issue['suggestion']}"
        )
    return "\n".join(lines)


async def reflection_node(state: AgentState, *, llm: Ollama) -> dict:
    try:
        draft: str = state.get("draft_article") or ""
        issues: List[SEOIssue] = state.get("seo_issues") or []
        seo_score: int = state.get("seo_score") or 0
        keyword_map = state.get("keyword_map") or {}
        primary_kw: str = keyword_map.get("primary", state.get("topic", ""))
        word_count_goal: int = state.get("word_count_goal", 1500)
        current_iteration: int = state.get("revision_iteration", 0)

        draft_preview = " ".join(draft.split()[:800])

        chain = _REFLECTION_PROMPT | llm | StrOutputParser()
        revision_instructions: str = await chain.ainvoke({
            "primary_kw":      primary_kw,
            "word_count_goal": word_count_goal,
            "seo_score":       seo_score,
            "issues_text":     _format_issues(issues),
            "draft_preview":   draft_preview,
        })

        # Convert the numbered instruction block to a list of strings
        instruction_lines: List[str] = [
            line.strip()
            for line in revision_instructions.strip().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

        new_iteration = current_iteration + 1

        logger.info(
            "Reflection node: %d instructions generated, advancing to iteration %d",
            len(instruction_lines), new_iteration,
        )

        return {
            "seo_suggestions":   instruction_lines,
            "revision_iteration": new_iteration,
            "error":             None,
            "error_node":        None,
        }

    except Exception as exc:
        logger.exception("Reflection node failed.")
        return {"error": str(exc), "error_node": "reflection"}