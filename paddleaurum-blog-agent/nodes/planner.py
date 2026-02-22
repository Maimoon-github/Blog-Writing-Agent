import json
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.llms import Ollama

from graph.state import AgentState, IntentType

logger = logging.getLogger(__name__)

_PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an editorial planning agent for paddleaurum.com, a pickleball coaching blog. "
        "Given a blog topic, produce a structured JSON planning document. "
        "Return ONLY valid JSON — no markdown fences, no extra text."
    )),
    ("human", (
        "Topic: {topic}\n"
        "Target keyword (may be empty): {target_keyword}\n"
        "Word count goal: {word_count_goal}\n\n"
        "Respond with exactly this JSON structure:\n"
        "{{\n"
        '  "needs_research": true,\n'
        '  "intent_type": "informational|commercial|navigational",\n'
        '  "sub_queries": ["query1", "query2", "query3", "query4"],\n'
        '  "task_plan": {{\n'
        '    "audience": "...",\n'
        '    "angle": "...",\n'
        '    "key_sections": ["...", "..."],\n'
        '    "tone_notes": "..."\n'
        "  }}\n"
        "}}\n\n"
        "Rules:\n"
        "- needs_research=false only for timeless conceptual topics needing no current data.\n"
        "- sub_queries: 3-5 specific search strings that together cover the topic thoroughly.\n"
        "- intent_type: commercial if query implies buying/comparing products; "
        "navigational if brand-specific; otherwise informational.\n"
        "- key_sections: 4-6 major H2 headings the article should contain."
    )),
])


async def planner_node(state: AgentState, *, llm: Ollama) -> dict:
    try:
        chain = _PLANNER_PROMPT | llm | JsonOutputParser()
        result: dict = await chain.ainvoke({
            "topic":           state["topic"],
            "target_keyword":  state.get("target_keyword") or "",
            "word_count_goal": state.get("word_count_goal", 1500),
        })

        needs_research: bool = bool(result.get("needs_research", True))
        sub_queries: list[str] = [
            q.strip() for q in result.get("sub_queries", []) if isinstance(q, str) and q.strip()
        ]
        if needs_research and not sub_queries:
            sub_queries = [state["topic"]]

        raw_intent = result.get("intent_type", "informational").lower().strip()
        intent_map = {i.value: i for i in IntentType}
        intent_type = intent_map.get(raw_intent, IntentType.INFORMATIONAL)

        task_plan: dict = result.get("task_plan", {})

        logger.info(
            "Planner: intent=%s needs_research=%s queries=%d",
            intent_type, needs_research, len(sub_queries),
        )

        return {
            "task_plan":      task_plan,
            "sub_queries":    sub_queries,
            "needs_research": needs_research,
            "intent_type":    intent_type,
            "error":          None,
            "error_node":     None,
        }

    except Exception as exc:
        logger.exception("Planner node failed.")
        return {"error": str(exc), "error_node": "planner"}