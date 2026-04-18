# """
# agents/router.py

# Entry point intent classifier for the Autonomous AI Blog Generation System.

# Determines whether a topic requires live web research or can be handled
# from LLM knowledge alone. Uses deterministic classification (temperature=0)
# and fails safely (defaults to True) on any error.

# Stack: langchain_ollama, langchain_core
# """

# import logging
# from pathlib import Path
# from typing import Dict

# from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama import ChatOllama

# from config import OLLAMA_BASE_URL, OLLAMA_MODEL, TEMPERATURE

# # ---------------------------------------------------------------------------
# # Logging
# # ---------------------------------------------------------------------------
# logger = logging.getLogger(__name__)
# if not logging.getLogger().handlers:
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
#         datefmt="%H:%M:%S",
#     )


# # ---------------------------------------------------------------------------
# # Default Prompt (used if file missing)
# # ---------------------------------------------------------------------------
# DEFAULT_ROUTER_PROMPT = """You are an intent classifier for a blog generation system.

# Your task: Determine if the given topic requires live web research or can be answered from general knowledge.

# Topic: {topic}

# Rules:
# - Return ONLY the word "true" if the topic contains time-sensitive keywords (latest, 2025, new, recent, current, today, breaking, news, trends) OR if it involves rapidly evolving technical information (AI models, software versions, current events).
# - Return ONLY the word "false" if the topic is a timeless factual subject (history, science concepts, established principles) that would not benefit from web search.
# - Do not explain. Do not add punctuation. Return ONLY "true" or "false".

# Decision:"""


# # ---------------------------------------------------------------------------
# # Core Functions
# # ---------------------------------------------------------------------------

# def load_prompt(prompt_path: str = "prompts/router_prompt.txt") -> str:
#     """
#     Load router prompt from file, or return default if file missing.

#     Args:
#         prompt_path: Path to custom prompt file

#     Returns:
#         Prompt template string with {topic} placeholder
#     """
#     try:
#         path = Path(prompt_path)
#         if path.exists():
#             content = path.read_text(encoding="utf-8")
#             logger.info("[router] Loaded custom prompt from %s", prompt_path)
#             return content
#     except Exception as exc:
#         logger.warning("[router] Could not load %s: %s", prompt_path, exc)

#     logger.info("[router] Using default classification prompt")
#     return DEFAULT_ROUTER_PROMPT


# def create_router_chain(prompt_text: str):
#     """
#     Create deterministic LLM chain for intent classification.

#     Args:
#         prompt_text: Prompt template string with {topic} variable

#     Returns:
#         Runnable chain (prompt | llm)
#     """
#     llm = ChatOllama(
#         model=OLLAMA_MODEL,
#         base_url=OLLAMA_BASE_URL,
#         temperature=0.0,  # Deterministic classification
#         # No streaming for classification — we need full response
#     )

#     prompt = ChatPromptTemplate.from_template(prompt_text)
#     chain = prompt | llm

#     logger.debug("[router] Created chain with model=%s, temp=0.0", OLLAMA_MODEL)
#     return chain


# def router_node(state: Dict) -> Dict[str, bool]:
#     """
#     Classify whether the topic requires web research.

#     Args:
#         state: Dict containing "topic" key

#     Returns:
#         Dict with "research_required": bool
#         Always returns valid dict, never raises.
#         On any error, defaults to True (safe fallback).
#     """
#     topic = state.get("topic", "") if isinstance(state, dict) else ""

#     if not topic or not str(topic).strip():
#         logger.warning("[router] Empty topic received, defaulting to research=True")
#         return {"research_required": True}

#     topic_str = str(topic).strip()

#     # Load prompt and create chain
#     prompt_text = load_prompt()
#     chain = create_router_chain(prompt_text)

#     try:
#         # Invoke chain
#         response = chain.invoke({"topic": topic_str})
#         content = response.content if hasattr(response, "content") else str(response)

#         # Parse response — look for "true" or "false" in the content
#         cleaned = content.strip().lower()

#         # Handle various formats: "true", "true.", "true", " true ", etc.
#         if "true" in cleaned and "false" not in cleaned:
#             decision = True
#         elif "false" in cleaned and "true" not in cleaned:
#             decision = False
#         else:
#             # Ambiguous or unexpected response — default to safe fallback
#             logger.warning(
#                 "[router] Ambiguous response for topic '%s...': '%s'. Defaulting to True.",
#                 topic_str[:50],
#                 cleaned[:100]
#             )
#             decision = True

#         logger.info(
#             "[router] Classified '%s...' as research_required=%s",
#             topic_str[:50],
#             decision
#         )
#         return {"research_required": decision}

#     except Exception as exc:
#         # Catch-all: connection errors, timeouts, parsing errors, etc.
#         logger.error(
#             "[router] Classification failed for '%s...': %s. Defaulting to True.",
#             topic_str[:50],
#             exc
#         )
#         return {"research_required": True}


# # ---------------------------------------------------------------------------
# # Self-test
# # ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     test_cases = [
#         # (topic, expected_research_likely)
#         ("latest AI trends 2025", True),   # time-sensitive
#         ("breaking news in quantum computing", True),  # current events
#         ("new features in Python 3.12", True),  # recent software
#         ("what is photosynthesis", False),  # timeless science
#         ("history of the Roman Empire", False),  # historical
#         ("how does neural backpropagation work", False),  # established concept
#         ("", True),  # empty — safe fallback
#     ]

#     print("=== ROUTER SELF-TEST ===\n")

#     for topic, expected in test_cases:
#         result = router_node({"topic": topic})
#         actual = result.get("research_required", True)
#         status = "✓" if actual == expected else "?"
#         print(f"{status} topic='{topic[:40]}...' -> research_required={actual} (expected ~{expected})")

#     print("\nNote: Exact classification depends on LLM judgment.")
#     print("Key requirement: Empty/failure cases must return True (safe).")
























# /*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-/*-












"""agents/router.py — FIXED for Windows path + template escaping"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict

# Add parent to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from config import OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "router_prompt.txt"

def router_node(state: Dict) -> Dict[str, bool]:
    """Router node — deterministic classification with safe fallback."""
    topic = state.get("topic", "").strip()
    if not topic:
        return {"research_required": True}

    try:
        # Read and manually substitute — NEVER use ChatPromptTemplate with JSON
        prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
        final_prompt = prompt_template.format(topic=topic)

        llm = ChatOllama(
            model=OLLAMA_MODEL,  # Should be "llama3.2:3b" in config.py
            base_url=OLLAMA_BASE_URL,
            temperature=0.0,
        )

        response = llm.invoke([HumanMessage(content=final_prompt)])
        content = response.content.strip()

        # Extract JSON robustly
        research_required = True  # safe default
        if "{" in content:
            try:
                json_str = content[content.find("{"): content.rfind("}") + 1]
                data = json.loads(json_str)
                research_required = bool(data.get("research_required", True))
            except json.JSONDecodeError:
                # Fallback to keyword matching
                research_required = "true" in content.lower() and "false" not in content.lower()
        else:
            research_required = "true" in content.lower()

        logger.info(f"[router] Classified '{topic[:60]}...' → research_required={research_required}")
        return {"research_required": research_required}

    except Exception as e:
        logger.error(f"[router] Failed: {e} → defaulting to research=True")
        return {"research_required": True}


# Self-test entry point
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_cases = [
        {"topic": "AI trends 2026"},
        {"topic": "History of the Roman Empire"},
        {"topic": ""},
    ]
    for case in test_cases:
        result = router_node(case)
        print(f"Input: {case['topic'][:40]!r:40} → {result}")