# #!/usr/bin/env python3
# """
# BWA Blog Writing Agent – Production-Grade Implementation
# Faithful to the Visual Representation specification (diagrams 2.1 – 2.4).

# Workflow (§2.1 Flowchart / §2.4 State Diagram):
#   START
#     → router_node          (Ollama)
#     → [research_node]      (DuckDuckGo + Ollama synthesis)  ← conditional
#     → orchestrator_node    (Ollama)
#     → [worker_node ×N]     (Ollama, parallel fan-out)
#     ┌─ Reducer Subgraph ──────────────────────────────────────────────────┐
#     │  merge_content → decide_images → generate_images → place_images    │
#     └─────────────────────────────────────────────────────────────────────┘
#   END

# Image generation uses Hugging Face Stable Diffusion (§2.1, §2.2).
# Research uses DuckDuckGo (WRE) then Ollama synthesis → EvidencePack (§2.2 Sequence).
# No SEO Optimizer node (not present in any diagram).
# """

# import asyncio
# import logging
# import operator
# import re
# from io import BytesIO
# from pathlib import Path
# from typing import Annotated, List, Literal, Optional, TypedDict

# from dotenv import load_dotenv
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_ollama import ChatOllama
# from langgraph.constants import END, START
# from langgraph.graph import StateGraph
# from langgraph.types import Send
# from pydantic import BaseModel, Field

# # ── Hugging Face image generation (§2.1 / §2.2) ─────────────────────────────
# try:
#     from huggingface_hub import InferenceClient  # type: ignore
#     from PIL import Image  # noqa: F401  (imported to validate bytes in tests)
#     HF_AVAILABLE = True
# except ImportError:
#     HF_AVAILABLE = False
#     logging.warning(
#         "huggingface_hub / Pillow not installed – image generation will produce "
#         "error blocks. Install with: pip install huggingface-hub pillow"
#     )

# # ── Web Research Engine (DuckDuckGo, §2.1 / §2.2) ───────────────────────────
# try:
#     from web_research_engine import ResearchEngine, ResearchReport  # type: ignore
#     WRE_AVAILABLE = True
# except ImportError:
#     WRE_AVAILABLE = False
#     logging.warning(
#         "Web Research Engine not found – research node will return empty results."
#     )

# load_dotenv()

# # ======================================================================
# # Logging
# # ======================================================================
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     handlers=[logging.StreamHandler()],
# )
# logger = logging.getLogger(__name__)

# # ======================================================================
# # WRE configuration constants
# # ======================================================================
# WRE_MAX_PAGES_PER_QUERY: int = 2
# WRE_TIMEOUT: int = 20
# WRE_RATE_LIMIT: float = 1.0
# WRE_MIN_WORDS: int = 100
# WRE_RESEARCH_TIMEOUT: int = 90   # hard async timeout for the entire search phase

# # ======================================================================
# # 1. Pydantic Schemas  (§2.3 Class Diagram)
# # ======================================================================

# class Task(BaseModel):
#     id: int
#     title: str
#     goal: str = Field(..., description="One sentence describing the reader's takeaway.")
#     bullets: List[str] = Field(..., min_length=3, max_length=6)
#     target_words: int = Field(..., description="Target words (120-550).")
#     tags: List[str] = Field(default_factory=list)
#     requires_research: bool = False
#     requires_citations: bool = False
#     requires_code: bool = False


# class Plan(BaseModel):
#     blog_title: str
#     audience: str
#     tone: str
#     blog_kind: Literal[
#         "explainer", "tutorial", "news_roundup", "comparison", "system_design"
#     ] = "explainer"
#     constraints: List[str] = Field(default_factory=list)
#     tasks: List[Task]


# class EvidenceItem(BaseModel):
#     title: str
#     url: str
#     published_at: Optional[str] = None   # ISO "YYYY-MM-DD"
#     snippet: Optional[str] = None
#     source: Optional[str] = None


# class EvidencePack(BaseModel):
#     """Produced by the Ollama synthesis step inside research_node (§2.2 Sequence)."""
#     evidence: List[EvidenceItem] = Field(default_factory=list)


# class RouterDecision(BaseModel):
#     needs_research: bool
#     mode: Literal["closed_book", "hybrid", "open_book"]
#     reason: str
#     queries: List[str] = Field(default_factory=list)
#     max_results_per_query: int = Field(3)


# class ImageSpec(BaseModel):
#     placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
#     filename: str = Field(..., description="Save under images/, e.g. architecture.png")
#     alt: str
#     caption: str
#     prompt: str = Field(..., description="Prompt sent to the HF Stable Diffusion model.")
#     size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
#     quality: Literal["low", "medium", "high"] = "medium"


# class GlobalImagePlan(BaseModel):
#     md_with_placeholders: str
#     images: List[ImageSpec] = Field(default_factory=list)


# # ======================================================================
# # 2. LangGraph State  (§2.3 Class Diagram – canonical State fields)
# #
# #    `mode`, `recency_days`, `user_research_mode` are internal routing
# #    helpers not shown in the conceptual class diagram but required for
# #    the workflow logic.
# # ======================================================================

# class State(TypedDict):
#     # § Canonical fields (Class Diagram §2.3)
#     topic: str
#     as_of: str
#     router_decision: Optional[dict]       # RouterDecision serialised as dict
#     needs_research: bool
#     queries: List[str]
#     evidence: List[dict]                  # EvidenceItem dicts
#     plan: Optional[Plan]
#     sections: Annotated[List[tuple], operator.add]
#     merged_md: str
#     md_with_placeholders: str
#     image_specs: List[dict]               # ImageSpec dicts (may contain 'error' key)
#     final: str

#     # § Internal routing helpers
#     mode: str
#     recency_days: int
#     user_research_mode: Optional[str]


# # ======================================================================
# # 3. LLM – module-level so Streamlit frontend can hot-swap them
# # ======================================================================
# llm = ChatOllama(model="qwen3:4b", temperature=0.7, timeout=60)
# llm_async = ChatOllama(model="qwen3:4b", temperature=0.7, timeout=60)


# # ======================================================================
# # 4. Router Node  (§2.2: Agent->>Ollama: router decision prompt)
# # ======================================================================
# ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

# Decide whether web research is needed BEFORE planning.

# Modes:
# - closed_book (needs_research=false): evergreen concepts that do not change.
# - hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
# - open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy content.

# If needs_research=true output 3-10 high-signal, scoped search queries.
# For open_book weekly roundup include queries reflecting the last 7 days.

# Return a RouterDecision object.
# """


# def router_node(state: State) -> dict:
#     logger.info(">>> router_node")
#     from datetime import datetime as _dt
#     as_of = state.get("as_of") or _dt.now().date().isoformat()

#     decider = llm.with_structured_output(RouterDecision)
#     try:
#         decision: RouterDecision = decider.invoke([
#             SystemMessage(content=ROUTER_SYSTEM),
#             HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {as_of}"),
#         ])
#     except Exception as exc:
#         logger.error(f"Router LLM failed: {exc} – defaulting to closed_book")
#         decision = RouterDecision(
#             needs_research=False,
#             mode="closed_book",
#             reason="LLM error; defaulting to closed-book",
#             queries=[],
#         )

#     # Respect optional user override from the Streamlit sidebar
#     user_mode = state.get("user_research_mode")
#     if user_mode is not None:
#         decision.mode = user_mode
#         decision.needs_research = user_mode in ("hybrid", "open_book")
#         logger.info(f"User mode override applied: {user_mode}")

#     recency_days = {"open_book": 7, "hybrid": 45}.get(decision.mode, 3650)

#     logger.info(f"Router: mode={decision.mode}, needs_research={decision.needs_research}")
#     logger.info("<<< router_node")
#     return {
#         "router_decision": decision.model_dump(),
#         "needs_research": decision.needs_research,
#         "mode": decision.mode,
#         "queries": decision.queries,
#         "recency_days": recency_days,
#         "as_of": as_of,
#     }


# def route_next(state: State) -> str:
#     """Conditional edge (§2.1 Decision diamond / §2.4 Routing → Researching|Planning)."""
#     return "research" if state["needs_research"] else "orchestrator"


# # ======================================================================
# # 5. Research Node  (§2.1 "Research Node (DuckDuckGo + Ollama)")
# #
# #    §2.2 Sequence two phases:
# #      Phase A – loop: Agent->>Search(DuckDuckGo) → raw results
# #      Phase B – Agent->>Ollama: synthesize evidence prompt → EvidencePack
# # ======================================================================
# RESEARCH_SYSTEM = """You are a research synthesizer.

# Given raw web search results, produce EvidenceItem objects.

# Rules:
# - Only include items with a non-empty url.
# - Prefer relevant and authoritative sources.
# - Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
# - Keep snippets concise (2 sentences maximum).
# - Deduplicate by URL.
# """


# async def research_node(state: State) -> dict:
#     """
#     Two-phase research matching §2.2 Sequence Diagram:
#       Phase A: DuckDuckGo search via Web Research Engine → raw result dicts
#       Phase B: Ollama synthesises raw results → EvidencePack (EvidenceItem list)
#     """
#     logger.info(">>> research_node")

#     queries: List[str] = state.get("queries") or []
#     if not queries:
#         logger.info("No queries – skipping research")
#         return {"evidence": []}

#     # ── Phase A: DuckDuckGo search (§2.2 loop: Agent->>Search) ──────────────
#     raw_results: List[dict] = []

#     if WRE_AVAILABLE:
#         engine = ResearchEngine(
#             max_pages_per_query=WRE_MAX_PAGES_PER_QUERY,
#             timeout=WRE_TIMEOUT,
#             rate_limit_delay=WRE_RATE_LIMIT,
#             min_content_words=WRE_MIN_WORDS,
#         )
#         try:
#             logger.info(
#                 f"Phase A – DuckDuckGo search: {len(queries)} quer(ies) "
#                 f"(timeout={WRE_RESEARCH_TIMEOUT}s)"
#             )
#             report: ResearchReport = await asyncio.wait_for(
#                 asyncio.to_thread(
#                     engine.research,
#                     topic=state["topic"],
#                     queries=queries,
#                     max_results_per_query=3,
#                 ),
#                 timeout=WRE_RESEARCH_TIMEOUT,
#             )
#             for content in report.extracted_content:
#                 if content.error or content.word_count < WRE_MIN_WORDS:
#                     continue
#                 snippet = (
#                     content.content[:500].rsplit(" ", 1)[0] + "..."
#                     if len(content.content) > 500
#                     else content.content
#                 )
#                 raw_results.append({
#                     "title": content.title or content.url,
#                     "url": content.url,
#                     "snippet": snippet,
#                     "published_at": None,
#                     "source": content.metadata.get("search_query", state["topic"]),
#                 })
#             logger.info(f"Phase A complete: {len(raw_results)} raw results")
#         except asyncio.TimeoutError:
#             logger.error(f"DuckDuckGo search timed out after {WRE_RESEARCH_TIMEOUT}s")
#         except Exception as exc:
#             logger.exception(f"Web Research Engine error: {exc}")
#     else:
#         logger.warning("WRE not available – Phase A skipped (no raw results)")

#     if not raw_results:
#         # §2.1 "Research -- No results --> Orchestrator"
#         logger.info("No raw results; passing empty evidence to orchestrator")
#         logger.info("<<< research_node")
#         return {"evidence": []}

#     # ── Phase B: Ollama synthesis → EvidencePack (§2.2 Agent->>Ollama) ───────
#     logger.info(f"Phase B – Ollama synthesis of {len(raw_results)} raw results")
#     extractor = llm.with_structured_output(EvidencePack)
#     try:
#         pack: EvidencePack = extractor.invoke([
#             SystemMessage(content=RESEARCH_SYSTEM),
#             HumanMessage(
#                 content=(
#                     f"As-of date: {state['as_of']}\n"
#                     f"Recency days: {state['recency_days']}\n\n"
#                     f"Raw search results:\n{raw_results}"
#                 )
#             ),
#         ])
#     except Exception as exc:
#         logger.error(f"Ollama synthesis failed: {exc} – falling back to raw results")
#         pack = EvidencePack(
#             evidence=[EvidenceItem(**r) for r in raw_results if r.get("url")]
#         )

#     # Deduplicate by URL
#     seen: dict = {}
#     for item in pack.evidence:
#         if item.url and item.url not in seen:
#             seen[item.url] = item
#     evidence_dicts = [e.model_dump() for e in seen.values()]

#     logger.info(f"Phase B complete: {len(evidence_dicts)} evidence items synthesised")
#     logger.info("<<< research_node")
#     return {"evidence": evidence_dicts}


# # ======================================================================
# # 6. Orchestrator Node  (§2.2: Agent->>Ollama: orchestrate plan prompt)
# # ======================================================================
# ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
# Produce a highly actionable outline for a technical blog post.

# Requirements:
# - 5-9 tasks, each with goal + 3-6 bullets + target_words.
# - Tags are flexible; do not force a fixed taxonomy.

# Grounding:
# - closed_book: evergreen content, no evidence dependence.
# - hybrid: use evidence for up-to-date examples; mark tasks with requires_research=True
#   and requires_citations=True.
# - open_book (news_roundup):
#   - Set blog_kind="news_roundup"
#   - No tutorial content unless explicitly requested
#   - If evidence is weak, reflect that honestly – do NOT invent events.

# Output must match the Plan schema exactly.
# """


# def orchestrator_node(state: State) -> dict:
#     logger.info(">>> orchestrator_node")
#     planner = llm.with_structured_output(Plan)
#     mode = state.get("mode", "closed_book")
#     evidence = state.get("evidence", [])
#     forced_kind = "news_roundup" if mode == "open_book" else None

#     try:
#         plan: Plan = planner.invoke([
#             SystemMessage(content=ORCH_SYSTEM),
#             HumanMessage(
#                 content=(
#                     f"Topic: {state['topic']}\n"
#                     f"Mode: {mode}\n"
#                     f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
#                     f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
#                     f"Evidence (up to 16 items):\n{evidence[:16]}"
#                 )
#             ),
#         ])
#     except Exception as exc:
#         logger.error(f"Orchestrator LLM failed: {exc} – using minimal fallback plan")
#         plan = Plan(
#             blog_title=f"Blog about {state['topic']}",
#             audience="general technical readers",
#             tone="informative",
#             blog_kind="explainer",
#             tasks=[
#                 Task(
#                     id=1,
#                     title="Introduction",
#                     goal="Introduce the topic to the reader.",
#                     bullets=[
#                         "Define the topic",
#                         "Why it matters",
#                         "What the reader will learn",
#                     ],
#                     target_words=200,
#                 )
#             ],
#         )

#     if forced_kind:
#         plan.blog_kind = "news_roundup"

#     logger.info(f"Orchestrator: '{plan.blog_title}' – {len(plan.tasks)} tasks")
#     logger.info("<<< orchestrator_node")
#     return {"plan": plan}


# # ======================================================================
# # 7. Fan-out  (§2.4 Writing composite state – FanOut → Worker×N → Join)
# # ======================================================================

# def fanout(state: State) -> List[Send]:
#     plan = state["plan"]
#     assert plan is not None, "fanout called without a plan"
#     logger.info(f"Fanout: dispatching {len(plan.tasks)} parallel worker tasks")
#     plan_dict = plan.model_dump()
#     evidence = state.get("evidence", [])
#     return [
#         Send(
#             "worker",
#             {
#                 "task": task.model_dump(),
#                 "topic": state["topic"],
#                 "mode": state["mode"],
#                 "as_of": state["as_of"],
#                 "recency_days": state["recency_days"],
#                 "plan": plan_dict,
#                 "evidence": evidence,
#             },
#         )
#         for task in plan.tasks
#     ]


# # ======================================================================
# # 8. Worker Node  (§2.2: par block – Agent->>Ollama: write section prompt)
# # ======================================================================
# WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
# Write ONE section of a technical blog post in Markdown.

# Constraints:
# - Cover ALL bullets in order.
# - Target words +/-15%.
# - Output only section markdown starting with "## <Section Title>".

# Scope guard:
# - If blog_kind=="news_roundup", do NOT drift into tutorials. Focus on events + implications.

# Grounding:
# - If mode=="open_book": do not introduce any specific event/company/model/funding/policy
#   claim unless supported by the provided Evidence URLs. Attach a Markdown link
#   ([Source](URL)) for each supported claim. Write "Not found in provided sources."
#   if unsupported.
# - If requires_citations==true (hybrid tasks): cite Evidence URLs for all external claims.

# Code:
# - If requires_code==true, include at least one minimal runnable snippet.
# """


# async def worker_node(payload: dict) -> dict:
#     task = Task(**payload["task"])
#     plan = Plan(**payload["plan"])
#     evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
#     logger.info(f">>> worker_node – task {task.id}: '{task.title}'")

#     bullets_text = "\n- " + "\n- ".join(task.bullets)
#     evidence_text = "\n".join(
#         f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
#         for e in evidence[:20]
#     )

#     try:
#         response = await llm_async.ainvoke([
#             SystemMessage(content=WORKER_SYSTEM),
#             HumanMessage(
#                 content=(
#                     f"Blog title: {plan.blog_title}\n"
#                     f"Audience: {plan.audience}\n"
#                     f"Tone: {plan.tone}\n"
#                     f"Blog kind: {plan.blog_kind}\n"
#                     f"Constraints: {plan.constraints}\n"
#                     f"Topic: {payload['topic']}\n"
#                     f"Mode: {payload.get('mode')}\n"
#                     f"As-of: {payload.get('as_of')} "
#                     f"(recency_days={payload.get('recency_days')})\n\n"
#                     f"Section title: {task.title}\n"
#                     f"Goal: {task.goal}\n"
#                     f"Target words: {task.target_words}\n"
#                     f"Tags: {task.tags}\n"
#                     f"requires_research: {task.requires_research}\n"
#                     f"requires_citations: {task.requires_citations}\n"
#                     f"requires_code: {task.requires_code}\n"
#                     f"Bullets:{bullets_text}\n\n"
#                     f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
#                 )
#             ),
#         ])
#         section_md = response.content.strip()
#     except Exception as exc:
#         logger.error(f"Worker LLM failed for task {task.id}: {exc}")
#         section_md = f"## {task.title}\n\n*Content generation failed for this section.*"

#     logger.info(f"<<< worker_node – task {task.id}")
#     return {"sections": [(task.id, section_md)]}


# # ======================================================================
# # 9. Reducer Subgraph  (§2.1 Reducer Subgraph / §2.4 Merging → Completed)
# #
# #    Four nodes matching §2.1 Flowchart exactly:
# #      merge_content → decide_images → generate_images → place_images
# # ======================================================================

# # ── 9a. Merge Content  (§2.4 Merging state) ──────────────────────────────────
# def merge_content(state: State) -> dict:
#     logger.info(">>> merge_content")
#     plan = state["plan"]
#     if plan is None:
#         raise ValueError("merge_content called without a plan")
#     ordered = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
#     body = "\n\n".join(ordered).strip()
#     merged_md = f"# {plan.blog_title}\n\n{body}\n"
#     logger.info("<<< merge_content")
#     return {"merged_md": merged_md}


# # ── 9b. Decide Images  (§2.4 DecidingImages state) ───────────────────────────
# DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
# Decide if images or diagrams are needed for this blog post.

# Rules:
# - Generate 3-5 images: one feature/hero image and supporting technical diagrams.
# - Each image must materially improve the reader's understanding.
# - Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], ... [[IMAGE_N]].
# - If genuinely no images are needed: md_with_placeholders equals the input and images=[].
# - Avoid purely decorative images; prefer diagrams with short descriptive labels.

# Return a GlobalImagePlan object.
# """


# def decide_images(state: State) -> dict:
#     logger.info(">>> decide_images")
#     planner = llm.with_structured_output(GlobalImagePlan)
#     merged_md = state["merged_md"]
#     plan = state["plan"]
#     assert plan is not None

#     try:
#         image_plan: GlobalImagePlan = planner.invoke([
#             SystemMessage(content=DECIDE_IMAGES_SYSTEM),
#             HumanMessage(
#                 content=(
#                     f"Blog kind: {plan.blog_kind}\n"
#                     f"Topic: {state['topic']}\n\n"
#                     "Insert placeholders and propose Stable Diffusion image prompts.\n\n"
#                     f"{merged_md}"
#                 )
#             ),
#         ])
#     except Exception as exc:
#         logger.error(f"Image planning failed: {exc} – skipping images")
#         image_plan = GlobalImagePlan(md_with_placeholders=merged_md, images=[])

#     logger.info(f"decide_images: {len(image_plan.images)} image spec(s)")
#     logger.info("<<< decide_images")
#     return {
#         "md_with_placeholders": image_plan.md_with_placeholders,
#         "image_specs": [img.model_dump() for img in image_plan.images],
#     }


# # ── 9c. Generate Images  (§2.1 "Hugging Face Stable Diffusion" / §2.2 Agent->>HF) ──
# def _hf_generate_image_bytes(prompt: str) -> bytes:
#     """
#     Generate an image via Hugging Face Inference API (Stable Diffusion).
#     Reads HF_TOKEN from environment. Raises RuntimeError on any failure so
#     generate_images can mark the spec with an 'error' key and continue.
#     """
#     if not HF_AVAILABLE:
#         raise RuntimeError(
#             "huggingface_hub is not installed. "
#             "Run: pip install huggingface-hub pillow"
#         )
#     try:
#         client = InferenceClient()   # reads HF_TOKEN env var
#         pil_image = client.text_to_image(
#             prompt,
#             model="stabilityai/stable-diffusion-2-1",
#         )
#         buf = BytesIO()
#         pil_image.save(buf, format="PNG")
#         return buf.getvalue()
#     except Exception as exc:
#         raise RuntimeError(f"Hugging Face image generation failed: {exc}") from exc


# def generate_images(state: State) -> dict:
#     """
#     §2.4 GeneratingImages state: loop over image_specs, call HF for each.
#     On success, image file is written to images/.
#     On failure, spec is annotated with 'error' key so place_images inserts
#     the error block instead of an img tag (§2.1 failure edge / §2.4 note).
#     """
#     logger.info(">>> generate_images")
#     image_specs: List[dict] = state.get("image_specs") or []

#     if not image_specs:
#         logger.info("No image specs – nothing to generate")
#         logger.info("<<< generate_images")
#         return {"image_specs": []}

#     images_dir = Path("images")
#     images_dir.mkdir(exist_ok=True)

#     updated_specs: List[dict] = []
#     for spec in image_specs:
#         filename = spec["filename"]
#         out_path = images_dir / filename
#         spec_copy = {k: v for k, v in spec.items() if k != "error"}  # clear old error

#         if out_path.exists():
#             logger.info(f"Reusing existing image: {filename}")
#             updated_specs.append(spec_copy)
#             continue

#         # §2.2: Agent->>HF: generate image from prompt
#         try:
#             img_bytes = _hf_generate_image_bytes(spec["prompt"])
#             out_path.write_bytes(img_bytes)
#             logger.info(f"Generated: {filename}")
#         except Exception as exc:
#             # §2.1 / §2.4: failure → annotate; place_images will insert error block
#             logger.error(f"Image generation failed for {filename}: {exc}")
#             spec_copy["error"] = str(exc)

#         updated_specs.append(spec_copy)

#     logger.info("<<< generate_images")
#     return {"image_specs": updated_specs}


# # ── 9d. Place Images  (§2.1 PlaceImages + Save) ──────────────────────────────
# def _safe_slug(title: str) -> str:
#     """Convert a blog title to a filesystem-safe slug."""
#     s = title.strip().lower()
#     s = re.sub(r"[^a-z0-9 _-]+", "", s)
#     s = re.sub(r"\s+", "_", s).strip("_")
#     return s or "blog"


# def place_images(state: State) -> dict:
#     """
#     §2.4 PlacingImages state: substitute placeholders with img markdown or error blocks.
#     §2.1: Save final .md file → transitions to Completed → End.
#     """
#     logger.info(">>> place_images")
#     plan = state["plan"]
#     assert plan is not None

#     md = state.get("md_with_placeholders") or state["merged_md"]
#     image_specs: List[dict] = state.get("image_specs") or []

#     for spec in image_specs:
#         placeholder = spec["placeholder"]

#         if spec.get("error"):
#             # §2.1 "GenerateImages -- Failure --> PlaceImages[error message]"
#             error_block = (
#                 f"> **[IMAGE GENERATION FAILED]** {spec.get('caption', '')}\n>\n"
#                 f"> **Alt:** {spec.get('alt', '')}\n>\n"
#                 f"> **Prompt:** {spec.get('prompt', '')}\n>\n"
#                 f"> **Error:** {spec['error']}\n"
#             )
#             md = md.replace(placeholder, error_block)
#         else:
#             img_md = (
#                 f"![{spec['alt']}](images/{spec['filename']})\n"
#                 f"*{spec['caption']}*"
#             )
#             md = md.replace(placeholder, img_md)

#     # §2.1 "Save final .md file"
#     final_filename = f"{_safe_slug(plan.blog_title)}.md"
#     Path(final_filename).write_text(md, encoding="utf-8")
#     logger.info(f"Final blog saved as: {final_filename}")
#     logger.info("<<< place_images")
#     return {"final": md}


# # ── Build Reducer Subgraph (§2.1 Reducer Subgraph) ───────────────────────────
# _reducer_graph = StateGraph(State)
# _reducer_graph.add_node("merge_content", merge_content)
# _reducer_graph.add_node("decide_images", decide_images)
# _reducer_graph.add_node("generate_images", generate_images)
# _reducer_graph.add_node("place_images", place_images)

# _reducer_graph.add_edge(START, "merge_content")
# _reducer_graph.add_edge("merge_content", "decide_images")
# _reducer_graph.add_edge("decide_images", "generate_images")
# _reducer_graph.add_edge("generate_images", "place_images")
# _reducer_graph.add_edge("place_images", END)

# reducer_subgraph = _reducer_graph.compile()


# # ======================================================================
# # 10. Main Graph  (§2.1 Flowchart – exact node and edge topology)
# #
# #   START → router → [research] → orchestrator → worker(xN) → reducer → END
# #
# #   There is NO seo_optimizer node: it does not appear in §2.1, §2.2,
# #   §2.3, or §2.4 and must not be present.
# # ======================================================================
# graph = StateGraph(State)

# graph.add_node("router", router_node)
# graph.add_node("research", research_node)
# graph.add_node("orchestrator", orchestrator_node)
# graph.add_node("worker", worker_node)
# graph.add_node("reducer", reducer_subgraph)

# graph.add_edge(START, "router")
# graph.add_conditional_edges(
#     "router",
#     route_next,
#     {"research": "research", "orchestrator": "orchestrator"},
# )
# graph.add_edge("research", "orchestrator")
# graph.add_conditional_edges("orchestrator", fanout, ["worker"])
# graph.add_edge("worker", "reducer")
# graph.add_edge("reducer", END)   # §2.1: Save → End (no intermediate SEO node)

# app = graph.compile()


# # ======================================================================
# # 11. CLI Entry Point
# # ======================================================================
# async def main() -> None:
#     import argparse
#     from datetime import datetime as _dt

#     parser = argparse.ArgumentParser(description="BWA Blog Writing Agent")
#     parser.add_argument("topic", help="Blog topic")
#     parser.add_argument(
#         "--as-of",
#         default=_dt.now().date().isoformat(),
#         help="Reference date (YYYY-MM-DD). Defaults to today.",
#     )
#     args = parser.parse_args()

#     initial_state: State = {
#         "topic": args.topic,
#         "as_of": args.as_of,
#         "sections": [],
#         "user_research_mode": None,
#     }

#     logger.info(f"Starting blog generation for topic: {args.topic}")
#     final_state = await app.ainvoke(initial_state)

#     plan = final_state.get("plan")
#     if plan:
#         filename = f"{_safe_slug(plan.blog_title)}.md"
#         print(f"\nBlog post generated: {filename}")
#         print("Preview (first 500 chars):")
#         print(final_state.get("final", "")[:500] + "...")
#     else:
#         print("Blog generation failed – check logs.")


# if __name__ == "__main__":
#     asyncio.run(main())







































# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




























#!/usr/bin/env python3
"""
BWA Blog Writing Agent – Production-Grade Implementation
Faithful to the Visual Representation specification (diagrams 2.1 – 2.4).

Workflow (§2.1 Flowchart / §2.4 State Diagram):
  START
    → router_node          (Ollama)
    → [research_node]      (DuckDuckGo + Ollama synthesis)  ← conditional
    → orchestrator_node    (Ollama)
    → [worker_node ×N]     (Ollama, parallel fan-out)
    ┌─ Reducer Subgraph ──────────────────────────────────────────────────┐
    │  merge_content → decide_images → generate_images → place_images    │
    └─────────────────────────────────────────────────────────────────────┘
  END

Image generation uses Hugging Face Stable Diffusion (§2.1, §2.2).
Research uses DuckDuckGo (WRE) then Ollama synthesis → EvidencePack (§2.2 Sequence).
No SEO Optimizer node (not present in any diagram).
"""

import asyncio
import logging
import operator
import re
from io import BytesIO
from pathlib import Path
from typing import Annotated, List, Literal, Optional, TypedDict
from datetime import datetime as _dt_module

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

# ── Hugging Face image generation (§2.1 / §2.2) ─────────────────────────────
try:
    from huggingface_hub import InferenceClient  # type: ignore
    from PIL import Image  # noqa: F401  (imported to validate bytes in tests)
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning(
        "huggingface_hub / Pillow not installed – image generation will produce "
        "error blocks. Install with: pip install huggingface-hub pillow"
    )

# ── Web Research Engine (DuckDuckGo, §2.1 / §2.2) ───────────────────────────
try:
    from web_research_engine import ResearchEngine, ResearchReport, OutputFormatter  # type: ignore
    WRE_AVAILABLE = True
except ImportError:
    WRE_AVAILABLE = False
    logging.warning(
        "Web Research Engine not found – research node will return empty results."
    )

load_dotenv()

# ======================================================================
# Logging
# ======================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ======================================================================
# Research output directory (WRE reports: JSON + MD + HTML)
# ======================================================================
RESEARCH_OUTPUT_DIR = Path("Analysis and Proposed Updates\research_outputs")

# ======================================================================
# Qwen3 Reasoning Trace Extraction
#
# Qwen3 emits <think>…</think> blocks when thinking mode is active.
# We strip them from content going into the blog and store them separately.
# ======================================================================
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _extract_thinking(text: str) -> tuple:
    """
    Return (clean_text, thinking_text).
    clean_text   – response with all <think>…</think> blocks removed.
    thinking_text – concatenated reasoning (empty string if none present).
    """
    blocks = _THINK_RE.findall(text)
    thinking = "\n\n---\n\n".join(b.strip() for b in blocks)
    clean = _THINK_RE.sub("", text).strip()
    return clean, thinking

# ======================================================================
# WRE configuration constants
# ======================================================================
WRE_MAX_PAGES_PER_QUERY: int = 2
WRE_TIMEOUT: int = 20
WRE_RATE_LIMIT: float = 1.0
WRE_MIN_WORDS: int = 100
WRE_RESEARCH_TIMEOUT: int = 90   # hard async timeout for the entire search phase

# ======================================================================
# 1. Pydantic Schemas  (§2.3 Class Diagram)
# ======================================================================

class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing the reader's takeaway.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120-550).")
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal[
        "explainer", "tutorial", "news_roundup", "comparison", "system_design"
    ] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None   # ISO "YYYY-MM-DD"
    snippet: Optional[str] = None
    source: Optional[str] = None


class EvidencePack(BaseModel):
    """Produced by the Ollama synthesis step inside research_node (§2.2 Sequence)."""
    evidence: List[EvidenceItem] = Field(default_factory=list)


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(3)


class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. architecture.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt sent to the HF Stable Diffusion model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)


# ======================================================================
# 2. LangGraph State  (§2.3 Class Diagram – canonical State fields)
#
#    `mode`, `recency_days`, `user_research_mode` are internal routing
#    helpers not shown in the conceptual class diagram but required for
#    the workflow logic.
# ======================================================================

class State(TypedDict):
    # § Canonical fields (Class Diagram §2.3)
    topic: str
    as_of: str
    router_decision: Optional[dict]       # RouterDecision serialised as dict
    needs_research: bool
    queries: List[str]
    evidence: List[dict]                  # EvidenceItem dicts
    plan: Optional[Plan]
    sections: Annotated[List[tuple], operator.add]
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]               # ImageSpec dicts (may contain 'error' key)
    final: str

    # § Internal routing helpers
    mode: str
    recency_days: int
    user_research_mode: Optional[str]

    # § Qwen3 reasoning traces captured per worker section
    # Each entry: {"task_id": int, "title": str, "thinking": str}
    thinking_traces: Annotated[List[dict], operator.add]


# ======================================================================
# 3. LLM – module-level so Streamlit frontend can hot-swap them
# ======================================================================
llm = ChatOllama(model="llama3.2:3b", temperature=0.7, timeout=60)
llm_async = ChatOllama(model="llama3.2:3b", temperature=0.7, timeout=60)


# ======================================================================
# 4. Router Node  (§2.2: Agent->>Ollama: router decision prompt)
# ======================================================================
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts that do not change.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy content.

If needs_research=true output 3-10 high-signal, scoped search queries.
For open_book weekly roundup include queries reflecting the last 7 days.

Return a RouterDecision object.
"""


def router_node(state: State) -> dict:
    logger.info(">>> router_node")
    from datetime import datetime as _dt
    as_of = state.get("as_of") or _dt.now().date().isoformat()

    decider = llm.with_structured_output(RouterDecision)
    try:
        decision: RouterDecision = decider.invoke([
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {as_of}"),
        ])
    except Exception as exc:
        logger.error(f"Router LLM failed: {exc} – defaulting to closed_book")
        decision = RouterDecision(
            needs_research=False,
            mode="closed_book",
            reason="LLM error; defaulting to closed-book",
            queries=[],
        )

    # Respect optional user override from the Streamlit sidebar
    user_mode = state.get("user_research_mode")
    if user_mode is not None:
        decision.mode = user_mode
        decision.needs_research = user_mode in ("hybrid", "open_book")
        logger.info(f"User mode override applied: {user_mode}")

    recency_days = {"open_book": 7, "hybrid": 45}.get(decision.mode, 3650)

    logger.info(f"Router: mode={decision.mode}, needs_research={decision.needs_research}")
    logger.info("<<< router_node")
    return {
        "router_decision": decision.model_dump(),
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
        "as_of": as_of,
    }


def route_next(state: State) -> str:
    """Conditional edge (§2.1 Decision diamond / §2.4 Routing → Researching|Planning)."""
    return "research" if state["needs_research"] else "orchestrator"


# ======================================================================
# 5. Research Node  (§2.1 "Research Node (DuckDuckGo + Ollama)")
#
#    §2.2 Sequence two phases:
#      Phase A – loop: Agent->>Search(DuckDuckGo) → raw results
#      Phase B – Agent->>Ollama: synthesize evidence prompt → EvidencePack
# ======================================================================
RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant and authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets concise (2 sentences maximum).
- Deduplicate by URL.
"""


async def research_node(state: State) -> dict:
    """
    Two-phase research matching §2.2 Sequence Diagram:
      Phase A: DuckDuckGo search via Web Research Engine → raw result dicts
      Phase B: Ollama synthesises raw results → EvidencePack (EvidenceItem list)
    """
    logger.info(">>> research_node")

    queries: List[str] = state.get("queries") or []
    if not queries:
        logger.info("No queries – skipping research")
        return {"evidence": []}

    # ── Phase A: DuckDuckGo search (§2.2 loop: Agent->>Search) ──────────────
    raw_results: List[dict] = []

    if WRE_AVAILABLE:
        engine = ResearchEngine(
            max_pages_per_query=WRE_MAX_PAGES_PER_QUERY,
            timeout=WRE_TIMEOUT,
            rate_limit_delay=WRE_RATE_LIMIT,
            min_content_words=WRE_MIN_WORDS,
        )
        try:
            logger.info(
                f"Phase A – DuckDuckGo search: {len(queries)} quer(ies) "
                f"(timeout={WRE_RESEARCH_TIMEOUT}s)"
            )
            report: ResearchReport = await asyncio.wait_for(
                asyncio.to_thread(
                    engine.research,
                    topic=state["topic"],
                    queries=queries,
                    max_results_per_query=3,
                ),
                timeout=WRE_RESEARCH_TIMEOUT,
            )
            for content in report.extracted_content:
                if content.error or content.word_count < WRE_MIN_WORDS:
                    continue
                snippet = (
                    content.content[:500].rsplit(" ", 1)[0] + "..."
                    if len(content.content) > 500
                    else content.content
                )
                raw_results.append({
                    "title": content.title or content.url,
                    "url": content.url,
                    "snippet": snippet,
                    "published_at": None,
                    "source": content.metadata.get("search_query", state["topic"]),
                })
            logger.info(f"Phase A complete: {len(raw_results)} raw results")
        except asyncio.TimeoutError:
            logger.error(f"DuckDuckGo search timed out after {WRE_RESEARCH_TIMEOUT}s")
        except Exception as exc:
            logger.exception(f"Web Research Engine error: {exc}")
    else:
        logger.warning("WRE not available – Phase A skipped (no raw results)")

    # ── Save WRE output files to research_output/ (JSON + MD + HTML) ────────
    if WRE_AVAILABLE and raw_results:
        try:
            RESEARCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            ts = _dt_module.now().strftime("%Y%m%d_%H%M%S")
            slug = re.sub(r"[^a-z0-9_]", "_", state["topic"].lower())[:40]
            base = RESEARCH_OUTPUT_DIR / f"{slug}_{ts}"
            OutputFormatter.to_json(report, Path(str(base) + ".json"))
            OutputFormatter.to_markdown(report, Path(str(base) + ".md"))
            OutputFormatter.to_html(report, Path(str(base) + ".html"))
            logger.info(
                f"Research outputs saved → {base}.{{json,md,html}}"
            )
        except Exception as exc:
            logger.warning(f"Could not save research output files: {exc}")

    if not raw_results:
        # §2.1 "Research -- No results --> Orchestrator"
        logger.info("No raw results; passing empty evidence to orchestrator")
        logger.info("<<< research_node")
        return {"evidence": []}

    # ── Phase B: Ollama synthesis → EvidencePack (§2.2 Agent->>Ollama) ───────
    logger.info(f"Phase B – Ollama synthesis of {len(raw_results)} raw results")
    extractor = llm.with_structured_output(EvidencePack)
    try:
        pack: EvidencePack = extractor.invoke([
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of date: {state['as_of']}\n"
                    f"Recency days: {state['recency_days']}\n\n"
                    f"Raw search results:\n{raw_results}"
                )
            ),
        ])
    except Exception as exc:
        logger.error(f"Ollama synthesis failed: {exc} – falling back to raw results")
        pack = EvidencePack(
            evidence=[EvidenceItem(**r) for r in raw_results if r.get("url")]
        )

    # Deduplicate by URL
    seen: dict = {}
    for item in pack.evidence:
        if item.url and item.url not in seen:
            seen[item.url] = item
    evidence_dicts = [e.model_dump() for e in seen.values()]

    logger.info(f"Phase B complete: {len(evidence_dicts)} evidence items synthesised")
    logger.info("<<< research_node")
    return {"evidence": evidence_dicts}


# ======================================================================
# 6. Orchestrator Node  (§2.2: Agent->>Ollama: orchestrate plan prompt)
# ======================================================================
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 5-9 tasks, each with goal + 3-6 bullets + target_words.
- Tags are flexible; do not force a fixed taxonomy.

Grounding:
- closed_book: evergreen content, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark tasks with requires_research=True
  and requires_citations=True.
- open_book (news_roundup):
  - Set blog_kind="news_roundup"
  - No tutorial content unless explicitly requested
  - If evidence is weak, reflect that honestly – do NOT invent events.

Output must match the Plan schema exactly.
"""


def orchestrator_node(state: State) -> dict:
    logger.info(">>> orchestrator_node")
    planner = llm.with_structured_output(Plan)
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])
    forced_kind = "news_roundup" if mode == "open_book" else None

    try:
        plan: Plan = planner.invoke([
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                    f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                    f"Evidence (up to 16 items):\n{evidence[:16]}"
                )
            ),
        ])
    except Exception as exc:
        logger.error(f"Orchestrator LLM failed: {exc} – using minimal fallback plan")
        plan = Plan(
            blog_title=f"Blog about {state['topic']}",
            audience="general technical readers",
            tone="informative",
            blog_kind="explainer",
            tasks=[
                Task(
                    id=1,
                    title="Introduction",
                    goal="Introduce the topic to the reader.",
                    bullets=[
                        "Define the topic",
                        "Why it matters",
                        "What the reader will learn",
                    ],
                    target_words=200,
                )
            ],
        )

    if forced_kind:
        plan.blog_kind = "news_roundup"

    logger.info(f"Orchestrator: '{plan.blog_title}' – {len(plan.tasks)} tasks")
    logger.info("<<< orchestrator_node")
    return {"plan": plan}


# ======================================================================
# 7. Fan-out  (§2.4 Writing composite state – FanOut → Worker×N → Join)
# ======================================================================

def fanout(state: State) -> List[Send]:
    plan = state["plan"]
    assert plan is not None, "fanout called without a plan"
    logger.info(f"Fanout: dispatching {len(plan.tasks)} parallel worker tasks")
    plan_dict = plan.model_dump()
    evidence = state.get("evidence", [])
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": plan_dict,
                "evidence": evidence,
            },
        )
        for task in plan.tasks
    ]


# ======================================================================
# 8. Worker Node  (§2.2: par block – Agent->>Ollama: write section prompt)
# ======================================================================
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order.
- Target words +/-15%.
- Output only section markdown starting with "## <Section Title>".

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials. Focus on events + implications.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy
  claim unless supported by the provided Evidence URLs. Attach a Markdown link
  ([Source](URL)) for each supported claim. Write "Not found in provided sources."
  if unsupported.
- If requires_citations==true (hybrid tasks): cite Evidence URLs for all external claims.

Code:
- If requires_code==true, include at least one minimal runnable snippet.
"""


async def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    logger.info(f">>> worker_node – task {task.id}: '{task.title}'")

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
    )

    try:
        response = await llm_async.ainvoke([
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {payload['topic']}\n"
                    f"Mode: {payload.get('mode')}\n"
                    f"As-of: {payload.get('as_of')} "
                    f"(recency_days={payload.get('recency_days')})\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                )
            ),
        ])
        raw_content = response.content.strip()
        section_md, thinking = _extract_thinking(raw_content)
        if not section_md:          # in case the whole reply was inside <think>
            section_md = raw_content
            thinking = ""
    except Exception as exc:
        logger.error(f"Worker LLM failed for task {task.id}: {exc}")
        section_md = f"## {task.title}\n\n*Content generation failed for this section.*"
        thinking = ""

    thinking_entry = (
        [{"task_id": task.id, "title": task.title, "thinking": thinking}]
        if thinking else []
    )

    logger.info(f"<<< worker_node – task {task.id}")
    return {
        "sections": [(task.id, section_md)],
        "thinking_traces": thinking_entry,
    }


# ======================================================================
# 9. Reducer Subgraph  (§2.1 Reducer Subgraph / §2.4 Merging → Completed)
#
#    Four nodes matching §2.1 Flowchart exactly:
#      merge_content → decide_images → generate_images → place_images
# ======================================================================

# ── 9a. Merge Content  (§2.4 Merging state) ──────────────────────────────────
def merge_content(state: State) -> dict:
    logger.info(">>> merge_content")
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without a plan")
    ordered = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    logger.info("<<< merge_content")
    return {"merged_md": merged_md}


# ── 9b. Decide Images  (§2.4 DecidingImages state) ───────────────────────────
DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images or diagrams are needed for this blog post.

Rules:
- Generate 3-5 images: one feature/hero image and supporting technical diagrams.
- Each image must materially improve the reader's understanding.
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], ... [[IMAGE_N]].
- If genuinely no images are needed: md_with_placeholders equals the input and images=[].
- Avoid purely decorative images; prefer diagrams with short descriptive labels.

Return a GlobalImagePlan object.
"""


def decide_images(state: State) -> dict:
    logger.info(">>> decide_images")
    planner = llm.with_structured_output(GlobalImagePlan)
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    try:
        image_plan: GlobalImagePlan = planner.invoke([
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Insert placeholders and propose Stable Diffusion image prompts.\n\n"
                    f"{merged_md}"
                )
            ),
        ])
    except Exception as exc:
        logger.error(f"Image planning failed: {exc} – skipping images")
        image_plan = GlobalImagePlan(md_with_placeholders=merged_md, images=[])

    logger.info(f"decide_images: {len(image_plan.images)} image spec(s)")
    logger.info("<<< decide_images")
    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }


# ── 9c. Generate Images  (§2.1 "Hugging Face Stable Diffusion" / §2.2 Agent->>HF) ──
def _hf_generate_image_bytes(prompt: str) -> bytes:
    """
    Generate an image via Hugging Face Inference API (Stable Diffusion).
    Reads HF_TOKEN from environment. Raises RuntimeError on any failure so
    generate_images can mark the spec with an 'error' key and continue.
    """
    if not HF_AVAILABLE:
        raise RuntimeError(
            "huggingface_hub is not installed. "
            "Run: pip install huggingface-hub pillow"
        )
    try:
        client = InferenceClient()   # reads HF_TOKEN env var
        pil_image = client.text_to_image(
            prompt,
            model="stabilityai/stable-diffusion-2-1",
        )
        buf = BytesIO()
        pil_image.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as exc:
        raise RuntimeError(f"Hugging Face image generation failed: {exc}") from exc


def generate_images(state: State) -> dict:
    """
    §2.4 GeneratingImages state: loop over image_specs, call HF for each.
    On success, image file is written to images/.
    On failure, spec is annotated with 'error' key so place_images inserts
    the error block instead of an img tag (§2.1 failure edge / §2.4 note).
    """
    logger.info(">>> generate_images")
    image_specs: List[dict] = state.get("image_specs") or []

    if not image_specs:
        logger.info("No image specs – nothing to generate")
        logger.info("<<< generate_images")
        return {"image_specs": []}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    updated_specs: List[dict] = []
    for spec in image_specs:
        filename = spec["filename"]
        out_path = images_dir / filename
        spec_copy = {k: v for k, v in spec.items() if k != "error"}  # clear old error

        if out_path.exists():
            logger.info(f"Reusing existing image: {filename}")
            updated_specs.append(spec_copy)
            continue

        # §2.2: Agent->>HF: generate image from prompt
        try:
            img_bytes = _hf_generate_image_bytes(spec["prompt"])
            out_path.write_bytes(img_bytes)
            logger.info(f"Generated: {filename}")
        except Exception as exc:
            # §2.1 / §2.4: failure → annotate; place_images will insert error block
            logger.error(f"Image generation failed for {filename}: {exc}")
            spec_copy["error"] = str(exc)

        updated_specs.append(spec_copy)

    logger.info("<<< generate_images")
    return {"image_specs": updated_specs}


# ── 9d. Place Images  (§2.1 PlaceImages + Save) ──────────────────────────────
def _safe_slug(title: str) -> str:
    """Convert a blog title to a filesystem-safe slug."""
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def place_images(state: State) -> dict:
    """
    §2.4 PlacingImages state: substitute placeholders with img markdown or error blocks.
    §2.1: Save final .md file → transitions to Completed → End.
    """
    logger.info(">>> place_images")
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs: List[dict] = state.get("image_specs") or []

    for spec in image_specs:
        placeholder = spec["placeholder"]

        if spec.get("error"):
            # §2.1 "GenerateImages -- Failure --> PlaceImages[error message]"
            error_block = (
                f"> **[IMAGE GENERATION FAILED]** {spec.get('caption', '')}\n>\n"
                f"> **Alt:** {spec.get('alt', '')}\n>\n"
                f"> **Prompt:** {spec.get('prompt', '')}\n>\n"
                f"> **Error:** {spec['error']}\n"
            )
            md = md.replace(placeholder, error_block)
        else:
            img_md = (
                f"![{spec['alt']}](images/{spec['filename']})\n"
                f"*{spec['caption']}*"
            )
            md = md.replace(placeholder, img_md)

    # §2.1 "Save final .md file"
    final_filename = f"{_safe_slug(plan.blog_title)}.md"
    Path(final_filename).write_text(md, encoding="utf-8")
    logger.info(f"Final blog saved as: {final_filename}")
    logger.info("<<< place_images")
    return {"final": md}


# ── Build Reducer Subgraph (§2.1 Reducer Subgraph) ───────────────────────────
_reducer_graph = StateGraph(State)
_reducer_graph.add_node("merge_content", merge_content)
_reducer_graph.add_node("decide_images", decide_images)
_reducer_graph.add_node("generate_images", generate_images)
_reducer_graph.add_node("place_images", place_images)

_reducer_graph.add_edge(START, "merge_content")
_reducer_graph.add_edge("merge_content", "decide_images")
_reducer_graph.add_edge("decide_images", "generate_images")
_reducer_graph.add_edge("generate_images", "place_images")
_reducer_graph.add_edge("place_images", END)

reducer_subgraph = _reducer_graph.compile()


# ======================================================================
# 10. Main Graph  (§2.1 Flowchart – exact node and edge topology)
#
#   START → router → [research] → orchestrator → worker(xN) → reducer → END
#
#   There is NO seo_optimizer node: it does not appear in §2.1, §2.2,
#   §2.3, or §2.4 and must not be present.
# ======================================================================
graph = StateGraph(State)

graph.add_node("router", router_node)
graph.add_node("research", research_node)
graph.add_node("orchestrator", orchestrator_node)
graph.add_node("worker", worker_node)
graph.add_node("reducer", reducer_subgraph)

graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    route_next,
    {"research": "research", "orchestrator": "orchestrator"},
)
graph.add_edge("research", "orchestrator")
graph.add_conditional_edges("orchestrator", fanout, ["worker"])
graph.add_edge("worker", "reducer")
graph.add_edge("reducer", END)   # §2.1: Save → End (no intermediate SEO node)

app = graph.compile()

app

# ======================================================================
# 11. CLI Entry Point
# ======================================================================
async def main() -> None:
    import argparse
    from datetime import datetime as _dt

    parser = argparse.ArgumentParser(description="BWA Blog Writing Agent")
    parser.add_argument("topic", help="Blog topic")
    parser.add_argument(
        "--as-of",
        default=_dt.now().date().isoformat(),
        help="Reference date (YYYY-MM-DD). Defaults to today.",
    )
    args = parser.parse_args()

    initial_state: State = {
        "topic": args.topic,
        "as_of": args.as_of,
        "sections": [],
        "thinking_traces": [],
        "user_research_mode": None,
    }

    logger.info(f"Starting blog generation for topic: {args.topic}")
    final_state = await app.ainvoke(initial_state)

    plan = final_state.get("plan")
    if plan:
        filename = f"{_safe_slug(plan.blog_title)}.md"
        print(f"\nBlog post generated: {filename}")
        print("Preview (first 500 chars):")
        print(final_state.get("final", "")[:500] + "...")
    else:
        print("Blog generation failed – check logs.")


if __name__ == "__main__":
    asyncio.run(main())