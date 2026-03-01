# Blog-Agent: Comprehensive Multi-Agent Architecture Analysis Report

**Analyst:** Agentic Workflow Architect (Simulated Multi-Agent Review)  
**Date:** 2026-03-01  
**Project:** `blog-agent` — Autonomous AI Blog Generation System  
**Stack:** LangGraph · Ollama (Mistral) · SearxNG · Stable Diffusion v1.4 · Streamlit · ChromaDB

---

## Directory Tree

```
blog-agent/
├── agents/
│   ├── citation_manager.py
│   ├── image_agent.py
│   ├── planner.py
│   ├── reducer.py
│   ├── researcher.py
│   ├── router.py
│   └── writer.py
├── app/
│   ├── config.py
│   └── ui.py
├── docker/
│   ├── dockerfile
│   ├── docker-compose.yml
│   └── searxng/
│       └── settings.yml
├── graph/
│   ├── checkpointer.py
│   ├── graph_builder.py
│   └── state.py
├── memory/
│   ├── cache.py
│   └── chroma_store.py
├── prompts/
│   ├── moderator_prompt.txt
│   ├── planner_prompt.txt
│   ├── researcher_prompt.txt
│   ├── router_prompt.txt
│   └── writer_prompt.txt
├── tests/
│   ├── fixtures/
│   │   ├── sample_blog_plan.json
│   │   ├── sample_research_result.json
│   │   ├── sample_search_results.json
│   │   └── sample_section_draft.json
│   ├── test_graph_integration.py
│   ├── test_planner.py
│   ├── test_researcher.py
│   ├── test_router.py
│   └── test_writer.py
├── tools/
│   ├── image_gen.py
│   ├── search.py
│   └── web_fetcher.py
├── .env.example
├── README.md
├── blog_agent_file_creation_plan.md
└── requirements.txt
```

---

## Agent Roles

| Agent | Node Function | Role |
|---|---|---|
| **Traversal Agent** | (this report) | Directory navigation and file ordering |
| **Analysis Agent** | Per-file sections below | Parses purpose, logic, dependencies |
| **Improvement Agent** | "Improvements/Changes" sections | Identifies bugs, anti-patterns, fixes |
| **Architecture Agent** | Final Report section | Reviews global system design |
| **Reporting Agent** | Full document | Compiles findings |

---

---

## Folder: `blog-agent/` (Root)

**Subfolders:** `agents/`, `app/`, `docker/`, `graph/`, `memory/`, `prompts/`, `tests/`, `tools/`  
**Files:** `requirements.txt`, `.env.example`, `README.md`

---

### File: `requirements.txt`

**Purpose:** Declares Python dependencies. Contains both a large commented-out pinned-version block (serving as documentation for exact tested versions) and active unpinned ranges.

**Logic and Dependencies:** Covers all stack layers — LangGraph, LangChain, Ollama, diffusers/torch, Streamlit, ChromaDB, httpx, BeautifulSoup, Pydantic, structlog, dotenv, DuckDuckGo, Pillow, markdown, pytest.

**Contribution:** Enables reproducible installs and CI setup.

**Improvements/Changes:**
1. **Dual-block confusion** — the commented-out pinned block and the active unpinned block coexist in the same file with no separator comment. Any developer who skims the file may not realize which section is active. Clean up by either keeping only one format or clearly labeling sections.
2. **Missing `langgraph-checkpoint-sqlite`** — the checkpointer module imports `langgraph.checkpoint.sqlite.SqliteSaver` and `AsyncSqliteSaver`, which live in the `langgraph-checkpoint-sqlite` package. This package is not listed in `requirements.txt` and will cause an `ImportError` at runtime.
3. **`aiosqlite` missing** — required by `AsyncSqliteSaver`.

**Updated Content (critical additions):**
```
# Add these to requirements.txt:
langgraph-checkpoint-sqlite>=1.0.0
aiosqlite>=0.19.0
```

---

### File: `.env.example`

**Purpose:** Documents all environment variables with sensible defaults and descriptions.

**Logic and Dependencies:** Covers Ollama, SearxNG, Stable Diffusion, ChromaDB, output dirs, cache TTL, image dimensions, and optional LangSmith tracing.

**Contribution:** Enables quick project setup via `cp .env.example .env`.

**Improvements/Changes:**
1. `SD_SAFETY_CHECKER` is imported in `tools/image_gen.py` (see below) but is absent from `.env.example`. Add it.

**Updated section:**
```env
# Whether to enable Stable Diffusion safety checker (true/false)
SD_SAFETY_CHECKER=false
```

---

---

## Folder: `app/`

**Files:** `config.py`, `ui.py`

---

### File: `app/config.py`

**Purpose:** Centralised configuration that loads `.env` and exposes typed module-level constants. Creates all output directories on import.

**Logic and Dependencies:** Uses `python-dotenv`, `os`, `pathlib`. Exports `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `SEARXNG_BASE_URL`, `SD_*`, `CHROMA_PERSIST_DIR`, `CACHE_DIR`, `CACHE_TTL_SECONDS`, `OUTPUT_DIR`, `IMAGES_DIR`, `BLOGS_DIR`, `LOGS_DIR`.

**Contribution:** Single source of truth for all runtime parameters.

**Critical Bug — Missing `SD_SAFETY_CHECKER`:**
`tools/image_gen.py` imports `SD_SAFETY_CHECKER` from `app.config`, but this constant is not defined in `config.py`. This causes an `ImportError` at startup and will crash the entire application before any user interaction occurs.

**Improvements/Changes:**
1. **Add `SD_SAFETY_CHECKER`** — critical missing constant.
2. **File has large commented-out first-draft block** — remove the stale commented-out version to keep the file readable. The production version starts after ~40 lines of dead comments.
3. **`CHROMA_PERSIST_DIR` default diverges from `.env.example`** — config defaults to `./outputs/chroma`, but `.env.example` defaults to `./memory/chroma_data`. Align them.

**Updated Code (additions only — remove commented block, add constant):**
```python
# app/config.py — corrected version (active section only)
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")
SEARXNG_BASE_URL: str = os.getenv("SEARXNG_BASE_URL", "http://localhost:8080")

SD_MODEL_ID: str = os.getenv("SD_MODEL_ID", "CompVis/stable-diffusion-v1-4")
SD_DEVICE: str = os.getenv("SD_DEVICE", "cuda")
SD_IMAGE_WIDTH: int = int(os.getenv("SD_IMAGE_WIDTH", "512"))
SD_IMAGE_HEIGHT: int = int(os.getenv("SD_IMAGE_HEIGHT", "512"))
SD_INFERENCE_STEPS: int = int(os.getenv("SD_INFERENCE_STEPS", "30"))
# ADDED: missing constant imported by tools/image_gen.py
SD_SAFETY_CHECKER: bool = os.getenv("SD_SAFETY_CHECKER", "false").lower() == "true"

CHROMA_PERSIST_DIR: Path = Path(os.getenv("CHROMA_PERSIST_DIR", "./memory/chroma_data"))
CACHE_DIR: Path = Path(os.getenv("CACHE_DIR", "./memory/cache"))
CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "86400"))

OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "./outputs"))
IMAGES_DIR: Path = OUTPUT_DIR / "images"
BLOGS_DIR: Path = OUTPUT_DIR / "blogs"
LOGS_DIR: Path = OUTPUT_DIR / "logs"

for directory in [CHROMA_PERSIST_DIR, CACHE_DIR, IMAGES_DIR, BLOGS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
```

---

### File: `app/ui.py`

**Purpose:** Streamlit frontend — topic input, real-time progress display, blog preview, image gallery, download buttons.

**Logic and Dependencies:** Imports `compiled_graph`, `BLOGS_DIR`, `IMAGES_DIR`. Runs graph in a daemon `threading.Thread`, communicating with Streamlit via `queue.Queue`. Uses `st.session_state` for cross-render persistence.

**Contribution:** Primary user-facing interface. Correctly isolates the blocking graph execution from the Streamlit render loop.

**Improvements/Changes:**
1. **Large commented-out block** — same pattern as other files; remove the first draft.
2. **No moderation node in progress loop** — the UI progress handler has no case for `moderator_node`. If a moderator node is added (described in the prompt plan), the UI will silently drop its updates.
3. **`st.session_state.section_drafts` initialisation** — the session state dict is built inside the generate handler but must be pre-initialised before use to avoid `KeyError`.
4. **Streamlit `st.status` context** — the `with st.status(...)` block should call `status_box.update(state="complete")` on success and `state="error"` on failure to give users visual feedback.

---

---

## Folder: `graph/`

**Files:** `state.py`, `graph_builder.py`, `checkpointer.py`

---

### File: `graph/state.py`

**Purpose:** Defines all Pydantic models and the `GraphState` TypedDict that flows through the LangGraph pipeline. The single most important file in the project.

**Logic and Dependencies:** Imports `operator`, `typing`, `pydantic`. Exports `Section`, `BlogPlan`, `ResearchResult`, `SectionDraft`, `GeneratedImage`, `GraphState`. Uses `Annotated[List[...], operator.add]` for parallel-safe list accumulation.

**Contribution:** Architectural backbone — all nodes read/write against this schema. Correct use of `operator.add` as a reducer enables LangGraph's fan-in from parallel workers without race conditions.

**Improvements/Changes:**
1. **Large commented-out first-draft** — remove for cleanliness.
2. **`Section.image_prompt` is required** — but `planner.py`'s fallback `_dict_to_blog_plan` and `_fallback_parse_blog_plan` create `Section` objects using `section_title` and `content` kwargs, which don't exist in the schema. This will cause a `ValidationError` at runtime (see Planner analysis).
3. **`citation_registry` reducer missing** — `citation_registry` is a plain `Dict[str, str]` with no reducer. If multiple nodes attempted to write it concurrently (they don't currently, but it's fragile), the second write would overwrite the first. Consider wrapping with a merge reducer for safety.
4. **`[citation:10]` in docstring** — the `GraphState` docstring contains `[citation:10]` which is a leftover artifact from the generation process. Remove.

---

### File: `graph/graph_builder.py`

**Purpose:** Constructs and compiles the complete LangGraph StateGraph with all nodes, edges, fan-out dispatch functions, and fan-in barrier nodes.

**Logic and Dependencies:** Imports all node functions, `GraphState`, `get_checkpointer`. Defines `dispatch_researchers`, `dispatch_writers`, `dispatch_image_agents`, `route_after_planner`. Uses `Send()` for parallel worker dispatch. Compiles with `SqliteSaver`.

**Contribution:** The wiring layer — defines the exact execution order, parallelism strategy, and persistence contract for the entire pipeline.

**Critical Issues:**

1. **Mixed sync/async node registration** — `researcher_node` in `agents/researcher.py` is `async def`, but it is registered with `builder.add_node(RESEARCHER, researcher_node)` as a regular node. LangGraph *does* support async nodes, but the sync graph execution path (`compiled_graph.invoke()`) used by the Streamlit UI (via a background thread) will not properly `await` the async researcher. Either make `researcher_node` synchronous or switch to async graph execution throughout.

2. **`route_after_planner` returns string on no-research path** — when `research_required=False`, the function returns the string `RESEARCH_JOIN`. `add_conditional_edges` interprets a returned string as a node name to route to directly, which means it routes to `research_complete` and then immediately dispatches writers. This is the intended behaviour, but it's non-obvious and should be documented clearly.

3. **`dispatch_writers` passes full `**state`** — each writer Send includes the entire parent state dict including all `research_results`, `section_drafts`, `generated_images`, etc. For large runs this is memory-inefficient. Pass only what the writer needs.

4. **No error-propagation gate** — if `router_node` returns `{"error": "Topic rejected..."}`, the graph continues to `planner_node` because there is no conditional edge that checks for `state["error"]` after the router. A safety-rejected topic will still proceed to planning.

**Improvements/Changes — Updated `route_after_planner` with error gate:**
```python
# UPDATED: Add error gate after router, before planner
def route_after_router(state: GraphState) -> str:
    """Stop pipeline if router detected an error or safety rejection."""
    if state.get("error"):
        return END  # or a dedicated "error_node"
    return PLANNER

# In build_graph(), replace:
#   builder.add_edge(ROUTER, PLANNER)
# With:
#   builder.add_conditional_edges(ROUTER, route_after_router, {PLANNER: PLANNER, END: END})
```

---

### File: `graph/checkpointer.py`

**Purpose:** Factory functions (`get_checkpointer`, `get_async_checkpointer`) that create SQLite-backed LangGraph checkpointers for crash recovery and state persistence.

**Logic and Dependencies:** Imports `SqliteSaver`, `AsyncSqliteSaver`, `LOGS_DIR`. Wraps all initialization in structured error handling. Exports a custom `CheckpointerError` exception.

**Contribution:** Enables mid-run persistence — the graph can be resumed from the last successful checkpoint after a crash.

**Improvements/Changes:**
1. **Large commented-out first-draft** — remove.
2. **`AsyncSqliteSaver.from_conn_string` may not be awaitable in all versions** — in some langgraph-checkpoint-sqlite versions, `from_conn_string` is a classmethod (not a coroutine). Test the version actually installed.
3. **`get_async_checkpointer` is never called** — the graph is compiled synchronously in `graph_builder.py` using `get_checkpointer()`. The async version is dead code currently but is valuable for future migration. Keep but mark clearly.

---

---

## Folder: `agents/`

**Files:** `router.py`, `planner.py`, `researcher.py`, `writer.py`, `image_agent.py`, `citation_manager.py`, `reducer.py`

---

### File: `agents/router.py`

**Purpose:** LangGraph Router node. Classifies topic for `research_required` flag and safety check via LLM.

**Logic and Dependencies:** Reads `prompts/router_prompt.txt` via absolute path. Instantiates `ChatOllama`. Uses `_parse_llm_response()` with three fallback strategies (direct JSON → regex extraction → safe defaults). Returns `{"research_required": bool}` or `{"error": ..., "research_required": False}`.

**Contribution:** Pipeline entry point and safety gate.

**Improvements/Changes:**
1. **Large commented-out first-draft** — remove.
2. **`PROMPT_PATH` uses `Path(__file__).parent.parent`** — this is correct for the package structure. Good practice.
3. **LLM instantiated on every call** — `ChatOllama` is recreated each time `router_node` is called. Consider a module-level singleton or dependency injection for testability.
4. **`[citation:10]` artifact in `GraphState` docstring** — (tracked in state.py above).
5. **`_parse_llm_response` validates booleans well** — the type-checking in `validate()` is a good defensive pattern.

Overall, `router.py` is the cleanest and most production-ready agent in the codebase. Minor issues only.

---

### File: `agents/planner.py`

**Purpose:** LangGraph Planner node. Generates a structured `BlogPlan` via `ChatOllama.with_structured_output(BlogPlan)` with a robust fallback chain.

**Logic and Dependencies:** Reads planner prompt, queries ChromaDB for prior research context, invokes LLM with structured output. Three-tier fallback: structured output → raw JSON parse → default plan.

**Contribution:** Produces the `BlogPlan` that drives all parallel worker dispatch.

**Critical Bugs:**

1. **`_dict_to_blog_plan` uses wrong `Section` kwargs** — creates `Section(section_title=title, content=content)` but the `Section` Pydantic model has fields `id`, `title`, `description`, `word_count`, `search_query`, `image_prompt`. There is no `section_title` or `content` field. This will raise `pydantic.ValidationError` on every fallback parse, defeating the fallback entirely.

2. **`_fallback_parse_blog_plan` regex-section path uses same wrong kwargs** — same bug in the text extraction fallback: `Section(section_title=..., content=...)`.

3. **`_create_default_plan` uses same wrong kwargs** — the deepest fallback also fails. All three fallback tiers are broken.

4. **`RetryPolicy` import from `langgraph.pregel`** — `PLANNER_RETRY_POLICY` is defined but never used to decorate or wrap any function. LangGraph retry policies must be applied during node registration via `builder.add_node(..., retry=policy)`, not as a module-level constant that sits unused.

5. **Large commented-out first-draft** — remove.

**Updated `_dict_to_blog_plan` (corrected kwargs):**
```python
# UPDATED: Match actual Section schema fields
def _dict_to_blog_plan(data: Dict[str, Any], topic: str) -> BlogPlan:
    blog_title = data.get("blog_title") or data.get("title") or f"Blog post about {topic}"
    sections_data = data.get("sections", [])
    sections = []
    for i, sec_data in enumerate(sections_data, 1):
        if isinstance(sec_data, dict):
            sections.append(Section(
                id=sec_data.get("id", f"section_{i}"),
                title=sec_data.get("title", f"Section {i}"),
                description=sec_data.get("description", ""),
                word_count=int(sec_data.get("word_count", 400)),
                search_query=sec_data.get("search_query"),
                image_prompt=sec_data.get("image_prompt", ""),
            ))
    if not sections:
        sections = [Section(
            id="section_1", title="Introduction",
            description="Introduction to the topic.", word_count=400,
            search_query=None, image_prompt="",
        )]
    return BlogPlan(
        blog_title=blog_title,
        feature_image_prompt=data.get("feature_image_prompt", ""),
        sections=sections,
        research_required=bool(data.get("research_required", True)),
    )
```

**Updated `_create_default_plan` (corrected):**
```python
# UPDATED: Correct Section field names
def _create_default_plan(topic: str, research_required: bool) -> BlogPlan:
    logger.error("planner_node.using_default_plan", topic=topic)
    return BlogPlan(
        blog_title=f"Blog Post About {topic}",
        feature_image_prompt=f"Professional illustration representing {topic}",
        research_required=research_required,
        sections=[
            Section(id="section_1", title="Introduction",
                    description=f"Introduction to {topic}.", word_count=400,
                    search_query=topic if research_required else None,
                    image_prompt=f"Illustration of {topic}"),
            Section(id="section_2", title="Main Content",
                    description=f"Core concepts of {topic}.", word_count=400,
                    search_query=None, image_prompt=f"Diagram about {topic}"),
            Section(id="section_3", title="Conclusion",
                    description=f"Summary of {topic}.", word_count=300,
                    search_query=None, image_prompt=f"Conclusion visual"),
        ],
    )
```

**Apply `RetryPolicy` during node registration in `graph_builder.py`:**
```python
# In build_graph():
builder.add_node(PLANNER, planner_node, retry=RetryPolicy(max_attempts=3))
```

---

### File: `agents/researcher.py`

**Purpose:** Async LangGraph worker. Per section: checks cache, searches web, fetches pages concurrently, invokes LLM to summarize, stores in cache and ChromaDB.

**Logic and Dependencies:** Uses `asyncio.to_thread` for blocking I/O, `asyncio.gather` for concurrent page fetches, `llm.ainvoke`. Returns `{"research_results": [ResearchResult]}`.

**Contribution:** The data-gathering backbone of the research pipeline.

**Critical Bug — `fetch_page_content` signature mismatch:**
`researcher.py` calls `fetch_page_content(result.url, timeout=FETCH_TIMEOUT)` but `tools/web_fetcher.py` exports `fetch_page_content_sync(url: str, max_chars: int = 3000)`. The actual function name and parameter name are both wrong. This will raise `TypeError` on every research call.

**Fix:**
```python
# UPDATED: Correct import and call
from tools.web_fetcher import fetch_page_content_sync  # was: fetch_page_content

# In _fetch_content_items:
extracted_text = await asyncio.to_thread(
    fetch_page_content_sync, result.url  # removed incorrect timeout= kwarg
)
```

**Other Improvements:**
1. **Large commented-out first-draft** — remove.
2. The async design with `asyncio.gather` for concurrent page fetching is excellent — this is exactly the right pattern.
3. **`asyncio.to_thread` wrapping cache.get/set** — good practice to avoid blocking the event loop for disk I/O.

---

### File: `agents/writer.py`

**Purpose:** Dual-mode (sync + async) LangGraph worker that drafts one blog section from research context and a system prompt.

**Logic and Dependencies:** Reads writer prompt, extracts section-specific fields from state, finds matching `ResearchResult`, builds JSON user payload, invokes LLM, extracts `[SOURCE_N]` citation keys.

**Contribution:** The primary content generation node.

**Improvements/Changes:**
1. **Large commented-out first-draft** — remove.
2. **Two versions (`writer_node` + `writer_node_async`) with near-identical code** — extract shared logic into a single `_write_section(...)` function that takes an `invoke_fn` argument (sync or async), then each variant is a thin wrapper. This eliminates ~150 lines of duplication.
3. **Graph uses sync `writer_node` but researcher is async** — the graph should decide on a consistent async or sync execution model. If keeping researcher async, consider also using `writer_node_async` and running the graph via `.ainvoke()`.
4. **Validation for `section_description` treats empty string as missing** — `if not section_description` returns True for empty string, which may be a legitimate plan output for simple sections. Use `if section_description is None` instead.

---

### File: `agents/image_agent.py`

**Purpose:** Dual-mode (sync + async) LangGraph worker. Constructs output path, calls `generate_image`, returns `GeneratedImage`.

**Logic and Dependencies:** Imports `generate_image` from `tools.image_gen`, uses `IMAGES_DIR` from config.

**Contribution:** Visual asset generation for each blog section and the feature image.

**Critical Bug — Thread safety with SD singleton:**
`tools/image_gen.py`'s `generate_image` calls `asyncio.new_event_loop().run_until_complete(load_pipeline())` internally. Calling `asyncio.new_event_loop()` inside a function that is itself called from `asyncio.to_thread()` (in the async variant) or from a thread pool (via LangGraph's parallel dispatch) creates conflicting event loops. This will likely cause `RuntimeError: This event loop is already running` or similar issues.

**Improvements/Changes:**
1. **Large commented-out first-draft** — remove.
2. **Fix event loop creation in `generate_image`** — the pipeline should be loaded once at application startup before the graph runs, not lazily inside the generation function. Add a `preload_pipeline()` call in `ui.py` startup, and make `generate_image` access the already-loaded `_pipeline` directly.
3. **`image_agent_node` and `image_agent_node_async`** — same duplication issue as writer. Refactor.

---

### File: `agents/citation_manager.py`

**Purpose:** Pure Python node. Maps `[SOURCE_N]` citation markers from all section drafts to their corresponding URLs from research results.

**Logic and Dependencies:** No I/O, no LLM. Iterates drafts and research results, resolves indices. Returns `{"citation_registry": Dict[str, str]}`.

**Contribution:** Bridges the gap between LLM-generated citation placeholders and real URLs.

**Improvements/Changes:**
1. **Large commented-out first-draft** — remove.
2. **"Last write wins" for duplicate citation keys** — if two sections both use `[SOURCE_1]` and point to different URLs, only the last one is stored. The current behavior is documented but a warning log would make this more visible.
3. **This is one of the best-written files in the codebase** — clean separation of `_resolve_citation()` helper, comprehensive logging with statistics, solid guard clauses. No critical issues.

---

### File: `agents/reducer.py`

**Purpose:** Final assembler. Sorts section drafts, injects images and citations, builds full Markdown, converts to HTML, saves both files.

**Logic and Dependencies:** Uses `re.sub`, `markdown` library, `BLOGS_DIR`. Both sync and async variants.

**Contribution:** The output-producing node — creates the user-visible deliverable.

**Bug — Sections not separated by blank lines:**
The Markdown assembly is:
```python
full_md = (
    f"# {blog_plan.blog_title}\n\n"
    f"{feature_img_md}\n\n"
    f"{''.join(sections_md)}\n\n"  # ← sections joined with NO separator
    f"{refs}"
)
```
`sections_md` is a list like `["## Intro\n\ncontent...", "## Part 2\n\ncontent..."]`. Joining with `''.join()` produces valid Markdown only if each section already ends with `\n\n`, but that is not guaranteed. Should be `'\n\n'.join(sections_md)`.

**Improvements/Changes:**
1. **Fix section separator** — use `'\n\n'.join(sections_md)`.
2. **`codehilite` extension** — added in the sync version but listed in the plan as only `tables` and `fenced_code`. `codehilite` requires `Pygments`; if not installed, `markdown.markdown()` silently ignores it. Add `Pygments` to requirements or remove the extension.
3. **`nonlocal missing_citations`** — the `replace_citation` closure uses `nonlocal missing_citations` but Python 3.x requires the `nonlocal` keyword only if you *assign* to the variable. Since we use `+=`, this is correct, but works only in Python 3. Ensure `python_requires >= "3.10"` is enforced.
4. **Large commented-out first-draft** — remove.

**Updated assembly line:**
```python
# UPDATED: sections separated by double newline
full_md = (
    f"# {blog_plan.blog_title}\n\n"
    f"{feature_img_md}\n\n"
    f"{chr(10).join(sections_md)}\n\n"  # FIXED: was ''.join
    f"{refs}"
).strip()
```

---

---

## Folder: `tools/`

**Files:** `search.py`, `web_fetcher.py`, `image_gen.py`

---

### File: `tools/search.py`

**Purpose:** Unified async search abstraction with SearxNG primary and DuckDuckGo fallback. Returns `List[SearchResult]`.

**Logic and Dependencies:** `SearchResult` dataclass, `_search_searx` (with 3 retries + exponential backoff + timeout), `_search_duckduckgo` (with 2 retries). Lazy imports of LangChain wrappers to avoid startup cost.

**Contribution:** Decouples researcher from search backend — single `await search(query)` call regardless of which engine is active.

**Improvements/Changes:**
1. **Large commented-out first-draft** — remove.
2. **`_parse_duckduckgo_response` is URL-dependent** — the function assumes each line from DDG contains a URL. In practice, `DuckDuckGoSearchRun.run()` returns a string formatted as a numbered list or a block of text. The actual format has changed multiple times across DDG library versions. Consider using `DuckDuckGoSearchResults` (which returns structured JSON) instead of `DuckDuckGoSearchRun`.
3. **This is one of the better-designed tools** — the three-attempt SearxNG with exponential backoff and the clean fallback chain are exactly right.

---

### File: `tools/web_fetcher.py`

**Purpose:** Async HTTP + BeautifulSoup text extractor with retries. Also exports a sync wrapper `fetch_page_content_sync`.

**Logic and Dependencies:** `httpx.AsyncClient`, `BeautifulSoup`, full jitter backoff. Removes `<script>`, `<style>`, `<nav>`, etc. Extracts from semantic tags (`<main>`, `<article>`, `<p>`, headings). Falls back to `<body>`.

**Contribution:** Converts raw HTML from search results into clean text context for the researcher LLM.

**Critical Bug — Function Name Mismatch:**
The file exports `async_fetch_page_content` (async) and `fetch_page_content_sync` (sync wrapper). But `researcher.py` imports `fetch_page_content` (name that doesn't exist). This breaks the researcher node.

**Fix in `researcher.py`:**
```python
from tools.web_fetcher import fetch_page_content_sync
# ...
extracted_text = await asyncio.to_thread(fetch_page_content_sync, result.url)
```

**Improvements/Changes:**
1. **Large commented-out first-draft** — remove.
2. **`fetch_page_content_sync` uses `asyncio.run()`** — calling `asyncio.run()` inside a thread (as done via `asyncio.to_thread`) is fine because `to_thread` creates a new OS thread with no running event loop. However, if ever called from a coroutine directly on the main thread, it will fail with `RuntimeError: asyncio.run() cannot be called when another event loop is running`. Better to also expose a true `fetch_page_content_sync` that uses `httpx.Client` (synchronous) for reliability.
3. **Nested tag deduplication** — the anti-nesting check (`is_nested = any(parent.name in semantic_tags...)`) is a solid approach to prevent duplicate text from parent and child elements.

---

### File: `tools/image_gen.py`

**Purpose:** Loads Stable Diffusion v1.4 pipeline (singleton with async lock) and generates images with OOM recovery.

**Logic and Dependencies:** `diffusers.StableDiffusionPipeline`, `torch`, memory optimizations (attention slicing, VAE slicing, xformers). Exports `generate_image` (sync) and `generate_image_async`.

**Contribution:** All image generation for the blog pipeline.

**Critical Bugs:**

1. **`SD_SAFETY_CHECKER` import from `app.config`** — this constant is missing from `config.py` (fixed above). Until that fix is applied, this file will fail on import.

2. **`generate_image` creates a new event loop internally** — the sync `generate_image` calls:
   ```python
   loop = asyncio.new_event_loop()
   asyncio.set_event_loop(loop)
   pipeline = loop.run_until_complete(load_pipeline())
   loop.close()
   ```
   This is problematic when called from threads (which it will be via `asyncio.to_thread` in `image_agent_node_async`). If any thread still has an event loop set, this overwrites it. The correct pattern is to load the pipeline synchronously at startup.

3. **`time.sleep` is imported implicitly** — the OOM retry loop calls `time.sleep(delay)` but `import time` is missing from the file. This is a `NameError` that will crash on the first OOM retry.

4. **`load_pipeline` is async but `generate_image` is sync** — the architectural mismatch between an async loader and a sync generator is the root cause of bug #2. Refactor by making `load_pipeline` synchronous (since loading a large model is fundamentally a blocking operation).

**Updated `load_pipeline` (sync) + `generate_image` fixes:**
```python
# UPDATED: Synchronous pipeline loader (remove asyncio.Lock complexity)
import threading
import time  # ADDED: was missing

_pipeline = None
_load_lock = threading.Lock()

def load_pipeline() -> StableDiffusionPipeline:
    """Synchronous singleton pipeline loader (thread-safe)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _load_lock:
        if _pipeline is not None:  # double-check
            return _pipeline
        logger.info("loading_stable_diffusion_pipeline", model_id=SD_MODEL_ID, device=SD_DEVICE)
        torch_dtype = torch.float16 if SD_DEVICE == "cuda" else torch.float32
        pipeline = StableDiffusionPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=torch_dtype)
        pipeline = pipeline.to(SD_DEVICE)
        _enable_memory_optimizations(pipeline)
        _pipeline = pipeline
        logger.info("stable_diffusion_pipeline_loaded", model_id=SD_MODEL_ID, device=SD_DEVICE)
        return _pipeline

def generate_image(prompt, output_path, width=SD_IMAGE_WIDTH, height=SD_IMAGE_HEIGHT,
                   num_inference_steps=SD_INFERENCE_STEPS) -> str:
    pipeline = load_pipeline()  # FIXED: direct sync call, no asyncio.new_event_loop()
    # ... rest of retry loop unchanged
```

---

---

## Folder: `memory/`

**Files:** `cache.py`, `chroma_store.py`

---

### File: `memory/cache.py`

**Purpose:** Thread-safe disk cache with TTL, atomic writes, and async wrappers.

**Logic and Dependencies:** `CacheStore` class using `threading.RLock`, `tempfile`+`os.replace` for atomic writes, MD5 hashing of keys, JSON storage. Module-level `cache = CacheStore()` singleton.

**Contribution:** Prevents redundant SearxNG calls for identical queries within 24 hours. Significantly reduces latency on repeated topics.

**Improvements/Changes:**
1. **Large commented-out first-draft** — remove.
2. **`clear_expired` acquires lock inside loop on each file** — this is correct and prevents holding the lock too long. Good design.
3. **`json.JSONEncodeError`** — this exception does not exist; Python's `json` module raises `TypeError` for non-serializable objects. Should be `except (OSError, TypeError)`.
4. **`async_clear_expired` convenience method** — the async `run_in_executor` pattern is clean and consistent across all async wrappers.
5. **Overall, this is the most production-ready file in the codebase.**

---

### File: `memory/chroma_store.py`

**Purpose:** Thread-safe ChromaDB wrapper using `upsert` for idempotent research storage and `query` for similarity search.

**Logic and Dependencies:** `chromadb.PersistentClient`, `chromadb.errors.ChromaError`, `threading.RLock`. Graceful degraded mode if ChromaDB initialization fails (`self.collection = None`). Async wrappers via `run_in_executor`.

**Contribution:** Long-term memory — allows the Planner to reuse prior research context across runs.

**Improvements/Changes:**
1. **Large commented-out first-draft** — remove.
2. **`upsert` instead of `add`** — excellent choice over the first-draft's `add` (which would raise on duplicate IDs). `upsert` is idempotent and handles reruns gracefully.
3. **Degraded mode (`self.collection = None`)** — the `_check_collection()` guard prevents crashes if ChromaDB fails to initialize. This is production-grade defensive programming.
4. **`chromadb.errors.ChromaError` import** — this exception class exists in ChromaDB >= 0.4.x. Ensure version compatibility with the `chromadb>=0.5.0` requirement.
5. **Embeddings** — ChromaDB uses its default embedding function (all-MiniLM-L6-v2) when no embedding function is specified. For a local-only setup this is acceptable, but for consistency with the "fully local" constraint, consider specifying an Ollama-backed embedding function explicitly.

---

---

## Folder: `prompts/`

**Files:** `router_prompt.txt`, `planner_prompt.txt`, `researcher_prompt.txt`, `writer_prompt.txt`, `moderator_prompt.txt`

---

### File: `prompts/router_prompt.txt`

**Purpose:** System prompt for intent classification and safety check. Instructs model to output only a JSON object with `research_required` and `safe` keys.

**Contribution:** The "brain" behind the Router node's decisions.

**Improvements/Changes:**
1. **The prompt is clear and well-structured.**
2. **No few-shot examples** — adding 2-3 example topic → JSON pairs would improve output reliability for edge cases.
3. **"CRITICAL: You must respond with ONLY a valid JSON object"** — good explicit instruction. Consider also adding: "Do not include ```json markdown fences." Some models wrap JSON in code blocks even when told not to.

---

### File: `prompts/planner_prompt.txt`

**Purpose:** Instructs the model to produce a `BlogPlan` JSON matching the schema.

**Contribution:** Drives structured blog outline generation.

**Improvements/Changes:**
1. Include the exact JSON schema inline in the prompt (with field types and constraints).
2. Add explicit instruction: "All `id` values must follow the pattern `section_1`, `section_2`, etc. with no gaps."
3. For Mistral specifically, JSON mode prompting benefits from ending with: "Output:" to cue the model.

---

### File: `prompts/researcher_prompt.txt`

**Purpose:** Instructs model to synthesize a research summary from provided web content.

**Improvements/Changes:**
1. **Instruct model to prefer specific statistics** over general claims.
2. **Specify that `source_urls` should only contain URLs that were actually cited** in the summary — not all URLs provided.

---

### File: `prompts/writer_prompt.txt`

**Purpose:** Instructs model to write a Markdown blog section with citation markers and an image placeholder.

**Improvements/Changes:**
1. **`[IMAGE_PLACEHOLDER_{section_id}]`** — the placeholder uses the runtime variable `section_id`. The prompt must tell the model exactly what to substitute (e.g., "Replace `{section_id}` with the actual value from the input"). Otherwise the model may write the literal string `{section_id}`.
2. **Word count enforcement** — emphasise that the ±10% tolerance is strict; models tend to over-write.

---

### File: `prompts/moderator_prompt.txt`

**Purpose:** Final output safety and quality check before saving.

**Improvements/Changes:**
1. The moderation node (`moderator_node`) is described in the plan but **never implemented** in `agents/`. There is no `agents/moderator.py` and no `moderator_node` function. The prompt file exists but the node does not. This is a missing feature.
2. Add `moderator_node` to the implementation and wire it between `reducer_node` and `END` in `graph_builder.py`.

---

---

## Folder: `tests/`

**Files:** `test_router.py`, `test_planner.py`, `test_researcher.py`, `test_writer.py`, `test_graph_integration.py`, `fixtures/`

---

### File: `tests/test_router.py`

**Purpose:** Unit tests for `router_node` with mocked `ChatOllama`.

**Improvements/Changes:**
1. Tests use `@patch("agents.router.ChatOllama")` which patches at the import location — correct.
2. **Add a test for `test_llm_failure`** — mock `llm.invoke` to raise an exception and assert that the node returns `{"research_required": True, "error": ...}` gracefully (the node does handle this).
3. All four planned tests cover the key paths adequately.

---

### File: `tests/test_planner.py`

**Purpose:** Unit tests for `planner_node`.

**Critical Issue:**
`test_structured_output_fallback` forces `with_structured_output` to raise, then expects planner to fall back to raw JSON parsing. But as documented above, `_dict_to_blog_plan` uses wrong `Section` kwargs, so this test will fail once the planner bug is reproduced. Fix `planner.py` first.

**Improvements/Changes:**
1. Add a test for `test_chromadb_unavailable` — simulate ChromaDB failure and assert the planner continues without context.
2. Add a test for `test_empty_topic` — assert early-exit error dict returned.

---

### File: `tests/test_researcher.py`

**Purpose:** Unit tests for `researcher_node`.

**Critical Issue:**
`test_cache_hit_skips_search` and other tests call `researcher_node(base_state)` but `researcher_node` is `async def`. Tests must use `pytest.mark.asyncio` and `await researcher_node(base_state)`, or `asyncio.run(researcher_node(base_state))`.

**Updated test pattern:**
```python
import pytest

@pytest.mark.asyncio
async def test_cache_hit_skips_search(...):
    result = await researcher_node(base_state)
    ...
```

---

### File: `tests/test_writer.py`

**Purpose:** Unit tests for `writer_node`.

**Improvements/Changes:**
1. Tests call the sync `writer_node` — correct (sync version exists).
2. Add `test_missing_section_fields` — pass state without `section_id` and assert error dict returned.

---

### File: `tests/test_graph_integration.py`

**Purpose:** End-to-end integration tests that invoke the full compiled graph.

**Improvements/Changes:**
1. The `check_ollama_available` fixture correctly uses `pytest.skip()` — proper pattern.
2. **Tests will fail until the planner and researcher bugs are fixed** — integration tests are meaningless against a broken pipeline. Fix critical bugs first, then run integration tests.
3. Add `@pytest.mark.timeout(300)` to prevent hung tests on slow hardware.

---

---

## Docker Configuration

### File: `docker/Dockerfile`

**Purpose:** App container based on `python:3.11-slim`.

**Improvements/Changes:**
1. **`torch` is very large** — consider a multi-stage build or using `python:3.11-slim` with a pre-installed CUDA base image for GPU builds.
2. **Stable Diffusion model not pre-downloaded** — the first run inside Docker will download ~4GB. Add a `RUN python -c "from diffusers import StableDiffusionPipeline; ..."` step to bake the model into the image (with appropriate HF_HOME volume mapping).

---

### File: `docker/docker-compose.yml`

**Purpose:** Orchestrates app + Ollama + SearxNG as a three-service stack.

**Improvements/Changes:**
1. **`SEARXNG_BASE_URL=http://searxng:8080` override in `app` service** — correct; uses Docker service name as hostname.
2. **Ollama `depends_on`** — add `condition: service_healthy` with a health check for Ollama's `/api/tags` endpoint to prevent app startup before Ollama is ready.
3. **Missing Ollama model pull** — there is no step to `ollama pull mistral` inside the container. Add an entrypoint script or init container.

---

---

## Final Report

### Overall Architecture Summary

**Strengths:**
- The orchestrator–worker pattern with LangGraph's `Send()` API is correctly implemented. Parallel dispatch of researchers, writers, and image agents is architecturally sound.
- `operator.add` reducers for list-type state fields correctly solve the parallel-write merge problem.
- Disk-based JSON cache (`memory/cache.py`) and ChromaDB long-term store (`memory/chroma_store.py`) are well-designed with thread safety, atomic writes, and graceful degradation.
- The three-tier JSON fallback in `router_node` and `planner_node` is appropriate for LLM output reliability.
- Structured logging with `structlog` throughout is consistent and production-grade.
- The `prompts/` directory as externalised text files (not hardcoded strings) is a sound architectural decision enabling prompt iteration without code changes.
- Every file has both sync and async variants of key functions — forward-thinking for eventual async graph migration.

**Weaknesses:**

1. **Three critical import/name errors will crash the app on startup or first use:**
   - `app/config.py` missing `SD_SAFETY_CHECKER` (→ `tools/image_gen.py` crashes on import)
   - `tools/web_fetcher.py` exports `fetch_page_content_sync`, not `fetch_page_content` (→ researcher crashes)
   - `tools/image_gen.py` missing `import time` (→ `NameError` on first OOM retry)

2. **Planner fallback is completely broken** — all three tiers use wrong `Section` field names. Every non-perfect LLM response triggers a cascade of `ValidationError` exceptions.

3. **Async/sync inconsistency** — `researcher_node` is `async def` but the graph builder and Streamlit thread use the synchronous `.stream()` path. This creates a subtle execution model mismatch.

4. **Missing moderator agent** — the moderator prompt exists but `agents/moderator.py` was never created.

5. **No error propagation gate after router** — a safety-rejected topic continues to the planner.

6. **Every file contains a large commented-out first-draft block** — roughly doubling file sizes and making all files much harder to read. These should be removed before any team review or production deployment.

---

### List of All Proposed Changes

| # | File | Change | Priority |
|---|---|---|---|
| 1 | `requirements.txt` | Add `langgraph-checkpoint-sqlite`, `aiosqlite` | 🔴 Critical |
| 2 | `app/config.py` | Add `SD_SAFETY_CHECKER` constant | 🔴 Critical |
| 3 | `app/config.py` | Align `CHROMA_PERSIST_DIR` default with `.env.example` | 🟡 Medium |
| 4 | `tools/image_gen.py` | Add `import time`; fix `generate_image` to not create new event loop; make `load_pipeline` sync | 🔴 Critical |
| 5 | `agents/researcher.py` | Fix `fetch_page_content` → `fetch_page_content_sync`; add `pytest.mark.asyncio` to tests | 🔴 Critical |
| 6 | `agents/planner.py` | Fix all `Section()` instantiations in `_dict_to_blog_plan`, `_fallback_parse_blog_plan`, `_create_default_plan` to use correct field names | 🔴 Critical |
| 7 | `agents/planner.py` | Apply `RetryPolicy` via `builder.add_node(PLANNER, planner_node, retry=...)` | 🟡 Medium |
| 8 | `graph/graph_builder.py` | Add error gate after router: conditional edge checking `state["error"]` before proceeding to planner | 🟠 High |
| 9 | `graph/graph_builder.py` | Add `route_after_router` conditional edge | 🟠 High |
| 10 | `agents/reducer.py` | Fix `''.join(sections_md)` → `'\n\n'.join(sections_md)` | 🟠 High |
| 11 | `agents/moderator.py` | Create missing moderator agent and wire it into graph | 🟡 Medium |
| 12 | All files | Remove large commented-out first-draft blocks | 🟡 Medium |
| 13 | `tests/test_researcher.py` | Add `pytest.mark.asyncio` and `await` to all researcher test calls | 🟠 High |
| 14 | `tools/search.py` | Consider `DuckDuckGoSearchResults` over `DuckDuckGoSearchRun` for structured output | 🟢 Low |
| 15 | `memory/cache.py` | Fix `json.JSONEncodeError` → `TypeError` | 🟡 Medium |
| 16 | `docker/docker-compose.yml` | Add Ollama health check and model pull step | 🟡 Medium |
| 17 | `prompts/writer_prompt.txt` | Clarify `{section_id}` substitution instructions | 🟡 Medium |

---

### Recommended Next Steps

**Phase 1 — Fix Critical Bugs (before any testing)**
1. Apply fixes #1–6 (all 🔴 Critical items). The project cannot run in its current state without these.
2. Run `python -c "from graph.graph_builder import compiled_graph"` as a smoke test. This import exercises config, state, all agents, and the checkpointer.
3. Remove all commented-out first-draft blocks to reduce cognitive load.

**Phase 2 — Stabilise Architecture (1–2 days)**
4. Decide on sync vs. async execution model. Recommendation: go fully async — make `writer_node` and `image_agent_node` async, and run the graph via `.astream()` in the Streamlit thread. This fully unlocks the concurrent I/O benefits already built into `researcher_node`.
5. Add the error propagation gate (fix #8–9) and create `agents/moderator.py` (fix #11).
6. Fix the section separator in `reducer_node` (fix #10).

**Phase 3 — Test and Validate (1–2 days)**
7. Run unit tests: `pytest tests/ -m "not slow" -v`.
8. Fix `tests/test_researcher.py` async decorators.
9. With Ollama running: `pytest tests/ -m slow --timeout=300`.
10. Manually generate a blog with a time-sensitive topic (to trigger research path) and an evergreen topic (knowledge-only path).

**Phase 4 — Production Hardening**
11. Integrate Redis as a shared checkpointer (replacing SQLite) for multi-user Streamlit deployments.
12. Add LangSmith tracing (`LANGCHAIN_TRACING_V2=true`) for production observability.
13. Add `conftest.py` with shared fixtures to DRY up test boilerplate.
14. Add `notebooks/graph_explorer.ipynb` (described in plan, not present in zip).

---

### Revised Directory Tree (Proposed Additions)

```
blog-agent/
├── agents/
│   ├── moderator.py          ← NEW: missing moderation agent
│   └── ... (existing)
├── app/
│   └── ... (existing)
├── scripts/
│   └── preload_models.py     ← NEW: pre-download SD + Mistral at setup time
├── tests/
│   ├── conftest.py           ← NEW: shared fixtures and async event loop config
│   └── ... (existing)
└── ... (existing)
```

---

---

## APPENDIX A — Deep-Dive Source Code Analysis (Iterative File-by-File Review)

> **Date:** 2026-03-01  
> **Method:** Exhaustive depth-first read of every source file in the unzipped project.  
> **Goal:** Validate original analysis, discover new bugs, and confirm cross-module inconsistencies.

---

### A.1 Cross-Module Consistency Matrix

The following table tracks import/export mismatches discovered across modules — each one is a **runtime crash** waiting to happen.

| # | Consumer File | Imports | Provider File | Actually Exports | Severity |
|---|---|---|---|---|---|
| 1 | `agents/researcher.py:27` | `fetch_page_content` | `tools/web_fetcher.py` | `fetch_page_content_sync`, `async_fetch_page_content` | 🔴 **CRITICAL** — `ImportError` at import time |
| 2 | `app/ui.py` | `SEARXNG_URL` | `app/config.py` | `SEARXNG_BASE_URL` | 🔴 **CRITICAL** — `ImportError` at import time |
| 3 | `tools/image_gen.py:214` | `SD_SAFETY_CHECKER` | `app/config.py` | Not defined | 🔴 **CRITICAL** — `ImportError` at import time |
| 4 | `agents/researcher.py:108` | `search` (treats as sync) | `tools/search.py` | `search` (is async) | 🟡 **MAJOR** — `asyncio.to_thread(search, ...)` on a coroutine won't work |
| 5 | `agents/researcher.py:240` | `fetch_page_content(url, timeout=...)` | `tools/web_fetcher.py` | `fetch_page_content_sync(url, max_chars=...)` | 🟡 **MAJOR** — wrong function name AND wrong kwarg |
| 6 | `tools/image_gen.py:508` | `time.sleep(delay)` | stdlib | `import time` not present | 🔴 **CRITICAL** — `NameError` at OOM retry |

---

### A.2 File-by-File Deep Analysis

#### A.2.1 `graph/state.py` (224 lines) — ✅ Well-Structured

**Positive findings:**
- Pydantic `Field(...)` with validators (`gt=0` on `word_count`) — **good practice**.
- `Annotated[List[...], operator.add]` reducers on `research_results`, `section_drafts`, `generated_images` — correctly enables parallel worker merging.
- `default_factory=list` on `source_urls` and `citation_keys` — avoids mutable default footgun.

**Issues found:**
| # | Line | Issue | Fix |
|---|---|---|---|
| 1 | 122 | `image_prompt: str = Field(...)` is **required** — if LLM omits it, `BlogPlan` validation fails | Make it `Field(default="", ...)` |
| 2 | 198 | `[citation:10]` stray marker in docstring | Remove |
| 3 | 220 | `citation_registry: Dict[str, str]` has **no reducer** — if two nodes write to it, last-write-wins | Add `Annotated[..., merge_dicts]` or document that only `citation_manager_node` writes this field |

---

#### A.2.2 `agents/planner.py` (555 lines) — 🔴 CRITICAL BUGS

**Massive commented-out block:** Lines 1–195 (the entire first draft) are commented out. The active code starts at line 196.

**Critical bugs in fallback functions:**

The `Section` model requires these fields:
```python
Section(id=..., title=..., description=..., word_count=..., image_prompt=...)
```

But `planner.py` fallback functions use **wrong kwargs:**

| Function | Line | What It Does | Why It Fails |
|---|---|---|---|
| `_fallback_parse_blog_plan` | 466–468 | `Section(section_title=..., content=...)` | Both kwargs are **invalid** — `Section` has `title`, not `section_title`, and `description`, not `content`. Missing `id`, `word_count`, `image_prompt`. → `ValidationError` |
| `_fallback_parse_blog_plan` | 476 | `Section(section_title=..., content=...)` | Same as above |
| `_dict_to_blog_plan` | 506 | `Section(section_title=..., content=...)` | Same as above |
| `_dict_to_blog_plan` | 509 | `Section(section_title=..., content=...)` | Same as above |
| `_create_default_plan` | 533–544 | `Section(section_title=..., content=...)` | Same as above — every single default section will crash |

**All three fallback paths crash with `pydantic.ValidationError`.** The only path that works is when `llm.with_structured_output(BlogPlan)` succeeds on the first try.

**Fix:** Replace all `Section(section_title=..., content=...)` with:
```python
Section(
    id=f"section_{i}",
    title="...",
    description="...",
    word_count=300,
    image_prompt=""
)
```

**Other issues:**
- Line 206: `from langgraph.pregel import RetryPolicy` — `PLANNER_RETRY_POLICY` is defined but **never passed** to the graph builder's `add_node()` call.
- Line 319: `[citation:5]` stray marker in comment.

---

#### A.2.3 `agents/researcher.py` (320 lines) — 🔴 CRITICAL IMPORT BUG

**Critical bug (Line 27):**
```python
from tools.web_fetcher import fetch_page_content  # ← does NOT exist
```
`web_fetcher.py` exports `fetch_page_content_sync` and `async_fetch_page_content`. This is an `ImportError`.

**Major bug (Line 108):**
```python
search_results = await asyncio.to_thread(
    search, query=search_query, num_results=MAX_SEARCH_RESULTS
)
```
But `search()` in `tools/search.py` is `async def search(...)`. You cannot pass an async function to `asyncio.to_thread()` — it expects a sync callable. This will either fail silently (returning a coroutine object instead of results) or raise a `TypeError`.

**Fix:** Either `await search(query, num_results)` directly (since `researcher_node` is already async), or create a sync wrapper in `tools/search.py`.

**Minor bug (Line 240):**
```python
fetch_page_content, result.url, timeout=FETCH_TIMEOUT
```
Even if the import were fixed, `fetch_page_content_sync` takes `max_chars`, not `timeout`.

---

#### A.2.4 `agents/writer.py` (612 lines) — 🟡 Code Duplication

**Finding:** Lines 1–203 are a complete commented-out first draft. Active code starts at line 205.

**Code duplication:** `writer_node` (sync, lines 323–466) and `writer_node_async` (async, lines 472–603) share ~90% identical logic. Only the LLM call differs (`invoke` vs `ainvoke`). This violates DRY.

**Fix:** Extract shared logic into private helpers (already partially done with `_build_user_payload`, `_extract_citations`, etc.) and make the async version call those helpers.

**Minor issue (Line 358):** `if not section_description` treats `""` as missing, but a section might legitimately have no description yet. Consider `section_description is None` instead.

---

#### A.2.5 `agents/image_agent.py` (369 lines) — ✅ Clean (With Caveat)

Lines 1–112 are commented-out first draft. Active code starts at line 113.

**Thread-safety caveat (Lines 360–369):** The docstring correctly warns that `generate_image` may not be thread-safe. However, the async version uses `asyncio.to_thread(generate_image, ...)` which **will** run concurrent calls in threads. This is a ticking time bomb if multiple sections generate images in parallel.

**Fix:** Add a `threading.Lock()` or `asyncio.Semaphore(1)` to serialize image generation calls.

---

#### A.2.6 `agents/citation_manager.py` (277 lines) — ✅ Clean

Lines 1–139 are commented-out first draft. Active code starts at line 140.

Good points:
- Extracted `_resolve_citation()` for testability.
- Proper statistics tracking (`skipped_drafts`, `invalid_keys`, `out_of_range_keys`).
- Correctly notes it's CPU-bound and doesn't need an async version.

**Minor note:** "Last write wins" semantics for duplicate citation keys means if `[SOURCE_1]` appears in two different sections with different research results, only the last one's URL is kept. This is documented but could surprise users.

---

#### A.2.7 `agents/reducer.py` (619 lines) — 🟡 JOIN BUG

Lines 1–245 are commented-out first draft. Active code starts at line 246.

**Join bug (Line 414):**
```python
f"{''.join(sections_md)}\n\n"
```
Sections are joined with **no separator** (`''.join`). This means the end of one section's content runs directly into the next section's `## Heading`. The original commented-out version correctly used `"\n\n".join(sections_md)`.

**Fix:** `'\n\n'.join(sections_md)`

**The same bug exists in the async version (Line 570).**

**Dependency issue (Line 424):**
```python
extensions=['tables', 'fenced_code', 'codehilite']
```
The `codehilite` extension requires `Pygments` to be installed. `Pygments` is **not** in `requirements.txt`.

**Closure bug in async version (Lines 547–553):**
```python
def replace_citation(match: re.Match) -> str:
    ...
    missing_citations += 1  # ← references outer variable
```
The `replace_citation` closure inside the `for draft in sorted_drafts` loop uses `nonlocal` in the sync version (line 383) but the async version at line 552 does NOT use `nonlocal`, causing an `UnboundLocalError`.

---

#### A.2.8 `agents/router.py` (266 lines) — ✅ Clean, Production-Ready

Lines 1–100 are commented-out first draft. Active code starts at line 101.

**Positive findings:**
- `_parse_llm_response()` with `validate()` inner function — robust JSON parsing.
- Good defaults (`DEFAULT_RESEARCH_REQUIRED`, `DEFAULT_SAFE`).
- Proper error handling for empty topic and LLM failures.
- File-level prompt path via `Path(__file__).parent.parent / "prompts" / ...` — avoids CWD dependency.

**Note (Line 159):** `ChatOllama` is instantiated on every call. For performance, consider caching or creating it once at module level. However, this is not a bug since the model may need different config per invocation.

---

#### A.2.9 `tools/search.py` (561 lines) — ✅ Well-Designed

Lines 1–234 are commented-out first draft. Active code starts at line 236.

**Significant improvements over the original draft:**
- Added retry logic with exponential backoff for both SearxNG and DuckDuckGo.
- `asyncio.wait_for()` with configurable timeout.
- Input validation (empty query, num_results < 1).
- `_search_searx_sync` properly wraps the sync `.results()` method.

**DuckDuckGo parser (Lines 382–434):** The heuristic parser splits by newlines and extracts URLs — this is fragile and depends on `DuckDuckGoSearchRun`'s output format. Any changes to the LangChain tool could break it. Consider using `DuckDuckGoSearchResults` instead, which returns structured JSON. **Low priority** — the current approach works for the known format.

---

#### A.2.10 `tools/web_fetcher.py` (366 lines) — 🟡 Naming Issue

Lines 1–193 are commented-out first draft. Active code starts at line 194.

**Core design:** Clean async-first architecture with `async_fetch_page_content` as the primary function and `fetch_page_content_sync` as a sync wrapper using `asyncio.run()`.

**Critical naming mismatch:** `researcher.py` imports `fetch_page_content` but this file exports:
- `async_fetch_page_content` (async)
- `fetch_page_content_sync` (sync)

There is **no** `fetch_page_content` function exported.

**`asyncio.run()` issue (Line 366):**
```python
def fetch_page_content_sync(url: str, max_chars: int = 3000) -> str:
    return asyncio.run(async_fetch_page_content(url, max_chars))
```
`asyncio.run()` creates a new event loop. If called from within an existing async context (e.g., from `researcher_node` via `asyncio.to_thread`), this will work because `to_thread` runs in a separate thread. However, if called directly from an already-running event loop (not via `to_thread`), it will raise `RuntimeError: This event loop is already running`.

---

#### A.2.11 `tools/image_gen.py` (561 lines) — 🔴 MULTIPLE CRITICAL BUGS

Lines 1–174 are commented-out first draft. Active code starts at line 175.

**Bug 1 — Missing import (causes crash at line 508):**
```python
time.sleep(delay)  # ← 'time' is never imported
```
`import time` is not present in the active code block. This means the OOM retry path will crash with `NameError: name 'time' is not defined`.

**Bug 2 — Missing config constant (causes crash at import time):**
```python
from app.config import SD_SAFETY_CHECKER  # ← not defined in config.py
```
`SD_SAFETY_CHECKER` does not exist in `app/config.py`. This is an `ImportError`.

**Bug 3 — Event loop anti-pattern (Lines 434–437):**
```python
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
pipeline = loop.run_until_complete(load_pipeline())
loop.close()
```
`generate_image` is synchronous but calls the async `load_pipeline()` by creating a **new event loop**. This has three problems:
1. It **replaces** the current thread's event loop with a new one, potentially breaking other async code.
2. If called from within an existing event loop's thread, `loop.run_until_complete()` will fail.
3. The `loop.close()` at the end destroys the loop, making the `_load_lock` asyncio.Lock invalid for future calls.

**Bug 4 — Safety checker logic is a no-op (Line 313):**
```python
safety_checker = None if not SD_SAFETY_CHECKER else None
```
This evaluates to `None` regardless of `SD_SAFETY_CHECKER`'s value. It should be:
```python
safety_checker = None if not SD_SAFETY_CHECKER else ...  # keep default
```
Or simply:
```python
kwargs = {}
if not SD_SAFETY_CHECKER:
    kwargs["safety_checker"] = None
```

**Bug 5 — Async wrapper event loop (Lines 552–555):**
```python
async def generate_image_async(...):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(...)
```
Uses deprecated `get_event_loop()`. Should use `asyncio.get_running_loop()`.

---

#### A.2.12 `memory/cache.py` (545 lines) — 🟡 Minor Bug

Lines 1–129 are commented-out first draft. Active code starts at line 130.

**Bug (Line 378):**
```python
except (OSError, json.JSONEncodeError) as e:
```
`json.JSONEncodeError` does not exist. The correct exception is `TypeError` (raised when a non-serializable object is passed to `json.dump`) or more broadly `(TypeError, ValueError, OSError)`.

**Good practices:**
- Atomic writes via `tempfile.NamedTemporaryFile` + `os.replace`. ✅
- `threading.RLock()` for thread safety. ✅
- Async wrappers via `loop.run_in_executor`. ✅
- Input validation on payloads, with corrupted file cleanup. ✅

---

#### A.2.13 `memory/chroma_store.py` (502 lines) — ✅ Solid

Lines 1–120 are commented-out first draft. Active code starts at line 121.

**Positive findings:**
- Uses `upsert` instead of `add` — idempotent, no duplicate key errors. ✅
- Graceful degradation when collection is `None` (init failure). ✅
- `threading.RLock()` for thread safety. ✅
- `ChromaError` specific catching plus generic fallback. ✅
- Async wrappers provided. ✅

**Minor concern:** ChromaDB's default embedding function uses `all-MiniLM-L6-v2` which requires a network download on first run. This may conflict with the "fully local, no API keys" constraint. Consider pre-downloading or bundling the model in Docker.

---

#### A.2.14 `app/config.py` (122 lines) — 🟡 Missing Constants

Lines 1–68 are commented-out first draft. Active code starts at line 69+.

**Missing constant:** `SD_SAFETY_CHECKER` is imported by `tools/image_gen.py` but not defined here.

**Fix:** Add to `config.py`:
```python
SD_SAFETY_CHECKER: bool = os.getenv("SD_SAFETY_CHECKER", "true").lower() in ("true", "1", "yes")
```

**Also missing from `.env.example`:**
```
SD_SAFETY_CHECKER=true
```

---

#### A.2.15 `app/ui.py` (665 lines) — 🔴 Import Crash

**Critical bug:** Imports `SEARXNG_URL` from `app.config`, but `config.py` exports `SEARXNG_BASE_URL`.

**Other issues:**
- Large commented-out code block.
- `st.rerun()` after download panel may cause UI flicker.
- No handling for the missing `moderator_node` — UI references it but no implementation exists.

---

#### A.2.16 `graph/graph_builder.py` (652 lines) — 🟡 Missing Pieces

**Issues:**
- `router_node` is sync (`def router_node`) but registered in the graph directly. This is fine for LangGraph (it accepts both sync and async nodes). However, the comment at L264–266 of `router.py` suggests it was designed to be async.
- `dispatch_writers()` passes the **entire state** to each `Send()` call. Per LangGraph best practices, only the necessary fields should be passed to minimize memory overhead.
- No error propagation gate after the router — if `router_node` returns `{"error": "..."}`, the workflow continues to `planner_node` anyway.
- Missing `moderator_node` — referenced in prompts and UI but no implementation file exists.
- `PLANNER_RETRY_POLICY` defined in `planner.py` but never wired into `graph_builder.py`'s `add_node()` call.

---

#### A.2.17 `graph/checkpointer.py` — Confirmed Issues

- Contains commented-out first draft.
- `AsyncSqliteSaver.from_conn_string` may need `await` depending on LangGraph version.
- `get_async_checkpointer` is defined but unused anywhere in the project.

---

### A.3 Pattern: Dual-Code Architecture (Commented-Out First Draft)

Every single file in the project contains a **fully commented-out first draft** followed by blank lines and then the **active second version**. This pattern:

| File | Commented Lines | Active Lines |
|---|---|---|
| `router.py` | 1–100 | 101–266 |
| `planner.py` | 1–195 | 196–555 |
| `researcher.py` | — | 1–320 (only file without commented-out draft) |
| `writer.py` | 1–203 | 205–612 |
| `image_agent.py` | 1–112 | 113–369 |
| `citation_manager.py` | 1–139 | 140–277 |
| `reducer.py` | 1–245 | 246–619 |
| `search.py` | 1–234 | 236–561 |
| `web_fetcher.py` | 1–193 | 194–366 |
| `image_gen.py` | 1–174 | 175–561 |
| `cache.py` | 1–129 | 130–545 |
| `chroma_store.py` | 1–120 | 121–502 |
| `config.py` | 1–67 | 68–122 |
| `ui.py` | Large block | Mixed |
| `checkpointer.py` | Present | Present |
| `state.py` | 1–86 | 87–224 |
| `graph_builder.py` | Large block | Present |

**Recommendation:** Remove all commented-out code blocks. They add ~3000+ lines of dead code, making the project appear 2.5× larger than it is. If version history is needed, use Git.

---

### A.4 Consolidated Fix Priority List

#### 🔴 P0 — Import Crashes (Project Won't Start)

| # | File | Fix |
|---|---|---|
| 1 | `app/config.py` | Add `SD_SAFETY_CHECKER` constant |
| 2 | `app/ui.py` | Change `SEARXNG_URL` → `SEARXNG_BASE_URL` |
| 3 | `agents/researcher.py:27` | Change `from tools.web_fetcher import fetch_page_content` → `from tools.web_fetcher import fetch_page_content_sync as fetch_page_content` |
| 4 | `tools/image_gen.py` | Add `import time` |
| 5 | `requirements.txt` | Add `langgraph-checkpoint-sqlite`, `aiosqlite`, `Pygments` |
| 6 | `.env.example` | Add `SD_SAFETY_CHECKER=true` |

#### 🟠 P1 — Runtime Crashes (Crashes During Execution)

| # | File | Fix |
|---|---|---|
| 7 | `agents/planner.py` | Fix all `Section(section_title=..., content=...)` calls to use correct kwargs (`id`, `title`, `description`, `word_count`, `image_prompt`) |
| 8 | `agents/researcher.py:108` | Change `asyncio.to_thread(search, ...)` to `await search(...)` (search is async) |
| 9 | `agents/researcher.py:240` | Fix `fetch_page_content(url, timeout=...)` → `fetch_page_content(url)` or add `max_chars` kwarg |
| 10 | `tools/image_gen.py:313` | Fix safety checker no-op: `None if not SD_SAFETY_CHECKER else None` → proper conditional |
| 11 | `tools/image_gen.py:434-437` | Replace `asyncio.new_event_loop()` pattern with sync pipeline loading |
| 12 | `agents/reducer.py:414, 570` | Change `''.join(sections_md)` → `'\n\n'.join(sections_md)` |
| 13 | `agents/reducer.py:552` | Add `nonlocal missing_citations` in async version's `replace_citation` closure |
| 14 | `memory/cache.py:378` | Change `json.JSONEncodeError` → `TypeError` |

#### 🟡 P2 — Missing Features / Design Gaps

| # | File | Fix |
|---|---|---|
| 15 | `agents/moderator.py` | Create this file (referenced in prompts/UI but doesn't exist) |
| 16 | `graph/graph_builder.py` | Add error propagation gate after router |
| 17 | `graph/graph_builder.py` | Wire `PLANNER_RETRY_POLICY` into `add_node()` |
| 18 | `graph/state.py:122` | Make `image_prompt` optional: `Field(default="")` |
| 19 | `graph/graph_builder.py` | Optimize `Send()` payloads to only include needed fields |
| 20 | All files | Remove commented-out first-draft code blocks (~3000 lines) |

---

### A.5 Dependency Gap Analysis

| Package | Why It's Needed | In `requirements.txt`? |
|---|---|---|
| `langgraph-checkpoint-sqlite` | `graph/checkpointer.py` imports `SqliteSaver` | ❌ Missing |
| `aiosqlite` | `graph/checkpointer.py` imports `AsyncSqliteSaver` | ❌ Missing |
| `Pygments` | `agents/reducer.py` uses `codehilite` Markdown extension | ❌ Missing |
| `httpx` | `tools/web_fetcher.py` | ✅ Present |
| `beautifulsoup4` | `tools/web_fetcher.py` | ✅ Present |
| `torch` | `tools/image_gen.py` | ✅ Present |
| `diffusers` | `tools/image_gen.py` | ✅ Present |
| `chromadb` | `memory/chroma_store.py` | ✅ Present |
| `structlog` | Used everywhere | ✅ Present |

---

*Report generated by simulated multi-agent analysis: Traversal Agent → Analysis Agent → Improvement Agent → Architecture Agent → Reporting Agent.*  
*Appendix A added via iterative depth-first source code review on 2026-03-01.*
