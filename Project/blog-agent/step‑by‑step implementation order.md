Here’s a **step‑by‑step implementation order** that follows the project’s natural dependencies and the roadmap. Start with the core pipeline (single‑threaded), then add parallelism, UI, and finally advanced features.

---

## Phase 1 – Core Pipeline (Single‑Threaded MVP)

**Goal:** Get a working end‑to‑end system that processes one section at a time, with no parallelism.

### Step 1: Project Setup
- Create the folder structure exactly as shown.
- Create `requirements.txt` with all necessary libraries (LangGraph, LangChain, Ollama, Streamlit, diffusers, etc.).
- Set up a virtual environment and install dependencies.

### Step 2: Define State & Models – `graph/state.py`
- Write the `GraphState` TypedDict.
- Define Pydantic models: `Section`, `BlogPlan`, `ResearchResult`, `SectionDraft`, `ImageResult`.
- These will be used across all agents.

### Step 3: Create Prompt Templates (Placeholders)
- Create empty `.txt` files in `prompts/`:
  - `router_prompt.txt`
  - `planner_prompt.txt`
  - `researcher_prompt.txt`
  - `writer_prompt.txt`
  - `moderator_prompt.txt`
- Fill them later with actual prompts; for now just add a basic instruction comment.

### Step 4: Router Node – `agents/router.py`
- Implement `router_node(state)` that calls Ollama with the router prompt.
- Return `{"research_required": bool}`.
- Test with a few topics manually.

### Step 5: Planner Node – `agents/planner.py`
- Implement `planner_node(state)` that uses `model.with_structured_output(BlogPlan)` with the planner prompt.
- Return `{"plan": blog_plan}`.
- Add retry logic for malformed JSON.

### Step 6: Researcher Worker – `agents/researcher.py` + Tools
- Build `tools/search.py` with a function `search(query)` that uses SearxNG (or DuckDuckGo fallback).
- Build `tools/web_fetcher.py` with `fetch_content(url)` using httpx + BeautifulSoup.
- Build `memory/cache.py` for disk caching of search results (MD5 hash, 24h TTL).
- In `researcher.py`, write `research_worker(inputs)` that:
  - Takes a section, gets `search_query`, checks cache, searches, fetches, summarizes (optional LLM call with researcher prompt).
  - Returns a `ResearchResult`.

### Step 7: Writer Worker – `agents/writer.py`
- Implement `writer_worker(inputs)` that:
  - Receives a section and its corresponding `ResearchResult`.
  - Builds a prompt using the writer prompt, section plan, and research.
  - Calls Ollama to generate the section content, embedding `[citation]` markers and `[IMAGE_PLACEHOLDER_{id}]`.
  - Returns a `SectionDraft`.

### Step 8: Image Worker – `agents/image_agent.py` + `tools/image_gen.py`
- In `tools/image_gen.py`, write a function `generate_image(prompt, output_dir)` that loads Stable Diffusion v1.4 (once) and saves the image.
- In `image_agent.py`, implement `image_worker(inputs)` that calls the above and returns an `ImageResult` (file path, prompt, section id).

### Step 9: Citation Manager – `agents/citation_manager.py`
- Implement `citation_manager_node(state)` that:
  - Aggregates all `ResearchResult` objects.
  - Builds a registry of unique URLs.
  - Replaces `[citation]` markers in each section draft with numbered references like `[1]`.
  - Returns updated sections and the registry (store in state).

### Step 10: Reducer – `agents/reducer.py`
- Implement `reducer_node(state)` that:
  - Orders sections per the original plan.
  - Replaces image placeholders with actual Markdown image tags using the generated image paths.
  - Appends a references section from the citation registry.
  - Writes the final `.md` (and optionally `.html`) file to `outputs/blogs/`.
  - Returns `{"final_markdown": ..., "final_html": ..., "output_path": ...}`.

### Step 11: Graph Builder – `graph/graph_builder.py`
- Create a `StateGraph` with `GraphState`.
- Add all nodes as Python functions.
- Define edges: `router` → `planner` → `researcher` → `writer` → `citation_manager` → `reducer` → END.
- For now, use a simple loop to process each section sequentially (no `Send` yet).
- Compile the graph.

### Step 12: Test End‑to‑End
- Write a small script or use `notebooks/graph_explorer.ipynb` to invoke the graph with a sample topic.
- Verify that a complete blog post is generated with images and citations.

---

## Phase 2 – Parallelization & Production Core

**Goal:** Introduce LangGraph’s `Send()` API for parallel workers, checkpointing, and long‑term memory.

### Step 13: Refactor Graph for Parallel Dispatch – `graph/graph_builder.py`
- Replace sequential loops with `Send` edges:
  - After planner, create a function that returns `[Send("researcher", {"section": s}) for s in sections if s.search_query]`.
  - Similarly for writers and image workers.
- Ensure state keys use `Annotated[list, operator.add]` to merge parallel results.
- Test with multiple sections to confirm parallel execution.

### Step 14: Add Checkpointer – `graph/checkpointer.py`
- Implement `SqliteSaver` and pass it when compiling the graph.
- Simulate a crash mid‑run and verify recovery.

### Step 15: Long‑Term Memory – `memory/chroma_store.py`
- Implement ChromaDB client and functions to store/retrieve past blog plans and research.
- Integrate into the Planner node: before generating a new plan, query ChromaDB for similar topics and use the results to augment the prompt (optional).

### Step 16: Enhance Error Handling & Retries
- Add retry decorators to all worker nodes (exponential backoff).
- Implement fallback logic as described in the design doc (e.g., if researcher fails, writer proceeds with empty research and adds a warning).

---

## Phase 3 – UI & Observability

**Goal:** Build the Streamlit interface and add logging/tracing.

### Step 17: Streamlit UI – `app/ui.py` and `app/config.py`
- Create `config.py` to hold all configurable parameters (model endpoints, directories, etc.).
- In `ui.py`, build:
  - Topic input field.
  - Sidebar for settings (temperature, section count, etc.).
  - Start generation button.
  - Run the graph in a background thread and stream progress using `graph.stream()`.
  - Display section drafts as they complete.
  - Show final blog preview with images.
  - Download buttons for `.md` and `.html`.

### Step 18: Logging & Tracing
- Add `structlog` configuration to output JSON logs to `outputs/logs/`.
- Optionally enable LangSmith tracing by reading `LANGCHAIN_TRACING_V2` from environment.

### Step 19: Dockerize the Stack – `docker/`
- Write `Dockerfile` for the app.
- Write `docker-compose.yml` with services for app, Ollama, and SearxNG.
- Provide a `searxng/settings.yml` with sensible defaults.
- Test the entire stack on a clean machine.

---

## Phase 4 – Advanced Features (Optional)

**Goal:** Add polish and extensibility for portfolio demonstration.

### Step 20: Moderation & Human‑in‑the‑Loop
- Enhance Router with safety classification (use moderator prompt).
- Add `interrupt_before` in LangGraph to pause after Planner and show the outline in Streamlit for user approval.

### Step 21: Multi‑Model Support
- Allow switching between Mistral, Llama 3.1, Phi-3 via the Streamlit sidebar.
- Implement a factory function that returns the appropriate Ollama model.

### Step 22: Export & Social Media
- Add optional WordPress export via REST API.
- Generate Twitter/X thread and LinkedIn post from the final blog (as additional Reducer outputs).

---

## Testing Along the Way
- For each component, write corresponding unit tests in `tests/` (e.g., `test_router.py`, `test_planner.py`, etc.).
- Use `tests/fixtures/` to store mock search results for offline testing.
- After Phase 2, write an integration test `test_graph_integration.py` that runs the full graph on three standard topics and validates the output.

Follow this order, and you’ll have a robust, incrementally‑built system that’s easy to debug and extend.