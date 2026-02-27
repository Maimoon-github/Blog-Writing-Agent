## File-by-File Implementation Guide

---

### **app/ui.py**  
Streamlit frontend:  
- Input field for topic, optional sidebar config (temperature, section count, etc.).  
- Real‑time progress stream using `graph.stream()` (run graph in background thread, push updates to UI via queue).  
- Section‑by‑section preview as sections complete.  
- Final rendered Markdown blog with image gallery.  
- Download buttons for `.md` and `.html` files.  
- All logic tied to `config.py` settings.

---

### **app/config.py**  
Application‑level constants and settings:  
- Model names (Mistral, fallback models).  
- Ollama endpoint (`localhost:11434`).  
- SearxNG URL (`http://localhost:8080`).  
- Output directories (`outputs/images/`, `outputs/blogs/`, `outputs/logs/`).  
- Stable Diffusion parameters (image size, steps, guidance scale).  
- Cache TTL, ChromaDB collection name, etc.

---

### **agents/router.py**  
Router node implementation:  
- Function `router_node(state: GraphState) -> dict`:  
  - Uses `ChatOllama` with a safety + intent prompt (loaded from `prompts/router_prompt.txt`).  
  - Returns `{"research_required": bool}`.  
  - Also performs content safety check; if unsafe, sets an error flag or returns early.

---

### **agents/planner.py**  
Planner node:  
- Defines Pydantic models `Section`, `BlogPlan` (may import from `graph/state.py`).  
- Function `planner_node(state: GraphState) -> dict`:  
  - Calls Ollama with `model.with_structured_output(BlogPlan)` using prompt from `prompts/planner_prompt.txt`.  
  - Returns `{"plan": blog_plan}`.  
  - Handles retries/fallback if structured output fails.

---

### **agents/researcher.py**  
Researcher worker:  
- Function `research_worker(inputs: dict) -> dict`:  
  - Receives `section` (with `search_query`).  
  - Uses `tools/search.py` to perform search and `tools/web_fetcher.py` to fetch/extract content.  
  - Summarizes results using Ollama (prompt from `researcher_prompt.txt`) or extracts key facts.  
  - Returns `ResearchResult` (Pydantic model) containing summarized text and source URLs.  
- Includes retry logic and caching via `memory/cache.py`.

---

### **agents/writer.py**  
Writer worker:  
- Function `writer_worker(inputs: dict) -> dict`:  
  - Receives `section` and its `ResearchResult` (waits until available).  
  - Builds prompt with section plan, research context, and instructions to use `[citation]` markers.  
  - Calls Ollama (prompt from `writer_prompt.txt`).  
  - Returns `SectionDraft` (content string, section id).  
  - Embeds `[IMAGE_PLACEHOLDER_{id}]` for later image injection.

---

### **agents/image_agent.py**  
Image worker:  
- Function `image_worker(inputs: dict) -> dict`:  
  - Receives `section` (or feature image prompt) with `image_prompt`.  
  - Calls `tools/image_gen.py` to generate image.  
  - Returns `ImageResult` (file path, prompt used, section id).  
- Runs in parallel via `ThreadPoolExecutor`.

---

### **agents/citation_manager.py**  
Citation manager node:  
- Function `citation_manager_node(state: GraphState) -> dict`:  
  - Aggregates all `ResearchResult` objects from state.  
  - Builds a registry: mapping from citation key to URL and snippet.  
  - Iterates over all completed sections, replaces `[citation]` markers with numbered references like `[1]`.  
  - Returns updated sections and the registry.  
- Pure Python (no LLM).

---

### **agents/reducer.py**  
Reducer / assembler node:  
- Function `reducer_node(state: GraphState) -> dict`:  
  - Orders sections according to original plan.  
  - Replaces `[IMAGE_PLACEHOLDER_{id}]` with actual Markdown image syntax using paths from `generated_images`.  
  - Injects citations (using registry from citation manager).  
  - Appends references section.  
  - Writes final `.md` and `.html` files to `outputs/blogs/`.  
  - Optionally runs moderation check using `prompts/moderator_prompt.txt`.  
  - Returns `{"final_markdown": str, "final_html": str, "output_path": str}`.

---

### **graph/state.py**  
Core state and Pydantic models:  
- `GraphState(TypedDict)` with fields:  
  - `topic: str`  
  - `research_required: bool`  
  - `plan: Optional[BlogPlan]`  
  - `research_results: Annotated[list[ResearchResult], operator.add]`  
  - `completed_sections: Annotated[list[SectionDraft], operator.add]`  
  - `generated_images: Annotated[list[ImageResult], operator.add]`  
  - `citation_registry: dict`  
  - `final_markdown: str`, `final_html: str`, `output_path: str`  
- Pydantic models: `Section`, `BlogPlan`, `ResearchResult`, `SectionDraft`, `ImageResult`.

---

### **graph/graph_builder.py**  
LangGraph construction:  
- Create `StateGraph` with `GraphState`.  
- Add nodes: `router`, `planner`, `citation_manager`, `reducer`.  
- Add parallel worker nodes using `Send` edges:  
  - Conditional edge from planner to a function that returns `[Send("researcher", section) for section in plan.sections if section.search_query]` etc.  
  - Similar for writers and image workers.  
- Define edges: router → planner → parallel workers → citation_manager → reducer → end.  
- Compile graph with checkpointer from `checkpointer.py`.

---

### **graph/checkpointer.py**  
Checkpointer setup:  
- Return a `SqliteSaver` instance:  
  ```python
  from langgraph.checkpoint.sqlite import SqliteSaver
  checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
  ```

---

### **tools/search.py**  
Search abstraction:  
- Class `SearxNGSearch` (or function) that uses `langchain_community.utilities.SearxSearchWrapper` pointing to local SearxNG.  
- Fallback `DuckDuckGoSearchRun`.  
- Function `search(query: str) -> list[dict]` returning title, link, snippet.  
- Integrates with `memory/cache.py` to cache results.

---

### **tools/web_fetcher.py**  
Content fetching and parsing:  
- Function `fetch_web_content(url: str) -> str`: uses `httpx` to get HTML, `BeautifulSoup` to extract main text (e.g., article body).  
- Handles timeouts and errors gracefully.

---

### **tools/image_gen.py**  
Stable Diffusion pipeline:  
- Singleton loader: `load_sd_pipeline()` that initializes once.  
- Function `generate_image(prompt: str, output_dir: str, size=(512,512), **kwargs) -> str` (returns file path).  
- Uses `diffusers` with GPU/CPU fallback, reduced steps on CPU.  
- Saves image with deterministic filename (e.g., hash of prompt + timestamp).

---

### **memory/chroma_store.py**  
ChromaDB client and helpers:  
- Initialize Chroma client, get or create collection.  
- Functions:  
  - `add_blog_plan(plan: BlogPlan, research_results: list)` – store embeddings.  
  - `search_similar_topics(topic: str, k=3) – retrieve similar past plans.  
- Used by planner in Phase 2+.

---

### **memory/cache.py**  
Disk cache for search results:  
- Simple JSON cache with TTL (24h).  
- Functions:  
  - `get_cached_result(query: str) -> Optional[dict]`  
  - `cache_result(query: str, data: dict)`  
- Key = MD5 hash of query.

---

### **prompts/router_prompt.txt**  
System prompt for Router:  
- Instructs to classify topic and check safety. Expected output: JSON with `research_required` bool, and optionally `safe` bool.  
- Example few‑shot.

---

### **prompts/planner_prompt.txt**  
System prompt for Planner:  
- Describes how to create a blog outline with sections, image prompts, search queries.  
- Emphasizes structured JSON output matching `BlogPlan` schema.

---

### **prompts/researcher_prompt.txt**  
System prompt for research summarization:  
- Given raw search results, summarize key facts, statistics, quotes.  
- Output format can be plain text with source URLs listed.

---

### **prompts/writer_prompt.txt**  
System prompt for section writing:  
- Write a well‑structured section based on plan and research.  
- Use `[citation]` inline for facts from research.  
- Include `[IMAGE_PLACEHOLDER_{id}]` where an image should go.  
- Adhere to target word count.

---

### **prompts/moderator_prompt.txt**  
System prompt for final content moderation:  
- Review the full blog for any hallucinated claims or unsafe content.  
- Flag issues; optionally rewrite or add disclaimer.

---

### **outputs/** (directories)  
- `images/`: generated PNG files.  
- `blogs/`: final `.md` and `.html` files.  
- `logs/`: JSON logs from `structlog`.

---

### **tests/test_router.py**  
Unit tests for Router node:  
- Mock LLM responses, test classification logic, safety checks.

---

### **tests/test_planner.py**  
Unit tests for Planner node:  
- Test structured output parsing, retry logic, plan generation.

---

### **tests/test_researcher.py**  
Unit tests for Researcher worker:  
- Mock search and fetch, test summarization, error handling.

---

### **tests/test_writer.py**  
Unit tests for Writer worker:  
- Test section drafting, citation markers, image placeholders.

---

### **tests/test_graph_integration.py**  
End‑to‑end test:  
- Run graph on 3 predefined topics with mock services.  
- Verify final output contains expected sections, citations, images.

---

### **tests/fixtures/**  
Pre‑saved search results (JSON files) for offline testing.

---

### **docker/docker-compose.yml**  
Docker Compose for full stack:  
- Services: `app`, `ollama`, `searxng`.  
- Volumes for models and outputs.  
- GPU reservation for Ollama if available.

---

### **docker/Dockerfile**  
Dockerfile for the app:  
- FROM python:3.10-slim.  
- Copy requirements, install dependencies.  
- Copy entire project.  
- Expose Streamlit port (8501).  
- CMD to run `streamlit run app/ui.py`.

---

### **docker/searxng/settings.yml**  
SearxNG configuration:  
- Enable desired engines (google, duckduckgo, bing, wikipedia).  
- Set `search_format: json`.  
- Disable tracking, set secret key from environment.

---

### **notebooks/graph_explorer.ipynb**  
Jupyter notebook for interactive debugging:  
- Load graph, compile, invoke with sample topics.  
- Visualize state transitions and worker outputs.  
- Useful for development and experimentation.

---

### **requirements.txt**  
Python dependencies:  
- `langgraph`, `langchain-community`, `langchain-ollama`  
- `streamlit`, `pydantic`  
- `httpx`, `beautifulsoup4`  
- `diffusers`, `torch`, `transformers`  
- `chromadb`  
- `structlog`, `langsmith` (optional)  
- `markdown` (for HTML conversion)

---

### **.env.example**  
Template for environment variables:  
- `OLLAMA_BASE_URL=http://localhost:11434`  
- `SEARXNG_URL=http://localhost:8080`  
- `OUTPUT_DIR=./outputs`  
- `LOG_LEVEL=INFO`  
- `LANGCHAIN_TRACING_V2=false` (optional)

---

### **README.md**  
Project documentation:  
- Overview, architecture diagram (from design doc).  
- Quick‑start guide (installation, setup, run).  
- Phase roadmap.  
- Troubleshooting tips.  
- License info.