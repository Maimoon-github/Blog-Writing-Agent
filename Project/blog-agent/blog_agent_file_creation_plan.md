# Blog-Agent: Step-by-Step File Creation Plan

---

## Step 1 — `requirements.txt`

**Description:** Declares all Python dependencies with pinned versions required to run the entire system. Includes: `langgraph`, `langchain`, `langchain-community`, `langchain-ollama`, `ollama`, `diffusers`, `transformers`, `accelerate`, `torch`, `streamlit`, `chromadb`, `httpx`, `beautifulsoup4`, `pydantic`, `structlog`, `python-dotenv`, `duckduckgo-search`, `Pillow`, `markdown`, `pytest`, `pytest-asyncio`.

**Code-Generation Prompt:**
```
Generate a requirements.txt file for a Python project called blog-agent. Include the following packages with compatible pinned versions:
- langgraph>=0.2.0
- langchain>=0.2.0
- langchain-community>=0.2.0
- langchain-ollama>=0.1.0
- ollama>=0.2.0
- diffusers>=0.27.0
- transformers>=4.40.0
- accelerate>=0.30.0
- torch>=2.2.0
- streamlit>=1.35.0
- chromadb>=0.5.0
- httpx>=0.27.0
- beautifulsoup4>=4.12.0
- pydantic>=2.7.0
- structlog>=24.1.0
- python-dotenv>=1.0.0
- duckduckgo-search>=6.1.0
- Pillow>=10.3.0
- markdown>=3.6
- pytest>=8.2.0
- pytest-asyncio>=0.23.0

Output only the requirements.txt content with one package per line. No comments, no extras.
```

---

## Step 2 — `.env.example`

**Description:** Template file listing all required and optional environment variables with placeholder values and inline comments. Variables include: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `SEARXNG_BASE_URL`, `SD_MODEL_ID`, `CHROMA_PERSIST_DIR`, `OUTPUT_DIR`, `CACHE_DIR`, `LOG_DIR`, `CACHE_TTL_SECONDS`, `SD_DEVICE`, `SD_IMAGE_WIDTH`, `SD_IMAGE_HEIGHT`, `SD_INFERENCE_STEPS`, `LANGSMITH_API_KEY` (optional), `LANGSMITH_PROJECT` (optional).

**Code-Generation Prompt:**
```
Generate a .env.example file for a Python project called blog-agent. Include the following environment variables with placeholder values and a one-line comment above each:

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
SEARXNG_BASE_URL=http://localhost:8080
SD_MODEL_ID=CompVis/stable-diffusion-v1-4
CHROMA_PERSIST_DIR=./memory/chroma_data
OUTPUT_DIR=./outputs
CACHE_DIR=./memory/cache
LOG_DIR=./outputs/logs
CACHE_TTL_SECONDS=86400
SD_DEVICE=cuda  # or cpu
SD_IMAGE_WIDTH=512
SD_IMAGE_HEIGHT=512
SD_INFERENCE_STEPS=30
LANGSMITH_API_KEY=  # optional, leave blank to disable
LANGSMITH_PROJECT=blog-agent  # optional

Output only the .env.example content. No extra explanation.
```

---

## Step 3 — `app/config.py`

**Description:** Centralized configuration module that loads all environment variables via `python-dotenv` and exposes them as typed constants. Defines `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `SEARXNG_BASE_URL`, `SD_MODEL_ID`, `CHROMA_PERSIST_DIR`, `OUTPUT_DIR` (with subdirectories `images/`, `blogs/`, `logs/`), `CACHE_DIR`, `CACHE_TTL_SECONDS`, `SD_DEVICE`, `SD_IMAGE_WIDTH`, `SD_IMAGE_HEIGHT`, `SD_INFERENCE_STEPS`. Creates all output directory paths on import using `pathlib.Path.mkdir(parents=True, exist_ok=True)`.

**Code-Generation Prompt:**
```
Generate app/config.py for a Python project called blog-agent.

Requirements:
- Use python-dotenv to load a .env file from the project root.
- Import os and pathlib.Path.
- Define the following typed constants loaded from environment variables with sensible defaults:
  - OLLAMA_BASE_URL: str = "http://localhost:11434"
  - OLLAMA_MODEL: str = "mistral"
  - SEARXNG_BASE_URL: str = "http://localhost:8080"
  - SD_MODEL_ID: str = "CompVis/stable-diffusion-v1-4"
  - CHROMA_PERSIST_DIR: Path
  - OUTPUT_DIR: Path
  - IMAGES_DIR: Path = OUTPUT_DIR / "images"
  - BLOGS_DIR: Path = OUTPUT_DIR / "blogs"
  - LOGS_DIR: Path = OUTPUT_DIR / "logs"
  - CACHE_DIR: Path
  - CACHE_TTL_SECONDS: int = 86400
  - SD_DEVICE: str = "cuda"
  - SD_IMAGE_WIDTH: int = 512
  - SD_IMAGE_HEIGHT: int = 512
  - SD_INFERENCE_STEPS: int = 30
- After defining all Path constants, call .mkdir(parents=True, exist_ok=True) on each directory path.
- No classes, no functions — only module-level constants.
```

---

## Step 4 — `graph/state.py`

**Description:** Defines the complete shared data model for the LangGraph pipeline. Contains Pydantic models: `Section` (id, title, description, word_count, search_query, image_prompt), `BlogPlan` (blog_title, feature_image_prompt, sections, research_required), `ResearchResult` (section_id, query, summary, source_urls), `SectionDraft` (section_id, title, content, citation_keys), `GeneratedImage` (section_id, image_path, prompt). Defines `GraphState` as a `TypedDict` with fields: `topic: str`, `research_required: bool`, `blog_plan: BlogPlan | None`, `research_results: Annotated[List[ResearchResult], operator.add]`, `section_drafts: Annotated[List[SectionDraft], operator.add]`, `generated_images: Annotated[List[GeneratedImage], operator.add]`, `citation_registry: Dict[str, str]`, `final_blog_md: str`, `final_blog_html: str`, `run_id: str`, `error: str | None`.

**Code-Generation Prompt:**
```
Generate graph/state.py for a LangGraph-based Python project called blog-agent.

Requirements:
- Import: from typing import TypedDict, List, Dict, Optional, Annotated
- Import: from pydantic import BaseModel
- Import: import operator

Define these Pydantic BaseModel classes:

class Section(BaseModel):
    id: str
    title: str
    description: str
    word_count: int
    search_query: Optional[str] = None
    image_prompt: str

class BlogPlan(BaseModel):
    blog_title: str
    feature_image_prompt: str
    sections: List[Section]
    research_required: bool

class ResearchResult(BaseModel):
    section_id: str
    query: str
    summary: str
    source_urls: List[str]

class SectionDraft(BaseModel):
    section_id: str
    title: str
    content: str
    citation_keys: List[str]

class GeneratedImage(BaseModel):
    section_id: str
    image_path: str
    prompt: str

Then define GraphState as a TypedDict with:
- topic: str
- research_required: bool
- blog_plan: Optional[BlogPlan]
- research_results: Annotated[List[ResearchResult], operator.add]
- section_drafts: Annotated[List[SectionDraft], operator.add]
- generated_images: Annotated[List[GeneratedImage], operator.add]
- citation_registry: Dict[str, str]
- final_blog_md: str
- final_blog_html: str
- run_id: str
- error: Optional[str]

Use operator.add as the reducer for all three list fields so parallel worker outputs merge correctly.
```

---

## Step 5 — `prompts/router_prompt.txt`

**Description:** System prompt text file for the Router LLM node. Instructs the model to act as an intent classifier. Defines exact rules for setting `research_required` to true (topic contains time-sensitive keywords: latest, 2025, new, recent, current, trending, today; or topic is a factual/technical subject with evolving information) versus false (timeless concepts, historical topics, evergreen how-to content). Instructs the model to output only a valid JSON object with keys `research_required: bool` and `safe: bool`. Defines what makes a topic unsafe (harmful, illegal, or unethical content). Specifies that the model must not add any text outside the JSON object.

**Code-Generation Prompt:**
```
Generate the content for prompts/router_prompt.txt — a plain-text system prompt for a LangGraph Router node that uses an Ollama/Mistral LLM.

The prompt must instruct the model to:
1. Act as an intent classifier and safety checker.
2. Evaluate the user-provided blog topic.
3. Set research_required = true if the topic contains any of: "latest", "2025", "2024", "new", "recent", "current", "trending", "today", "this year", or is a factual/technical subject with fast-changing information (AI models, crypto, politics, geopolitics, scientific discoveries, software releases).
4. Set research_required = false if the topic is timeless, historical, or evergreen (e.g., how photosynthesis works, stoic philosophy, history of Rome).
5. Set safe = false if the topic involves harmful, illegal, hateful, or unethical content. Otherwise set safe = true.
6. Return ONLY a valid JSON object in this exact format with no extra text, no markdown, no explanation:
   {"research_required": true, "safe": true}

Output only the prompt text. No Python. No file headers.
```

---

## Step 6 — `prompts/planner_prompt.txt`

**Description:** System prompt text file for the Planner LLM node. Instructs the model to decompose the topic into a structured blog plan. Specifies that the output must be a valid JSON object matching the `BlogPlan` schema: blog_title, feature_image_prompt, research_required, and sections array (each with id, title, description, word_count of 300–500, search_query if research is needed, and image_prompt). Requires exactly 4–6 sections. Specifies that search queries must be concise web-search-style queries. Image prompts must be descriptive, photorealistic Stable Diffusion-compatible strings. Instructs the model to output only the raw JSON object with no markdown fences or explanation.

**Code-Generation Prompt:**
```
Generate the content for prompts/planner_prompt.txt — a plain-text system prompt for a LangGraph Planner node that uses an Ollama/Mistral LLM.

The prompt must instruct the model to:
1. Act as a strategic blog content planner.
2. Accept a blog topic and a research_required boolean flag.
3. Output a structured blog plan as a raw JSON object (no markdown, no explanation) matching this exact schema:
{
  "blog_title": "string",
  "feature_image_prompt": "string — detailed photorealistic SD prompt for the hero image",
  "research_required": true,
  "sections": [
    {
      "id": "section_1",
      "title": "string",
      "description": "string — 1-2 sentences describing what this section covers",
      "word_count": 400,
      "search_query": "string or null — concise web search query for this section",
      "image_prompt": "string — detailed photorealistic SD prompt for this section's image"
    }
  ]
}
4. Generate exactly 4 to 6 sections.
5. Set search_query to null for sections that don't need web research.
6. Word counts must be between 300 and 500 per section.
7. Section IDs must follow the pattern: section_1, section_2, etc.
8. Feature image prompt and section image prompts must be Stable Diffusion compatible: descriptive, photorealistic, specific visual style.
9. Output ONLY the raw JSON. No markdown fences, no commentary.

Output only the prompt text. No Python.
```

---

## Step 7 — `prompts/researcher_prompt.txt`

**Description:** System prompt text file for the Researcher worker LLM node. Instructs the model to act as a research analyst. Given a search query, a section description, and raw web content snippets (page title, URL, extracted text), the model must synthesize a factual research summary of 150–250 words. Must extract and list all real source URLs used. Must not fabricate facts not present in the provided content. Must flag if content is insufficient. Output must be a raw JSON object with keys: `summary` (string), `source_urls` (array of strings), `sufficient` (bool).

**Code-Generation Prompt:**
```
Generate the content for prompts/researcher_prompt.txt — a plain-text system prompt for a LangGraph Researcher worker node using an Ollama/Mistral LLM.

The prompt must instruct the model to:
1. Act as a research analyst.
2. Accept: a search_query string, a section_description string, and a list of web content items each containing: title, url, and extracted_text.
3. Synthesize a factual, dense research summary of 150 to 250 words based only on the provided content.
4. Never fabricate facts, statistics, or quotes not present in the input content.
5. List every real URL used as a source.
6. Set sufficient = false if the content is too thin to write a proper section; otherwise set sufficient = true.
7. Output ONLY a raw JSON object with this schema (no markdown, no explanation):
{
  "summary": "string",
  "source_urls": ["url1", "url2"],
  "sufficient": true
}

Output only the prompt text. No Python.
```

---

## Step 8 — `prompts/writer_prompt.txt`

**Description:** System prompt text file for the Writer worker LLM node. Instructs the model to act as a professional blog writer. Given section title, description, target word count, research summary, and source URLs, the model must write the full section body in Markdown. Must embed inline citation markers as `[SOURCE_N]` immediately after any fact or statistic drawn from research. Must embed exactly one `[IMAGE_PLACEHOLDER_{section_id}]` token at the most visually appropriate point in the section. Must match the target word count within ±10%. Must not add a top-level H1 heading (section title is handled externally). Output is only the raw Markdown section body.

**Code-Generation Prompt:**
```
Generate the content for prompts/writer_prompt.txt — a plain-text system prompt for a LangGraph Writer worker node using an Ollama/Mistral LLM.

The prompt must instruct the model to:
1. Act as a professional blog writer with expertise in SEO content.
2. Accept: section_title, section_description, target_word_count, research_summary, source_urls (a list), section_id.
3. Write the full section body in Markdown format.
4. Do NOT include a top-level H1 heading — start directly with the content (use H2 or H3 subheadings as appropriate within the section).
5. Embed inline citation markers in the format [SOURCE_N] (where N starts at 1) immediately after any fact, statistic, or claim drawn from the research.
6. Embed exactly one image placeholder token in the format [IMAGE_PLACEHOLDER_{section_id}] at the most visually relevant point in the section body.
7. Write to match the target_word_count within plus or minus 10 percent.
8. Use clear, engaging, professional prose. Avoid filler phrases.
9. Output ONLY the raw Markdown content. No JSON, no explanation, no preamble.

Output only the prompt text. No Python.
```

---

## Step 9 — `prompts/moderator_prompt.txt`

**Description:** System prompt text file for the final output moderation LLM node. Instructs the model to evaluate the assembled blog content for safety, factual integrity markers, and quality. Checks for: harmful/offensive content, hallucinated citation patterns (`[SOURCE_N]` markers with no corresponding URL), placeholder tokens that were never replaced, empty sections. Outputs a raw JSON object with keys: `approved: bool`, `issues: List[str]`, `flagged_sections: List[str]`. If approved is false, issues must describe each specific problem found.

**Code-Generation Prompt:**
```
Generate the content for prompts/moderator_prompt.txt — a plain-text system prompt for a LangGraph moderation node using an Ollama/Mistral LLM.

The prompt must instruct the model to:
1. Act as a content safety and quality moderator.
2. Accept the fully assembled blog post as a Markdown string.
3. Check for the following issues:
   a. Harmful, hateful, offensive, or illegal content.
   b. Any [SOURCE_N] citation markers that appear without a matching URL in the citation registry.
   c. Any [IMAGE_PLACEHOLDER_*] tokens that were never replaced with an image.
   d. Sections that are unusually short (under 100 words) suggesting incomplete generation.
   e. Repeated or duplicated paragraphs indicating a generation loop error.
4. Set approved = true only if none of the above issues are found.
5. List every specific issue found in the issues array. If no issues, issues = [].
6. List section titles of any flagged sections in flagged_sections. If none, flagged_sections = [].
7. Output ONLY a raw JSON object with this schema (no markdown, no explanation):
{
  "approved": true,
  "issues": [],
  "flagged_sections": []
}

Output only the prompt text. No Python.
```

---

## Step 10 — `tools/search.py`

**Description:** Provides a unified `search(query: str, num_results: int = 5) -> List[SearchResult]` interface. `SearchResult` is a dataclass with fields: `title: str`, `url: str`, `snippet: str`. Primary backend uses `SearxSearchWrapper` from `langchain_community` pointing to `SEARXNG_BASE_URL` from config. Fallback uses `DuckDuckGoSearchRun` from `langchain_community`. Implements automatic fallback: if SearxNG raises an exception or returns zero results, retries with DuckDuckGo. Logs which backend was used via `structlog`. Returns an empty list (never raises) on total failure.

**Code-Generation Prompt:**
```
Generate tools/search.py for a Python project called blog-agent.

Requirements:
- Define a dataclass SearchResult with fields: title: str, url: str, snippet: str.
- Define an async-compatible function: search(query: str, num_results: int = 5) -> List[SearchResult].
- Primary search backend: use langchain_community.utilities.SearxSearchWrapper initialized with searx_host from app.config.SEARXNG_BASE_URL. Call .results(query, num_results=num_results) to get results. Parse each result dict for keys: "title", "link" (use as url), "snippet".
- Fallback backend: use langchain_community.tools.DuckDuckGoSearchRun. Call .run(query) which returns a plain string. Parse the string into SearchResult objects (split by newlines, extract URL with a regex).
- Logic: Try SearxNG first. If it raises any exception or returns an empty list, log the fallback trigger using structlog and try DuckDuckGo.
- If both fail, log the error and return an empty list — never raise.
- Import SEARXNG_BASE_URL from app.config.
- Use structlog.get_logger(__name__) for logging.
- No classes beyond SearchResult. Only the search() function and SearchResult dataclass.
```

---

## Step 11 — `tools/web_fetcher.py`

**Description:** Provides a `fetch_page_content(url: str, max_chars: int = 3000) -> str` function. Uses `httpx` with a 10-second timeout and a standard browser User-Agent header to GET the URL. Passes the response HTML to `BeautifulSoup` with `html.parser`. Removes all `<script>`, `<style>`, `<nav>`, `<footer>`, `<header>`, and `<aside>` tags. Extracts visible text from `<p>`, `<article>`, `<main>`, and `<section>` tags. Joins, strips, and truncates to `max_chars`. Returns empty string on any HTTP or parsing error. Logs errors via `structlog`.

**Code-Generation Prompt:**
```
Generate tools/web_fetcher.py for a Python project called blog-agent.

Requirements:
- Import: httpx, BeautifulSoup from bs4, structlog, re.
- Define one function: fetch_page_content(url: str, max_chars: int = 3000) -> str.
- Use httpx.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0 ..."}) inside a try/except block.
- Pass response.text to BeautifulSoup with "html.parser".
- Use soup.decompose() pattern to remove all <script>, <style>, <nav>, <footer>, <header>, <aside> tags.
- Extract visible text by finding all <p>, <article>, <main>, <section> tags and joining their .get_text(separator=" ", strip=True).
- Collapse multiple whitespace characters into a single space using re.sub.
- Truncate the result to max_chars characters.
- Return empty string "" on any exception (httpx.RequestError, httpx.HTTPStatusError, Exception).
- Log each error with structlog.get_logger(__name__).error(...) including the url and error message.
- No classes. Only the fetch_page_content function.
```

---

## Step 12 — `tools/image_gen.py`

**Description:** Manages the Stable Diffusion v1.4 image generation pipeline. Implements `load_pipeline() -> StableDiffusionPipeline` which loads `CompVis/stable-diffusion-v1-4` using `diffusers`, applies `torch.float16` precision on CUDA or `float32` on CPU, enables `enable_attention_slicing()` for memory efficiency, and caches the loaded pipeline as a module-level singleton to avoid reloading. Implements `generate_image(prompt: str, output_path: str, width: int, height: int, num_inference_steps: int) -> str` which calls the pipeline, saves the resulting PIL image as PNG to `output_path`, and returns the path. Handles OOM by catching `torch.cuda.OutOfMemoryError` and retrying at half resolution.

**Code-Generation Prompt:**
```
Generate tools/image_gen.py for a Python project called blog-agent.

Requirements:
- Import: from diffusers import StableDiffusionPipeline, torch, structlog, os, pathlib.
- Import SD_MODEL_ID, SD_DEVICE, SD_IMAGE_WIDTH, SD_IMAGE_HEIGHT, SD_INFERENCE_STEPS from app.config.
- Define a module-level variable _pipeline = None for singleton caching.

Define load_pipeline() -> StableDiffusionPipeline:
- If _pipeline is not None, return it immediately.
- Load StableDiffusionPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=torch.float16 if SD_DEVICE=="cuda" else torch.float32).
- Call pipeline.to(SD_DEVICE).
- Call pipeline.enable_attention_slicing().
- Set _pipeline = pipeline.
- Return _pipeline.

Define generate_image(prompt: str, output_path: str, width: int = SD_IMAGE_WIDTH, height: int = SD_IMAGE_HEIGHT, num_inference_steps: int = SD_INFERENCE_STEPS) -> str:
- Call load_pipeline() to get the pipeline.
- Try: result = pipeline(prompt=prompt, width=width, height=height, num_inference_steps=num_inference_steps).
- Save result.images[0] as PNG to output_path using pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True) then image.save(output_path).
- Catch torch.cuda.OutOfMemoryError: log warning, retry with width=width//2, height=height//2.
- Return output_path as string.
- Log generation start and completion with structlog.
```

---

## Step 13 — `memory/cache.py`

**Description:** Implements a disk-based JSON cache for search results with a 24-hour TTL. `CacheStore` class initialized with `cache_dir: Path` from config. Method `get(key: str) -> dict | None`: computes `hashlib.md5(key.encode()).hexdigest()` as filename, reads the JSON file if it exists, checks if `timestamp` field is within `CACHE_TTL_SECONDS`, returns `data` field or None on miss/expiry. Method `set(key: str, value: dict) -> None`: writes JSON file with `{"timestamp": time.time(), "data": value}`. Method `delete(key: str) -> None`: removes the cache file if it exists. Method `clear_expired() -> int`: iterates all `.json` files in cache_dir, deletes expired ones, returns count deleted.

**Code-Generation Prompt:**
```
Generate memory/cache.py for a Python project called blog-agent.

Requirements:
- Import: hashlib, json, time, pathlib, structlog.
- Import CACHE_DIR, CACHE_TTL_SECONDS from app.config.

Define class CacheStore:
  __init__(self, cache_dir: Path = CACHE_DIR):
    - self.cache_dir = Path(cache_dir)
    - self.cache_dir.mkdir(parents=True, exist_ok=True)
    - self.ttl = CACHE_TTL_SECONDS
    - self.logger = structlog.get_logger(__name__)

  _key_to_path(self, key: str) -> Path:
    - Return self.cache_dir / (hashlib.md5(key.encode()).hexdigest() + ".json")

  get(self, key: str) -> dict | None:
    - path = self._key_to_path(key)
    - If path does not exist, return None.
    - Read and parse the JSON file. If parsing fails, return None.
    - If time.time() - data["timestamp"] > self.ttl, delete the file and return None.
    - Return data["data"].

  set(self, key: str, value: dict) -> None:
    - path = self._key_to_path(key)
    - Write JSON: {"timestamp": time.time(), "data": value}.
    - Log cache write with structlog.

  delete(self, key: str) -> None:
    - path = self._key_to_path(key)
    - If path exists, call path.unlink().

  clear_expired(self) -> int:
    - Iterate all .json files in self.cache_dir.
    - Parse each. If expired or unreadable, delete.
    - Return count of deleted files.

At module level, define: cache = CacheStore() as a shared singleton.
```

---

## Step 14 — `memory/chroma_store.py`

**Description:** Wraps ChromaDB for storing and retrieving research summaries. `ChromaStore` class initialized with `persist_directory` from config and collection name `blog_research`. Method `add_research(section_id: str, query: str, summary: str, source_urls: List[str]) -> None`: adds the summary as a document with metadata `{section_id, query, source_urls_json}` and ID derived from `section_id`. Method `search_similar(query: str, n_results: int = 3) -> List[dict]`: queries the collection with the query string, returns list of dicts with `summary`, `source_urls`, `section_id` from metadata. Method `clear() -> None`: deletes and recreates the collection. Uses `chromadb.PersistentClient`.

**Code-Generation Prompt:**
```
Generate memory/chroma_store.py for a Python project called blog-agent.

Requirements:
- Import: chromadb, json, structlog, List from typing.
- Import CHROMA_PERSIST_DIR from app.config.

Define class ChromaStore:
  __init__(self, persist_directory: str = str(CHROMA_PERSIST_DIR), collection_name: str = "blog_research"):
    - self.client = chromadb.PersistentClient(path=persist_directory)
    - self.collection = self.client.get_or_create_collection(name=collection_name)
    - self.logger = structlog.get_logger(__name__)

  add_research(self, section_id: str, query: str, summary: str, source_urls: List[str]) -> None:
    - Call self.collection.add(
        documents=[summary],
        metadatas=[{"section_id": section_id, "query": query, "source_urls_json": json.dumps(source_urls)}],
        ids=[f"research_{section_id}"]
      )
    - Log the add operation.

  search_similar(self, query: str, n_results: int = 3) -> List[dict]:
    - Call self.collection.query(query_texts=[query], n_results=n_results).
    - Parse results. For each result, return a dict with: summary (from documents), section_id, query, source_urls (parsed from source_urls_json metadata).
    - Return the list. Return empty list on any exception.

  clear(self) -> None:
    - Delete and recreate the collection.
    - Log the clear.

At module level, define: chroma_store = ChromaStore() as a shared singleton.
```

---

## Step 15 — `agents/router.py`

**Description:** LangGraph Router node function. Signature: `router_node(state: GraphState) -> dict`. Loads the system prompt from `prompts/router_prompt.txt`. Constructs a `ChatOllama` instance using `OLLAMA_MODEL` and `OLLAMA_BASE_URL` from config. Invokes the LLM with the system prompt and user message containing `state["topic"]`. Parses the JSON response to extract `research_required` and `safe`. If `safe` is false, returns `{"error": "Topic rejected by safety check"}`. Returns `{"research_required": research_required}`. Wraps JSON parsing in try/except with fallback to regex extraction. Logs the routing decision.

**Code-Generation Prompt:**
```
Generate agents/router.py for a LangGraph-based Python project called blog-agent.

Requirements:
- Import: json, re, pathlib, structlog.
- Import: ChatOllama from langchain_ollama.
- Import: HumanMessage, SystemMessage from langchain_core.messages.
- Import: GraphState from graph.state.
- Import: OLLAMA_MODEL, OLLAMA_BASE_URL from app.config.

Define function router_node(state: GraphState) -> dict:

1. Read the system prompt from prompts/router_prompt.txt using pathlib.Path.
2. Instantiate ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL).
3. Call llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=f"Topic: {state['topic']}")]).
4. Extract response.content as a string.
5. Try to parse the string as JSON using json.loads(). 
   - If json.loads fails, use re.search to extract the JSON object pattern \{.*?\} with re.DOTALL flag and retry json.loads on that match.
   - If still fails, default to {"research_required": True, "safe": True}.
6. If parsed_result.get("safe", True) is False, return {"error": "Topic rejected by safety filter", "research_required": False}.
7. Log the routing decision (research_required value) using structlog.get_logger(__name__).
8. Return {"research_required": parsed_result.get("research_required", True)}.
```

---

## Step 16 — `agents/planner.py`

**Description:** LangGraph Planner node function. Signature: `planner_node(state: GraphState) -> dict`. Reads `prompts/planner_prompt.txt`. Instantiates `ChatOllama` and calls `.with_structured_output(BlogPlan)` to enforce Pydantic output. Falls back to raw JSON parsing if structured output fails. Constructs the user message including `state["topic"]` and `state["research_required"]`. Returns `{"blog_plan": blog_plan_instance}`. Logs the generated plan title and section count. Checks ChromaDB for similar prior research using `chroma_store.search_similar(state["topic"])` and includes any hits in the prompt context.

**Code-Generation Prompt:**
```
Generate agents/planner.py for a LangGraph-based Python project called blog-agent.

Requirements:
- Import: json, re, pathlib, structlog.
- Import: ChatOllama from langchain_ollama.
- Import: HumanMessage, SystemMessage from langchain_core.messages.
- Import: GraphState, BlogPlan, Section from graph.state.
- Import: OLLAMA_MODEL, OLLAMA_BASE_URL from app.config.
- Import: chroma_store from memory.chroma_store.

Define function planner_node(state: GraphState) -> dict:

1. Read system prompt from prompts/planner_prompt.txt.
2. Query chroma_store.search_similar(state["topic"], n_results=2) to get prior_research list.
3. Build a context string from prior_research (if non-empty): "Prior research context:\n" + each summary concatenated.
4. Instantiate llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL).
5. Try structured output: structured_llm = llm.with_structured_output(BlogPlan).
6. Build user message: f"Topic: {state['topic']}\nResearch required: {state['research_required']}\n{context_string}"
7. Try to call structured_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_message)]).
   - If that raises any exception, fall back to llm.invoke(...).content, parse raw JSON using json.loads with regex fallback, then manually construct BlogPlan and Section objects from the parsed dict.
8. Log blog_plan.blog_title and len(blog_plan.sections) using structlog.
9. Return {"blog_plan": blog_plan}.
```

---

## Step 17 — `agents/researcher.py`

**Description:** LangGraph Researcher worker node for parallel execution. Signature: `researcher_node(state: GraphState) -> dict`. Designed to be dispatched by `Send()` per section. Receives a single section's data from state (section_id, search_query, section_description). Checks `cache.get(search_query)` first; on hit, skips search. On miss: calls `search(query, num_results=5)` from `tools/search.py`, then calls `fetch_page_content(url)` for top 3 results. Assembles raw content snippets. Reads `prompts/researcher_prompt.txt`. Invokes the LLM to produce a research summary. Parses JSON response into `ResearchResult`. Stores result in `cache.set()` and `chroma_store.add_research()`. Returns `{"research_results": [research_result]}`.

**Code-Generation Prompt:**
```
Generate agents/researcher.py for a LangGraph-based Python project called blog-agent.

Requirements:
- Import: json, re, pathlib, structlog.
- Import: ChatOllama from langchain_ollama.
- Import: HumanMessage, SystemMessage from langchain_core.messages.
- Import: GraphState, ResearchResult from graph.state.
- Import: OLLAMA_MODEL, OLLAMA_BASE_URL from app.config.
- Import: search from tools.search.
- Import: fetch_page_content from tools.web_fetcher.
- Import: cache from memory.cache.
- Import: chroma_store from memory.chroma_store.

Define function researcher_node(state: GraphState) -> dict:

This function is called once per section via LangGraph Send(). State will contain the section-specific fields injected by Send().

1. Extract section_id, search_query, section_description from state. If search_query is None or empty, return {"research_results": []}.
2. Check cache: cached = cache.get(search_query). If cached is not None, build ResearchResult from cached dict and return {"research_results": [research_result]}.
3. Call search_results = search(search_query, num_results=5).
4. For the top 3 results (or fewer if less available), call fetch_page_content(result.url).
5. Build a content_snippets list of dicts: [{"title": r.title, "url": r.url, "extracted_text": fetched_text}] for each.
6. Read prompts/researcher_prompt.txt.
7. Build user message: json.dumps({"search_query": search_query, "section_description": section_description, "content_items": content_snippets}).
8. Instantiate ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL).
9. Call llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_message)]).
10. Parse JSON response. Extract summary, source_urls, sufficient.
11. Build ResearchResult(section_id=section_id, query=search_query, summary=parsed["summary"], source_urls=parsed["source_urls"]).
12. Cache result: cache.set(search_query, {"summary": research_result.summary, "source_urls": research_result.source_urls}).
13. Store in chroma: chroma_store.add_research(section_id, search_query, research_result.summary, research_result.source_urls).
14. Log completion with structlog.
15. Return {"research_results": [research_result]}.
```

---

## Step 18 — `agents/writer.py`

**Description:** LangGraph Writer worker node for parallel execution. Signature: `writer_node(state: GraphState) -> dict`. Dispatched by `Send()` per section. Finds the matching `ResearchResult` for the section_id from `state["research_results"]`. Reads `prompts/writer_prompt.txt`. Constructs the user message with section_title, section_description, target_word_count, research_summary, source_urls, section_id. Invokes `ChatOllama` and gets raw Markdown. Extracts all `[SOURCE_N]` citation keys from the content using regex. Builds `SectionDraft(section_id, title, content, citation_keys)`. Returns `{"section_drafts": [draft]}`.

**Code-Generation Prompt:**
```
Generate agents/writer.py for a LangGraph-based Python project called blog-agent.

Requirements:
- Import: json, re, pathlib, structlog.
- Import: ChatOllama from langchain_ollama.
- Import: HumanMessage, SystemMessage from langchain_core.messages.
- Import: GraphState, SectionDraft, ResearchResult from graph.state.
- Import: OLLAMA_MODEL, OLLAMA_BASE_URL from app.config.

Define function writer_node(state: GraphState) -> dict:

This function is called once per section via LangGraph Send(). State will contain the section-specific fields injected by Send() plus the full research_results list.

1. Extract section_id, section_title, section_description, word_count, image_prompt from state.
2. Find matching research: research = next((r for r in state.get("research_results", []) if r.section_id == section_id), None).
3. Build research context:
   - If research is not None: research_summary = research.summary, source_urls = research.source_urls.
   - Else: research_summary = "No web research available. Use general knowledge.", source_urls = [].
4. Read prompts/writer_prompt.txt.
5. Build user_message as a JSON-serialized dict with keys: section_title, section_description, target_word_count, research_summary, source_urls, section_id.
6. Instantiate ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL).
7. Invoke llm with system and user messages. Get response.content as raw Markdown string.
8. Extract citation keys: re.findall(r'\[SOURCE_\d+\]', content) — deduplicate, preserve order.
9. Build SectionDraft(section_id=section_id, title=section_title, content=content, citation_keys=citation_keys).
10. Log section_id and approximate word count of content.
11. Return {"section_drafts": [draft]}.
```

---

## Step 19 — `agents/image_agent.py`

**Description:** LangGraph Image worker node for parallel execution. Signature: `image_agent_node(state: GraphState) -> dict`. Dispatched by `Send()` per section (and once for the feature image). Extracts `section_id` and `image_prompt` from state. Constructs the output image path as `IMAGES_DIR / f"{run_id}_{section_id}.png"`. Calls `generate_image(prompt, output_path)` from `tools/image_gen.py`. Builds `GeneratedImage(section_id, image_path, prompt)`. Returns `{"generated_images": [generated_image]}`.

**Code-Generation Prompt:**
```
Generate agents/image_agent.py for a LangGraph-based Python project called blog-agent.

Requirements:
- Import: structlog, pathlib.
- Import: GraphState, GeneratedImage from graph.state.
- Import: generate_image from tools.image_gen.
- Import: IMAGES_DIR from app.config.

Define function image_agent_node(state: GraphState) -> dict:

This function is dispatched via LangGraph Send() once per section plus once for the feature image.

1. Extract section_id, image_prompt, run_id from state. Use state.get("run_id", "default") for run_id.
2. Construct output_path = str(IMAGES_DIR / f"{run_id}_{section_id}.png").
3. Log the start of image generation with section_id and a truncated prompt (first 60 chars).
4. Call result_path = generate_image(prompt=image_prompt, output_path=output_path).
5. Build generated_image = GeneratedImage(section_id=section_id, image_path=result_path, prompt=image_prompt).
6. Log completion with result_path.
7. Return {"generated_images": [generated_image]}.
```

---

## Step 20 — `agents/citation_manager.py`

**Description:** LangGraph citation registry builder node. Signature: `citation_manager_node(state: GraphState) -> dict`. Iterates all `SectionDraft` objects in `state["section_drafts"]`. For each draft, maps each `[SOURCE_N]` citation key to the corresponding URL from the draft's matching `ResearchResult.source_urls` (by index: SOURCE_1 → index 0, SOURCE_2 → index 1). Builds a flat `citation_registry: Dict[str, str]` where key is the citation marker string and value is the URL. Deduplicates URLs. Returns `{"citation_registry": citation_registry}`.

**Code-Generation Prompt:**
```
Generate agents/citation_manager.py for a LangGraph-based Python project called blog-agent.

Requirements:
- Import: re, structlog, Dict from typing.
- Import: GraphState from graph.state.

Define function citation_manager_node(state: GraphState) -> dict:

1. Initialize citation_registry: Dict[str, str] = {}.
2. Build a lookup: research_by_section = {r.section_id: r for r in state.get("research_results", [])}.
3. Iterate each draft in state.get("section_drafts", []):
   a. Get matching research = research_by_section.get(draft.section_id).
   b. If research is None or research.source_urls is empty, skip.
   c. For each citation_key in draft.citation_keys (e.g., "[SOURCE_1]", "[SOURCE_2]"):
      - Extract the integer N from the key using re.search(r'\d+', citation_key).
      - url_index = N - 1.
      - If url_index < len(research.source_urls), map citation_registry[citation_key] = research.source_urls[url_index].
4. Log total citations registered and total unique URLs.
5. Return {"citation_registry": citation_registry}.
```

---

## Step 21 — `agents/reducer.py`

**Description:** LangGraph final assembler node. Signature: `reducer_node(state: GraphState) -> dict`. Assembles the final blog post from all component outputs. Steps: (1) Sort `section_drafts` by section order from `blog_plan.sections`. (2) For each draft, replace `[IMAGE_PLACEHOLDER_{section_id}]` with a Markdown image tag using the matching `GeneratedImage.image_path`. (3) Replace all `[SOURCE_N]` markers with Markdown inline links using `citation_registry`. (4) Append a `## References` section at the end with all unique URLs as numbered list. (5) Add a feature image at the top. (6) Assemble full Markdown: blog_title as H1 + feature image + section H2 headings + section bodies. (7) Convert to HTML using the `markdown` library. (8) Save both `.md` and `.html` to `BLOGS_DIR`. (9) Return `{"final_blog_md": md_content, "final_blog_html": html_content}`.

**Code-Generation Prompt:**
```
Generate agents/reducer.py for a LangGraph-based Python project called blog-agent.

Requirements:
- Import: re, pathlib, structlog, markdown (the markdown library).
- Import: GraphState from graph.state.
- Import: BLOGS_DIR from app.config.

Define function reducer_node(state: GraphState) -> dict:

1. blog_plan = state["blog_plan"]. section_order = {s.id: i for i, s in enumerate(blog_plan.sections)}.
2. Sort state["section_drafts"] by section_order.get(draft.section_id, 999).
3. Build image_by_section: {img.section_id: img.image_path for img in state.get("generated_images", [])}.
4. Build citation_registry = state.get("citation_registry", {}).
5. Feature image: find image where section_id == "feature" from generated_images. Build feature_img_md = f"![Feature Image]({path})" if found.

6. For each draft in sorted order:
   a. content = draft.content.
   b. Replace [IMAGE_PLACEHOLDER_{draft.section_id}] with f"![{draft.title}]({image_by_section.get(draft.section_id, '')})" if image exists, else remove the placeholder.
   c. Replace each [SOURCE_N] marker with a Markdown inline link: if citation_registry has the key, replace with [SOURCE_N](url); else remove the marker.
   d. Prepend "## {draft.title}\n\n" to the content.
   e. Append to sections_md list.

7. Build references_md:
   - Collect all unique URLs from citation_registry.values().
   - Format as "## References\n\n" + "\n".join(f"{i+1}. {url}" for i, url in enumerate(unique_urls)).

8. Assemble full_md = f"# {blog_plan.blog_title}\n\n{feature_img_md}\n\n" + "\n\n".join(sections_md) + "\n\n" + references_md.

9. Convert to HTML: html_content = markdown.markdown(full_md, extensions=["tables", "fenced_code"]).

10. run_id = state.get("run_id", "blog").
    - Save full_md to BLOGS_DIR / f"{run_id}.md".
    - Save html_content to BLOGS_DIR / f"{run_id}.html".

11. Log file paths.
12. Return {"final_blog_md": full_md, "final_blog_html": html_content}.
```

---

## Step 22 — `graph/checkpointer.py`

**Description:** Sets up the LangGraph SQLite checkpointer for crash recovery and mid-run state persistence. Imports `SqliteSaver` from `langgraph.checkpoint.sqlite`. Defines `get_checkpointer() -> SqliteSaver` which creates the SQLite database file at `outputs/logs/checkpoints.db`, initializes the saver, and returns it. The checkpointer is used by `graph_builder.py` to compile the graph with persistence enabled. Ensures the parent directory exists before creating the DB file.

**Code-Generation Prompt:**
```
Generate graph/checkpointer.py for a LangGraph-based Python project called blog-agent.

Requirements:
- Import: from langgraph.checkpoint.sqlite import SqliteSaver.
- Import: pathlib, structlog.
- Import: LOGS_DIR from app.config.

Define function get_checkpointer() -> SqliteSaver:
1. db_path = LOGS_DIR / "checkpoints.db".
2. db_path.parent.mkdir(parents=True, exist_ok=True).
3. saver = SqliteSaver.from_conn_string(str(db_path)).
4. Log the db_path using structlog.get_logger(__name__).
5. Return saver.

No classes. Only the get_checkpointer function.
```

---

## Step 23 — `graph/graph_builder.py`

**Description:** Defines and compiles the complete LangGraph StateGraph. Imports all node functions and state. Defines the graph topology: START → router_node → planner_node → conditional dispatch. Defines `dispatch_researchers(state) -> List[Send]`: returns a `Send("researcher_node", {...})` for each section with a non-null `search_query`. Defines `dispatch_writers(state) -> List[Send]`: returns a `Send("writer_node", {...})` for each section in `blog_plan.sections`. Defines `dispatch_image_agents(state) -> List[Send]`: returns `Send("image_agent_node", {...})` for each section plus one for the feature image. Adds all nodes. Adds conditional edges using `add_conditional_edges` with `Send()`. After all workers complete, routes to `citation_manager_node` → `reducer_node` → END. Compiles with `SqliteSaver` checkpointer. Exports `compiled_graph`.

**Code-Generation Prompt:**
```
Generate graph/graph_builder.py for a LangGraph-based Python project called blog-agent.

Requirements:
- Import: StateGraph, END, START, Send from langgraph.graph.
- Import: GraphState from graph.state.
- Import: router_node from agents.router.
- Import: planner_node from agents.planner.
- Import: researcher_node from agents.researcher.
- Import: writer_node from agents.writer.
- Import: image_agent_node from agents.image_agent.
- Import: citation_manager_node from agents.citation_manager.
- Import: reducer_node from agents.reducer.
- Import: get_checkpointer from graph.checkpointer.
- Import: List from typing.

Define these dispatch functions:

def dispatch_researchers(state: GraphState) -> List[Send]:
  Return a list of Send("researcher_node", {
    "topic": state["topic"],
    "run_id": state["run_id"],
    "research_results": [],
    "section_drafts": [],
    "generated_images": [],
    "section_id": section.id,
    "search_query": section.search_query,
    "section_description": section.description,
    "blog_plan": state["blog_plan"],
    "citation_registry": {},
    "research_required": state["research_required"],
    "final_blog_md": "",
    "final_blog_html": "",
    "error": None
  }) for each section in state["blog_plan"].sections where section.search_query is not None.
  If no sections need research, return [Send("writer_dispatch", state)].

def dispatch_writers(state: GraphState) -> List[Send]:
  Return a list of Send("writer_node", {
    **state,
    "section_id": section.id,
    "section_title": section.title,
    "section_description": section.description,
    "word_count": section.word_count,
    "image_prompt": section.image_prompt
  }) for each section in state["blog_plan"].sections.

def dispatch_image_agents(state: GraphState) -> List[Send]:
  sends = []
  For feature image: sends.append(Send("image_agent_node", {**state, "section_id": "feature", "image_prompt": state["blog_plan"].feature_image_prompt}))
  For each section: sends.append(Send("image_agent_node", {**state, "section_id": section.id, "image_prompt": section.image_prompt}))
  Return sends.

Build the graph:
builder = StateGraph(GraphState)
Add nodes: "router_node", "planner_node", "researcher_node", "writer_node", "image_agent_node", "citation_manager_node", "reducer_node"

Add edges:
START → "router_node"
"router_node" → "planner_node"
"planner_node" → dispatch_researchers (conditional, using add_conditional_edges)
"researcher_node" → dispatch_writers (conditional edge from fan-in point — use a barrier/join node)

Note: Use a simple join node pattern:
- Add a no-op node "research_complete" that just returns state unchanged.
- Route all researcher_node outputs → "research_complete".
- "research_complete" → dispatch_writers (conditional).
- Add "writer_complete" no-op node. Route all writer_node outputs → "writer_complete".
- "writer_complete" → dispatch_image_agents (conditional).
- Add "image_complete" no-op node. Route all image_agent_node outputs → "image_complete".
- "image_complete" → "citation_manager_node"
- "citation_manager_node" → "reducer_node"
- "reducer_node" → END

Compile: checkpointer = get_checkpointer()
compiled_graph = builder.compile(checkpointer=checkpointer)

Export compiled_graph at module level.
```

---

## Step 24 — `app/ui.py`

**Description:** Streamlit frontend for the blog generation system. Renders: a sidebar with model configuration display and settings; a main panel with a topic text input and "Generate" button; a real-time progress display using `st.status` and `st.empty` containers that stream graph node outputs via `compiled_graph.stream()`; a section-by-section live preview with expandable sections showing content as it's written; an image gallery displaying generated images; a final download panel with buttons for `.md` and `.html` downloads. Runs the LangGraph pipeline in a background `threading.Thread`, communicating progress to Streamlit via `queue.Queue`. Generates a UUID `run_id` per session. Handles errors by displaying the `state["error"]` field.

**Code-Generation Prompt:**
```
Generate app/ui.py for a Streamlit-based Python project called blog-agent.

Requirements:
- Import: streamlit as st, threading, queue, uuid, time, pathlib, structlog.
- Import: compiled_graph from graph.graph_builder.
- Import: BLOGS_DIR, IMAGES_DIR from app.config.

Page config: st.set_page_config(page_title="Blog Agent", layout="wide", page_icon="✍️").

Sidebar:
- Display title "Blog Agent ⚙️".
- Show current model config (OLLAMA_MODEL) and SearxNG status as static text.
- Show a "Clear Cache" button that calls cache.clear_expired() from memory.cache and st.success(...).

Main panel:
- H1 title and subtitle.
- topic_input = st.text_input("Enter a blog topic") with placeholder.
- generate_btn = st.button("🚀 Generate Blog").

On generate_btn click and topic_input non-empty:
1. Generate run_id = str(uuid.uuid4())[:8].
2. Initialize a queue.Queue() called progress_queue.
3. Build initial_state dict with: topic=topic_input, run_id=run_id, research_required=False, blog_plan=None, research_results=[], section_drafts=[], generated_images=[], citation_registry={}, final_blog_md="", final_blog_html="", error=None.

4. Define run_graph() function that runs in a background thread:
   - Iterates compiled_graph.stream(initial_state, config={"configurable": {"thread_id": run_id}}).
   - For each chunk (dict of node_name → state_update), puts (node_name, state_update) onto progress_queue.
   - When done, puts ("__DONE__", {}) onto the queue.
   - On exception, puts ("__ERROR__", {"error": str(e)}) onto the queue.

5. Start threading.Thread(target=run_graph, daemon=True).start().

6. Create st.status("Generating blog...") context with progress_placeholder inside.
7. Poll progress_queue in a while loop (time.sleep(0.1) between polls):
   - On ("router_node", update): show "🔍 Routing topic..."
   - On ("planner_node", update): show "📋 Planning blog structure..." and display blog_plan.blog_title if present.
   - On ("researcher_node", update): show "🔬 Researching sections..."
   - On ("writer_node", update): show "✍️ Writing section: {section_id}..." Use a dict to accumulate section_drafts and display them in st.expander blocks.
   - On ("image_agent_node", update): show "🎨 Generating images..."
   - On ("reducer_node", update): extract final_blog_md and final_blog_html. Break loop.
   - On ("__DONE__", {}): break loop.
   - On ("__ERROR__", update): st.error(update["error"]). Break loop.

8. After loop: display final blog preview using st.markdown(final_blog_md).
9. Image gallery: st.image() for each generated PNG in IMAGES_DIR matching run_id prefix.
10. Download buttons: st.download_button for .md and .html files read from BLOGS_DIR.
```

---

## Step 25 — `docker/searxng/settings.yml`

**Description:** SearxNG engine configuration file that enables multiple search engines to prevent rate limiting and maximize result coverage. Enables engines: `google`, `duckduckgo`, `bing`, `wikipedia`, `startpage`. Sets safe_search to 0, result format to JSON, request rate limiting parameters, and disables engines that require API keys. Configures the server to bind to `0.0.0.0:8080`.

**Code-Generation Prompt:**
```
Generate docker/searxng/settings.yml — a valid SearxNG configuration file for use in a Docker container.

Requirements:
- Set server.bind_address to "0.0.0.0".
- Set server.port to 8080.
- Set server.secret_key to a placeholder string "changeme_secret_key".
- Set search.safe_search to 0.
- Set search.default_lang to "en".
- Set search.formats to ["json", "html"].
- Enable these engines: duckduckgo, wikipedia, bing. Set each engine's disabled: false.
- Disable these engines that require API keys: google (set disabled: true). This avoids immediate bans.
- Set outgoing.request_timeout to 3.0.
- Set outgoing.max_request_timeout to 10.0.
- Set outgoing.useragent_suffix to "blog-agent-searxng".
- Set ui.default_theme to "simple".
- Output only the valid YAML content. No explanation.
```

---

## Step 26 — `docker/Dockerfile`

**Description:** Docker image definition for the blog-agent application container. Based on `python:3.11-slim`. Sets working directory to `/app`. Copies `requirements.txt` and runs `pip install` with no cache. Copies the entire project directory. Creates output subdirectories. Exposes port 8501. Sets the `CMD` to `streamlit run app/ui.py --server.port=8501 --server.address=0.0.0.0`.

**Code-Generation Prompt:**
```
Generate docker/Dockerfile for a Python Streamlit application called blog-agent.

Requirements:
- FROM python:3.11-slim.
- WORKDIR /app.
- COPY requirements.txt .
- RUN pip install --no-cache-dir -r requirements.txt.
- COPY . .
- RUN mkdir -p outputs/images outputs/blogs outputs/logs memory/cache memory/chroma_data.
- EXPOSE 8501.
- ENV PYTHONUNBUFFERED=1.
- CMD ["streamlit", "run", "app/ui.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"].

Output only the Dockerfile content.
```

---

## Step 27 — `docker/docker-compose.yml`

**Description:** Docker Compose stack definition for the full project. Defines three services: `searxng` (image: `searxng/searxng:latest`, mounts `./searxng/settings.yml`, port 8080:8080, restart always), `ollama` (image: `ollama/ollama:latest`, mounts a named volume for model persistence, port 11434:11434, with optional GPU device reservation for NVIDIA), `app` (builds from `./Dockerfile`, depends on `searxng` and `ollama`, mounts the project directory, sets all `.env` variables via `env_file`, port 8501:8501). Defines a named volume `ollama_models`.

**Code-Generation Prompt:**
```
Generate docker/docker-compose.yml for a multi-service Docker application called blog-agent.

Requirements:
Define these three services:

1. searxng:
   - image: searxng/searxng:latest
   - container_name: blog-agent-searxng
   - volumes: - ./searxng/settings.yml:/etc/searxng/settings.yml:ro
   - ports: - "8080:8080"
   - restart: always

2. ollama:
   - image: ollama/ollama:latest
   - container_name: blog-agent-ollama
   - volumes: - ollama_models:/root/.ollama
   - ports: - "11434:11434"
   - restart: always
   - deploy: resources: reservations: devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]

3. app:
   - build: context: .. (one level up from docker/ directory), dockerfile: docker/Dockerfile
   - container_name: blog-agent-app
   - depends_on: [searxng, ollama]
   - ports: - "8501:8501"
   - env_file: - ../.env
   - volumes: - ../outputs:/app/outputs - ../memory:/app/memory
   - environment: OLLAMA_BASE_URL=http://ollama:11434, SEARXNG_BASE_URL=http://searxng:8080
   - restart: on-failure

Define named volume: ollama_models.

Output only the docker-compose.yml content.
```

---

## Step 28 — `tests/fixtures/` (fixture files)

**Description:** Directory containing pre-saved JSON fixture files for offline unit testing. Contains: `sample_search_results.json` (list of 5 SearchResult-like dicts with title, url, snippet), `sample_research_result.json` (a ResearchResult dict for section_1), `sample_blog_plan.json` (a full BlogPlan dict with 4 sections), `sample_section_draft.json` (a SectionDraft dict with citation keys and image placeholder). These fixtures are loaded by test files to avoid live network and LLM calls during unit tests.

**Code-Generation Prompt:**
```
Generate four JSON fixture files for the blog-agent test suite. Output each as a clearly labeled separate JSON block.

1. tests/fixtures/sample_search_results.json:
A JSON array of 5 objects, each with: title (string), url (a realistic fake HTTPS URL), snippet (2 sentences of realistic placeholder text about AI trends).

2. tests/fixtures/sample_research_result.json:
A JSON object matching ResearchResult schema: section_id="section_1", query="latest AI trends 2025", summary (3 sentences of realistic research summary text about AI), source_urls (list of 3 fake HTTPS URLs).

3. tests/fixtures/sample_blog_plan.json:
A JSON object matching BlogPlan schema: blog_title="The Future of AI in 2025", feature_image_prompt (detailed SD prompt), research_required=true, sections array with 4 Section objects each having: id (section_1 through section_4), title, description, word_count (between 300-500), search_query (non-null for first 3, null for last), image_prompt.

4. tests/fixtures/sample_section_draft.json:
A JSON object matching SectionDraft schema: section_id="section_1", title="Introduction to AI Trends", content (a 200-word Markdown string including [SOURCE_1] and [SOURCE_2] markers and [IMAGE_PLACEHOLDER_section_1] token), citation_keys=["[SOURCE_1]", "[SOURCE_2]"].

Output all four as clearly labeled JSON content.
```

---

## Step 29 — `tests/test_router.py`

**Description:** Unit tests for the `router_node` function. Uses `pytest` and `unittest.mock.patch`. Tests: (1) `test_research_required_true` — patches `ChatOllama.invoke` to return JSON `{"research_required": true, "safe": true}` for a time-sensitive topic, asserts returned dict has `research_required=True`. (2) `test_research_required_false` — patches response for evergreen topic, asserts `research_required=False`. (3) `test_unsafe_topic` — patches response with `safe=false`, asserts returned dict contains `"error"` key. (4) `test_malformed_json_fallback` — patches response to return malformed text with embedded JSON, asserts regex fallback extracts correct values.

**Code-Generation Prompt:**
```
Generate tests/test_router.py for a Python project called blog-agent.

Requirements:
- Import: pytest, unittest.mock.patch, MagicMock.
- Import: router_node from agents.router.

Use @patch("agents.router.ChatOllama") to mock the LLM in all tests.

Test 1 — test_research_required_true:
- Mock llm.invoke to return a MagicMock with .content = '{"research_required": true, "safe": true}'.
- Call router_node({"topic": "latest AI trends 2025", "research_required": False, "blog_plan": None, "research_results": [], "section_drafts": [], "generated_images": [], "citation_registry": {}, "final_blog_md": "", "final_blog_html": "", "run_id": "test", "error": None}).
- Assert result["research_required"] is True.
- Assert "error" not in result or result.get("error") is None.

Test 2 — test_research_required_false:
- Mock .content = '{"research_required": false, "safe": true}'.
- Topic = "History of ancient Rome".
- Assert result["research_required"] is False.

Test 3 — test_unsafe_topic:
- Mock .content = '{"research_required": false, "safe": false}'.
- Topic = "harmful content topic".
- Assert "error" in result and result["error"] is not None.

Test 4 — test_malformed_json_fallback:
- Mock .content = 'Here is the result: {"research_required": true, "safe": true} — done.'.
- Assert result["research_required"] is True.

All tests use a minimal but valid GraphState dict as input.
```

---

## Step 30 — `tests/test_planner.py`

**Description:** Unit tests for the `planner_node` function. Mocks `ChatOllama.with_structured_output` to return a mock that produces a `BlogPlan` object built from `sample_blog_plan.json` fixture. Tests: (1) `test_plan_returns_blog_plan` — asserts returned dict has `blog_plan` key with correct type and section count. (2) `test_plan_section_ids_sequential` — asserts section IDs follow `section_1, section_2, ...` pattern. (3) `test_structured_output_fallback` — forces `with_structured_output` to raise, patches raw `invoke` to return the fixture JSON as a string, asserts planner still returns a valid `BlogPlan`.

**Code-Generation Prompt:**
```
Generate tests/test_planner.py for a Python project called blog-agent.

Requirements:
- Import: pytest, json, pathlib, unittest.mock.patch, MagicMock.
- Import: planner_node from agents.planner.
- Import: BlogPlan, Section from graph.state.

Load fixture: sample_plan = json.loads(Path("tests/fixtures/sample_blog_plan.json").read_text()).

Build a helper function make_blog_plan_from_fixture(data: dict) -> BlogPlan that constructs a BlogPlan from the fixture dict.

Test 1 — test_plan_returns_blog_plan:
- Patch "agents.planner.ChatOllama".
- Mock instance.with_structured_output.return_value.invoke.return_value = make_blog_plan_from_fixture(sample_plan).
- Call planner_node({"topic": "The Future of AI", "research_required": True, "blog_plan": None, "research_results": [], "section_drafts": [], "generated_images": [], "citation_registry": {}, "final_blog_md": "", "final_blog_html": "", "run_id": "test", "error": None}).
- Assert "blog_plan" in result.
- Assert isinstance(result["blog_plan"], BlogPlan).
- Assert len(result["blog_plan"].sections) == 4.

Test 2 — test_plan_section_ids_sequential:
- Same mock setup.
- Assert section IDs are ["section_1", "section_2", "section_3", "section_4"].

Test 3 — test_structured_output_fallback:
- Patch "agents.planner.ChatOllama".
- Mock instance.with_structured_output.side_effect = Exception("structured output failed").
- Mock instance.invoke.return_value.content = json.dumps(sample_plan).
- Call planner_node with same state.
- Assert "blog_plan" in result.
- Assert isinstance(result["blog_plan"], BlogPlan).
```

---

## Step 31 — `tests/test_researcher.py`

**Description:** Unit tests for the `researcher_node` function. Mocks `search`, `fetch_page_content`, `cache.get`, `cache.set`, and `ChatOllama.invoke`. Tests: (1) `test_cache_hit_skips_search` — primes `cache.get` to return fixture data, asserts `search` was never called. (2) `test_successful_research` — mocks full pipeline (search returns 3 results, fetch returns text, LLM returns valid JSON), asserts `research_results` list contains one `ResearchResult` with correct section_id. (3) `test_empty_search_query_returns_empty` — passes `search_query=None`, asserts returns `{"research_results": []}`.

**Code-Generation Prompt:**
```
Generate tests/test_researcher.py for a Python project called blog-agent.

Requirements:
- Import: pytest, json, pathlib, unittest.mock.patch, MagicMock.
- Import: researcher_node from agents.researcher.
- Import: ResearchResult from graph.state.

Load fixtures: 
- sample_research = json.loads(Path("tests/fixtures/sample_research_result.json").read_text())
- sample_search = json.loads(Path("tests/fixtures/sample_search_results.json").read_text())

Build base_state dict: topic="test topic", research_required=True, blog_plan=None, research_results=[], section_drafts=[], generated_images=[], citation_registry={}, final_blog_md="", final_blog_html="", run_id="test", error=None, section_id="section_1", search_query="latest AI trends 2025", section_description="Introduction to AI trends".

Test 1 — test_cache_hit_skips_search:
- Patch "agents.researcher.cache.get" to return {"summary": sample_research["summary"], "source_urls": sample_research["source_urls"]}.
- Patch "agents.researcher.search" as mock_search.
- Call researcher_node(base_state).
- Assert mock_search.call_count == 0.
- Assert len(result["research_results"]) == 1.
- Assert result["research_results"][0].section_id == "section_1".

Test 2 — test_successful_research:
- Patch "agents.researcher.cache.get" to return None.
- Patch "agents.researcher.search" to return a list of 3 MagicMock objects each with .url, .title, .snippet attributes.
- Patch "agents.researcher.fetch_page_content" to return "Sample article text content.".
- Patch "agents.researcher.ChatOllama" — mock invoke returns .content = json.dumps({"summary": "AI is transforming industries.", "source_urls": ["https://example.com"], "sufficient": True}).
- Patch "agents.researcher.cache.set" as no-op.
- Patch "agents.researcher.chroma_store.add_research" as no-op.
- Call researcher_node(base_state).
- Assert isinstance(result["research_results"][0], ResearchResult).
- Assert result["research_results"][0].section_id == "section_1".

Test 3 — test_empty_search_query_returns_empty:
- Modify base_state: search_query = None.
- Call researcher_node(base_state).
- Assert result == {"research_results": []}.
```

---

## Step 32 — `tests/test_writer.py`

**Description:** Unit tests for the `writer_node` function. Mocks `ChatOllama.invoke` to return a Markdown content string containing `[SOURCE_1]`, `[SOURCE_2]`, and `[IMAGE_PLACEHOLDER_section_1]`. Tests: (1) `test_writer_returns_draft` — asserts returned dict has `section_drafts` list with one `SectionDraft` and correct `section_id`. (2) `test_citation_keys_extracted` — asserts `citation_keys` list contains `["[SOURCE_1]", "[SOURCE_2]"]`. (3) `test_image_placeholder_in_content` — asserts content string contains `[IMAGE_PLACEHOLDER_section_1]`. (4) `test_no_research_uses_knowledge_fallback` — passes state with empty `research_results`, asserts writer still produces a draft.

**Code-Generation Prompt:**
```
Generate tests/test_writer.py for a Python project called blog-agent.

Requirements:
- Import: pytest, json, pathlib, unittest.mock.patch, MagicMock.
- Import: writer_node from agents.writer.
- Import: SectionDraft, ResearchResult from graph.state.

Load fixture: sample_draft = json.loads(Path("tests/fixtures/sample_section_draft.json").read_text()).
Load fixture: sample_research = json.loads(Path("tests/fixtures/sample_research_result.json").read_text()).

Build research_result = ResearchResult(**sample_research).

Build base_state with: topic="test", research_required=True, blog_plan=None, research_results=[research_result], section_drafts=[], generated_images=[], citation_registry={}, final_blog_md="", final_blog_html="", run_id="test", error=None, section_id="section_1", section_title="Introduction to AI Trends", section_description="Overview of current AI trends.", word_count=400, image_prompt="Futuristic AI visualization".

MOCK_CONTENT = sample_draft["content"]  # contains [SOURCE_1], [SOURCE_2], [IMAGE_PLACEHOLDER_section_1]

Test 1 — test_writer_returns_draft:
- Patch "agents.writer.ChatOllama" — mock invoke returns .content = MOCK_CONTENT.
- Call writer_node(base_state).
- Assert "section_drafts" in result.
- Assert len(result["section_drafts"]) == 1.
- Assert result["section_drafts"][0].section_id == "section_1".
- Assert isinstance(result["section_drafts"][0], SectionDraft).

Test 2 — test_citation_keys_extracted:
- Same mock. Call writer_node.
- Assert "[SOURCE_1]" in result["section_drafts"][0].citation_keys.
- Assert "[SOURCE_2]" in result["section_drafts"][0].citation_keys.

Test 3 — test_image_placeholder_in_content:
- Same mock. Assert "[IMAGE_PLACEHOLDER_section_1]" in result["section_drafts"][0].content.

Test 4 — test_no_research_uses_knowledge_fallback:
- Patch ChatOllama. Set base_state["research_results"] = [].
- Call writer_node. Assert len(result["section_drafts"]) == 1.
```

---

## Step 33 — `tests/test_graph_integration.py`

**Description:** End-to-end integration test that runs the full compiled graph against 3 standard topics. Uses `pytest` with `pytest-asyncio`. For each test topic (a tech topic, an evergreen topic, a research-heavy topic), invokes `compiled_graph.invoke(initial_state, config={"configurable": {"thread_id": run_id}})`. Asserts: `final_blog_md` is non-empty, `len(state["section_drafts"]) >= 4`, `len(state["generated_images"]) >= 1`, `state["citation_registry"]` is a non-empty dict (for research topics), no `error` in state. Marks the test as `slow` via pytest mark and skips if `OLLAMA_BASE_URL` is not reachable.

**Code-Generation Prompt:**
```
Generate tests/test_graph_integration.py for a Python project called blog-agent.

Requirements:
- Import: pytest, uuid, os, httpx.
- Import: compiled_graph from graph.graph_builder.

Define a pytest fixture check_ollama_available that does httpx.get(OLLAMA_BASE_URL + "/api/tags", timeout=3). If it raises, call pytest.skip("Ollama not available").

Define TEST_TOPICS = [
    "Latest developments in large language models 2025",
    "How photosynthesis works",
    "Quantum computing applications in cryptography"
]

Define helper build_initial_state(topic: str) -> dict that builds a full GraphState-compatible dict with all required keys, run_id = str(uuid.uuid4())[:8], and all list fields as [].

For each topic in TEST_TOPICS, define a separate test function:
- test_integration_tech_topic, test_integration_evergreen_topic, test_integration_research_topic

Each test:
1. Use the check_ollama_available fixture.
2. Build initial_state.
3. Call final_state = compiled_graph.invoke(initial_state, config={"configurable": {"thread_id": initial_state["run_id"]}}).
4. Assert final_state["final_blog_md"] != "".
5. Assert len(final_state["section_drafts"]) >= 4.
6. Assert len(final_state["generated_images"]) >= 1.
7. Assert final_state.get("error") is None.
8. For tech and research topics only: assert len(final_state["citation_registry"]) > 0.

Mark all integration tests with @pytest.mark.slow.
```

---

## Step 34 — `notebooks/graph_explorer.ipynb`

**Description:** Jupyter notebook for interactive debugging and visualization of the LangGraph pipeline. Contains cells that: import and display the graph structure using `compiled_graph.get_graph().draw_mermaid()`; run a single-topic test invocation with a short topic and display the state after each node using `compiled_graph.stream()`; display the `BlogPlan` as a formatted table; show generated image thumbnails inline using `IPython.display.Image`; inspect the `citation_registry` dict; display the final Markdown blog using `IPython.display.Markdown`.

**Code-Generation Prompt:**
```
Generate notebooks/graph_explorer.ipynb as a valid Jupyter notebook JSON structure (nbformat 4).

Include these cells in order:

Cell 1 (markdown): # Blog Agent — Graph Explorer\nInteractive debugging notebook for the blog-agent LangGraph pipeline.

Cell 2 (code): Imports — import uuid, json, from IPython.display import display, Image, Markdown, from graph.graph_builder import compiled_graph, from graph.state import GraphState.

Cell 3 (code): Display graph structure — print(compiled_graph.get_graph().draw_mermaid()). Include a comment: "# Copy the Mermaid output to https://mermaid.live to visualize".

Cell 4 (code): Define TOPIC = "The future of renewable energy 2025" and build initial_state dict with all required GraphState fields and run_id = str(uuid.uuid4())[:8].

Cell 5 (code): Stream the graph and print each node update:
for node_name, update in compiled_graph.stream(initial_state, config={"configurable": {"thread_id": initial_state["run_id"]}}):
    print(f"\n=== Node: {node_name} ===")
    for key, value in update.items():
        print(f"  {key}: {str(value)[:200]}")

Cell 6 (code): Get final state by running invoke (separate run_id) and display BlogPlan as formatted JSON:
final_state = compiled_graph.invoke(initial_state_2, ...)
display(Markdown("## BlogPlan\n```json\n" + json.dumps(final_state["blog_plan"].model_dump(), indent=2) + "\n```"))

Cell 7 (code): Display generated images inline:
for img in final_state.get("generated_images", []):
    display(Image(filename=img.image_path, width=300))

Cell 8 (code): Display citation registry as formatted dict.

Cell 9 (code): Display final blog:
display(Markdown(final_state["final_blog_md"]))

Output the complete valid .ipynb JSON. Use nbformat=4, nbformat_minor=5.
```

---

## Step 35 — `README.md`

**Description:** Complete project documentation file. Contains: project title and one-line description, tech stack badges, architecture overview diagram (Mermaid), prerequisite requirements (Docker, Ollama, Python 3.11+, 8GB VRAM), step-by-step quick-start instructions (clone, install, pull Mistral, start SearxNG, first-run SD download, launch app), environment variable reference table, agent description table (7 agents with role and responsibility), LangGraph node flow diagram, known limitations and hardware requirements, how to run tests, how to contribute, and MIT license notice.

**Code-Generation Prompt:**
```
Generate README.md for a Python project called blog-agent.

Requirements:
Include the following sections in order:

1. # Blog Agent — Autonomous AI Blog Generation System
   One-line description. Stack badges for: LangGraph, Ollama, Streamlit, ChromaDB, Stable Diffusion, Python 3.11.

2. ## Architecture Overview
   A Mermaid flowchart showing: User Input → Router → Planner → [Researcher x N (parallel)] → [Writer x N (parallel)] + [Image Agent x N (parallel)] → Citation Manager → Reducer → Output (MD + HTML).

3. ## Prerequisites
   Bullet list: Docker, Ollama, Python 3.11+, 8GB VRAM (RTX 3060+ recommended) or CPU fallback, 10GB disk for models.

4. ## Quick Start
   Numbered steps with exact bash commands:
   1. git clone + cd
   2. cp .env.example .env
   3. pip install -r requirements.txt
   4. ollama pull mistral
   5. docker compose -f docker/docker-compose.yml up -d searxng
   6. python -c "from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4')"
   7. streamlit run app/ui.py

5. ## Environment Variables
   Markdown table with columns: Variable | Default | Description. Include all variables from .env.example.

6. ## Agent Roster
   Markdown table with columns: Agent | Role | Tools. One row per agent (Router, Planner, Researcher, Writer, Image Agent, Citation Manager, Reducer).

7. ## Running Tests
   pytest tests/ -m "not slow" for unit tests.
   pytest tests/ -m slow for integration tests (requires Ollama running).

8. ## Limitations
   Bullet: CPU-only generation is slow (20+ min). Mistral structured output may rarely fail (auto-retry handles it). SearxNG may get rate-limited (fallback to DuckDuckGo).

9. ## License
   MIT License.

Output only the README.md content.
```
