#!/usr/bin/env python3
"""
Streamlit Frontend for BWA Blog Writing Agent
Reflects the Visual Representation specification (diagrams 2.1 – 2.4).

Workflow tabs mirror the §2.4 State Diagram stages:
  Router | Research | Plan | Workers | Images | Blog Preview | Logs
"""

# nest_asyncio MUST be applied first so asyncio.run() works inside
# Streamlit's already-running event loop.
import nest_asyncio
nest_asyncio.apply()

import asyncio
import logging
import warnings
from datetime import datetime
from typing import Any, Dict, Optional

import ollama
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

import update_bwa_backend_1 as backend

# Suppress noisy Streamlit thread-context warnings from background threads
warnings.filterwarnings("ignore", message="Thread '.*': missing ScriptRunContext!")

load_dotenv()

# ======================================================================
# Custom logging handler – feeds backend log lines into session state
# ======================================================================

class StreamlitLogHandler(logging.Handler):
    """Append formatted log records to st.session_state.logs."""

    def emit(self, record: logging.LogRecord) -> None:
        log_entry = self.format(record)
        if "logs" not in st.session_state:
            st.session_state.logs = []
        st.session_state.logs.append(log_entry)


def setup_logging() -> None:
    """Replace all root-logger handlers with the Streamlit capture handler."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in root.handlers[:]:
        root.removeHandler(h)
    handler = StreamlitLogHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root.addHandler(handler)


# ======================================================================
# Helpers
# ======================================================================

def get_ollama_models() -> list:
    """Return installed Ollama model names, or sensible defaults on error."""
    try:
        result = ollama.list()
        models = result.get("models") or []
        names = [m["name"] for m in models]
        return names if names else ["qwen3:4b"]
    except Exception:
        return ["qwen3:4b", "llama3:8b", "mistral:7b"]


# ======================================================================
# Async blog generation
# ======================================================================

async def run_blog_generation(
    topic: str,
    as_of: str,
    model_name: str,
    temperature: float,
    user_research_mode: Optional[str],
) -> Dict[str, Any]:
    """
    Hot-swap the module-level LLMs so compiled graph nodes pick up the
    user's selected model/temperature, then invoke the LangGraph app.
    """
    backend.llm = ChatOllama(model=model_name, temperature=temperature, timeout=60)
    backend.llm_async = ChatOllama(model=model_name, temperature=temperature, timeout=60)

    initial_state: backend.State = {
        "topic": topic,
        "as_of": as_of,
        "sections": [],
        "user_research_mode": user_research_mode,
    }
    return await backend.app.ainvoke(initial_state)


# ======================================================================
# Display helpers  (one per §2.4 state / §2.1 node)
# ======================================================================

def render_workflow_graph() -> None:
    """
    Render the §2.1 Flowchart as a Mermaid diagram.
    Node labels and subgraph boundaries match the spec exactly.
    """
    st.markdown(
        """
```mermaid
flowchart TD
    Start([Start: User provides topic]) --> Router["Router Node\\n(Ollama/Mistral)"]
    Router --> Decision{Research needed?}
    Decision -- No --> Orchestrator["Orchestrator Node\\n(Ollama/Mistral)"]
    Decision -- Yes --> Research["Research Node\\n(DuckDuckGo + Ollama)"]
    Research --> Orchestrator
    Orchestrator --> Fanout[Fan-out: For each task]
    Fanout --> Worker["Worker Node\\n(Ollama/Mistral)"]

    Worker --> Merge[Merge Content]
    Merge --> DecideImages["Decide Images\\n(Ollama/Mistral)"]
    DecideImages --> GenerateImages["Generate Images\\n(Hugging Face Stable Diffusion)"]
    GenerateImages --> PlaceImages[Place Images in Markdown]
    PlaceImages --> Save[Save final .md file]
    Save --> End([End])

    subgraph ReducerSubgraph [Reducer Subgraph]
        Merge
        DecideImages
        GenerateImages
        PlaceImages
    end

    Research -- No results --> Orchestrator
    GenerateImages -- Failure --> PlaceImages
```
        """
    )


def display_router(router_decision: Optional[Dict]) -> None:
    """Display §2.2 RouterDecision output."""
    if not router_decision:
        st.info("No router decision available.")
        return
    st.subheader("🚦 Router Decision")
    col1, col2 = st.columns(2)
    reason = router_decision.get("reason") or "N/A"
    with col1:
        st.metric("Mode", router_decision.get("mode", "N/A"))
        st.metric("Needs Research", str(router_decision.get("needs_research", False)))
    with col2:
        reason_preview = reason[:60] + ("..." if len(reason) > 60 else "")
        st.metric("Reason (preview)", reason_preview)

    queries = router_decision.get("queries") or []
    if queries:
        st.markdown("**Search Queries (sent to DuckDuckGo):**")
        for q in queries:
            st.markdown(f"- `{q}`")
    with st.expander("🔍 Full RouterDecision (JSON)"):
        st.json(router_decision)


def display_research(evidence: list) -> None:
    """
    Display §2.2 Phase A + Phase B output:
    DuckDuckGo results synthesised by Ollama into EvidenceItems.
    """
    if not evidence:
        st.info(
            "No research evidence collected. "
            "This is expected in closed_book mode or when all searches return no results."
        )
        return
    st.subheader("📚 Research Evidence  (DuckDuckGo + Ollama synthesis)")
    df = pd.DataFrame(evidence)
    if not df.empty:
        cols = [c for c in ["title", "url", "published_at", "source"] if c in df.columns]
        df_display = df[cols].copy()
        if "title" in df_display.columns:
            df_display["title"] = df_display["title"].apply(
                lambda x: (str(x)[:65] + "...") if len(str(x)) > 65 else str(x)
            )
        st.dataframe(df_display, use_container_width=True)
    with st.expander("📄 Raw EvidencePack (JSON)"):
        st.json(evidence)


def display_plan(plan: Optional[backend.Plan]) -> None:
    """Display §2.2 Plan output from orchestrator_node."""
    if not plan:
        st.info("No plan generated.")
        return
    st.subheader("📋 Blog Plan")
    st.markdown(f"**Title:** {plan.blog_title}")
    st.markdown(f"**Audience:** {plan.audience}")
    st.markdown(f"**Tone:** {plan.tone}")
    st.markdown(f"**Kind:** `{plan.blog_kind}`")
    if plan.constraints:
        st.markdown("**Constraints:** " + ", ".join(plan.constraints))
    st.markdown("---")
    st.markdown("### Tasks")
    for i, task in enumerate(plan.tasks, 1):
        with st.expander(f"Task {i}: {task.title}  ({task.target_words} words)"):
            st.markdown(f"**Goal:** {task.goal}")
            st.markdown("**Bullets:**")
            for bullet in task.bullets:
                st.markdown(f"- {bullet}")
            if task.tags:
                st.markdown(f"**Tags:** {', '.join(task.tags)}")
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Research:** {'✅' if task.requires_research else '❌'}")
            c2.markdown(f"**Citations:** {'✅' if task.requires_citations else '❌'}")
            c3.markdown(f"**Code:** {'✅' if task.requires_code else '❌'}")


def display_workers(sections: list, plan: Optional[backend.Plan]) -> None:
    """Display §2.4 Writing state output – one expander per worker section."""
    if not sections:
        st.info("No worker sections generated.")
        return
    st.subheader("👷 Worker Sections  (parallel Ollama calls)")
    sorted_sections = sorted(sections, key=lambda x: x[0])
    task_dict: Dict[int, backend.Task] = (
        {task.id: task for task in plan.tasks} if plan else {}
    )
    for task_id, md in sorted_sections:
        task = task_dict.get(task_id)
        label = task.title if task else f"Task {task_id}"
        with st.expander(f"📝 {label}", expanded=False):
            st.markdown(md)


def display_images(image_specs: list, md_with_placeholders: Optional[str]) -> None:
    """
    Display §2.3 ImageSpec list and §2.1 Reducer Subgraph results.
    Shows whether each image was generated by HF SD or failed.
    """
    if not image_specs:
        st.info("No images were requested for this blog post.")
        return
    st.subheader("🖼️ Image Generation  (Hugging Face Stable Diffusion)")
    for spec in image_specs:
        status = "❌ FAILED" if spec.get("error") else "✅ Generated"
        with st.expander(
            f"{status}  {spec.get('placeholder', '?')} → {spec.get('filename', '')}"
        ):
            if spec.get("error"):
                st.error(f"Error: {spec['error']}")
            st.markdown(f"**Alt:** {spec.get('alt', '')}")
            st.markdown(f"**Caption:** {spec.get('caption', '')}")
            st.markdown(f"**HF SD Prompt:** {spec.get('prompt', '')}")
            st.markdown(
                f"**Size:** `{spec.get('size', '1024x1024')}`  |  "
                f"**Quality:** `{spec.get('quality', 'medium')}`"
            )
    if md_with_placeholders:
        with st.expander("📄 Markdown with Placeholders  (pre-substitution)"):
            st.text(md_with_placeholders)


# ======================================================================
# Page configuration – must come before any other st.* call
# ======================================================================
st.set_page_config(
    page_title="BWA Blog Writing Agent",
    page_icon="✍️",
    layout="wide",
)

# Session-state initialisation
if "logs" not in st.session_state:
    st.session_state.logs = []
if "last_state" not in st.session_state:
    st.session_state.last_state = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

setup_logging()

# ======================================================================
# Sidebar
# ======================================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    available_models = get_ollama_models()
    default_idx = (
        available_models.index("qwen3:4b") if "qwen3:4b" in available_models else 0
    )
    selected_model = st.selectbox("Ollama Model", options=available_models, index=default_idx)

    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)

    # Research mode options mirror §2.4 Routing states
    research_mode_options: Dict[str, Optional[str]] = {
        "Auto (let router decide)": None,
        "Standard (closed_book)": "closed_book",
        "Hybrid": "hybrid",
        "Research (open_book)": "open_book",
    }
    selected_mode_label = st.radio(
        "Research Mode",
        options=list(research_mode_options.keys()),
        index=0,
    )
    user_research_mode = research_mode_options[selected_mode_label]

    as_of_date = st.date_input("As-of Date", value=datetime.now().date())

    st.markdown("---")
    st.header("🎛️ Actions")
    run_button = st.button("🚀 Generate Blog Post", type="primary")

    if st.button("🧹 Clear Results & Logs"):
        st.session_state.logs = []
        st.session_state.last_state = None
        st.rerun()

    st.markdown("---")
    if st.session_state.chat_history:
        st.subheader("📜 History")
        for entry in reversed(st.session_state.chat_history[-10:]):
            st.caption(f"[{entry['timestamp'][:16]}] {entry['topic'][:40]}")

# ======================================================================
# Main area
# ======================================================================
st.title("✍️ BWA Blog Writing Agent")
st.caption(
    "Workflow: Router → (DuckDuckGo + Ollama Research) → Orchestrator → "
    "Workers → Merge → Decide Images → Generate (HF SD) → Place Images"
)

topic = st.text_area(
    "Blog Topic",
    placeholder=(
        "e.g.  LangGraph vs AutoGen  |  "
        "Weekly AI news roundup  |  "
        "How to build RAG pipelines"
    ),
    height=100,
)

# ======================================================================
# Blog generation trigger
# ======================================================================
if run_button and topic.strip():
    with st.spinner("Generating blog post… this may take several minutes"):
        st.session_state.logs = []
        try:
            final_state = asyncio.run(
                run_blog_generation(
                    topic=topic.strip(),
                    as_of=as_of_date.isoformat(),
                    model_name=selected_model,
                    temperature=temperature,
                    user_research_mode=user_research_mode,
                )
            )
            st.session_state.last_state = final_state
            st.session_state.chat_history.append(
                {"topic": topic.strip(), "timestamp": datetime.now().isoformat()}
            )
            st.success("✅ Blog post generated successfully!")
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
            st.session_state.logs.append(f"ERROR: {exc}")

elif run_button and not topic.strip():
    st.warning("Please enter a topic before generating.")

# ======================================================================
# Results tabs – ordered to match §2.4 State Diagram flow
# ======================================================================
if st.session_state.last_state:
    fs = st.session_state.last_state

    # Normalise plan: may be Pydantic object or dict depending on LangGraph internals
    raw_plan = fs.get("plan")
    plan_obj: Optional[backend.Plan] = None
    if raw_plan is not None:
        if isinstance(raw_plan, backend.Plan):
            plan_obj = raw_plan
        elif isinstance(raw_plan, dict):
            try:
                plan_obj = backend.Plan(**raw_plan)
            except Exception:
                plan_obj = None

    router_decision: Optional[dict] = fs.get("router_decision")
    evidence: list = fs.get("evidence") or []
    sections: list = fs.get("sections") or []
    image_specs: list = fs.get("image_specs") or []
    md_with_placeholders: Optional[str] = fs.get("md_with_placeholders")
    final_content: str = fs.get("final") or ""

    # Tab ordering follows §2.4 State Diagram left-to-right execution order
    tabs = st.tabs([
        "📄 Blog Preview",       # Final output
        "📊 Workflow Graph",     # §2.1 Flowchart
        "🚦 Router",             # §2.4 Routing state
        "📚 Research",           # §2.4 Researching state (DuckDuckGo + Ollama)
        "📋 Plan",               # §2.4 Planning state
        "👷 Workers",            # §2.4 Writing composite state
        "🖼️ Images",             # §2.4 GeneratingImages + PlacingImages states
        "📜 Logs",               # Execution trace
    ])

    # ── Tab 0: Blog Preview ──────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Final Blog Post")
        if final_content:
            st.markdown(final_content)
            slug = topic.strip()[:40].replace(" ", "_")
            st.download_button(
                "⬇️ Download Markdown",
                data=final_content,
                file_name=f"{slug}.md",
                mime="text/markdown",
            )
        else:
            st.info("No content generated yet.")

    # ── Tab 1: Workflow Graph ────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Workflow Graph  (§2.1 Flowchart)")
        render_workflow_graph()
        st.caption(
            "Diagram is a faithful reproduction of §2.1 from Visual_Representation.md."
        )

    # ── Tab 2: Router ────────────────────────────────────────────────────────
    with tabs[2]:
        display_router(router_decision)

    # ── Tab 3: Research ──────────────────────────────────────────────────────
    with tabs[3]:
        display_research(evidence)

    # ── Tab 4: Plan ──────────────────────────────────────────────────────────
    with tabs[4]:
        display_plan(plan_obj)

    # ── Tab 5: Workers ───────────────────────────────────────────────────────
    with tabs[5]:
        display_workers(sections, plan_obj)

    # ── Tab 6: Images ────────────────────────────────────────────────────────
    with tabs[6]:
        display_images(image_specs, md_with_placeholders)

    # ── Tab 7: Logs ──────────────────────────────────────────────────────────
    with tabs[7]:
        st.subheader("Execution Logs")
        if st.session_state.logs:
            # Newest first
            for log_entry in reversed(st.session_state.logs[-150:]):
                st.code(log_entry, language="text")
        else:
            st.info("No log entries captured yet.")

else:
    st.info("👆 Enter a topic above and click **Generate Blog Post** to start.")















# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@@@@@@@@@@@@@@
################!@!#!@!$$$$$$$$$#@@@@@@@@@@@@@@@@@@@@@$$$$$$$$$$$$$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%








































# #!/usr/bin/env python3
# """
# BWA Blog Writing Agent – Streamlit Frontend
# Features:
#   • Real-time execution stage indicator (thread-based, non-blocking)
#   • All tabs mounted and accessible DURING execution (not only after)
#   • Progressive data: each tab fills as its stage completes
#   • Qwen3 reasoning trace capture and dedicated display tab
#   • WRE research output files visible in UI
# """

# # ── nest_asyncio first: allows asyncio.run() inside Streamlit's event loop ──
# import nest_asyncio
# nest_asyncio.apply()

# import asyncio
# import logging
# import queue
# import re
# import threading
# import time
# import warnings
# from datetime import datetime
# from pathlib import Path
# from typing import Any, Dict, List, Optional

# import ollama
# import pandas as pd
# import streamlit as st
# from dotenv import load_dotenv
# from langchain_ollama import ChatOllama

# import update_bwa_backend_1 as backend

# warnings.filterwarnings("ignore", message="Thread '.*': missing ScriptRunContext!")
# load_dotenv()

# # ======================================================================
# # Stage definitions – used by both the log parser and the UI renderer
# # ======================================================================
# STAGES: List[Dict] = [
#     {"key": "router",       "icon": "🚦", "label": "Router",            "log_trigger": ">>> router_node"},
#     {"key": "research",     "icon": "📡", "label": "Research (DDG)",     "log_trigger": ">>> research_node"},
#     {"key": "synthesis",    "icon": "🔬", "label": "Synthesising evidence","log_trigger": "Phase B"},
#     {"key": "orchestrator", "icon": "📋", "label": "Orchestrator",       "log_trigger": ">>> orchestrator_node"},
#     {"key": "workers",      "icon": "👷", "label": "Workers (parallel)", "log_trigger": "Fanout:"},
#     {"key": "merge",        "icon": "🔗", "label": "Merging sections",   "log_trigger": ">>> merge_content"},
#     {"key": "decide_img",   "icon": "🎨", "label": "Planning images",    "log_trigger": ">>> decide_images"},
#     {"key": "gen_img",      "icon": "🖼️", "label": "Generating images (HF)","log_trigger": ">>> generate_images"},
#     {"key": "place_img",    "icon": "📝", "label": "Placing images",     "log_trigger": ">>> place_images"},
#     {"key": "complete",     "icon": "✅", "label": "Complete",           "log_trigger": None},
# ]
# STAGE_KEYS = [s["key"] for s in STAGES]


# # ======================================================================
# # Log handler – captures log entries AND parses stage transitions
# # ======================================================================

# class StreamlitLogHandler(logging.Handler):
#     """
#     Thread-safe handler that:
#     1. Appends formatted log lines to st.session_state.logs
#     2. Updates st.session_state.current_stage on stage-trigger matches
#     3. Marks completed stages in st.session_state.completed_stages
#     """

#     def emit(self, record: logging.LogRecord) -> None:
#         msg = self.format(record)
#         # These writes are safe from the background thread; Streamlit
#         # session_state is thread-safe for append operations.
#         if "logs" not in st.session_state:
#             st.session_state.logs = []
#         st.session_state.logs.append(msg)

#         # Parse stage transitions
#         for stage in STAGES:
#             trigger = stage.get("log_trigger")
#             if trigger and trigger in msg:
#                 st.session_state.current_stage = stage["key"]
#                 break

#         # Mark previous stages as completed when we see a <<< exit marker
#         if "<<<" in msg:
#             for stage in STAGES:
#                 trigger = stage.get("log_trigger")
#                 if trigger and trigger.replace(">>>", "<<<").strip() in msg:
#                     if "completed_stages" not in st.session_state:
#                         st.session_state.completed_stages = set()
#                     st.session_state.completed_stages.add(stage["key"])
#                     break


# def setup_logging() -> None:
#     root = logging.getLogger()
#     root.setLevel(logging.INFO)
#     # Remove handlers added by previous script runs
#     for h in root.handlers[:]:
#         root.removeHandler(h)
#     handler = StreamlitLogHandler()
#     handler.setFormatter(
#         logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)s  %(message)s")
#     )
#     root.addHandler(handler)


# # ======================================================================
# # Ollama model enumeration
# # ======================================================================

# def get_ollama_models() -> List[str]:
#     try:
#         result = ollama.list()
#         names = [m["name"] for m in (result.get("models") or [])]
#         return names if names else ["qwen3:4b"]
#     except Exception:
#         return ["llama3.2:3b", "llama3:8b", "mistral:7b"]


# # ======================================================================
# # Background pipeline runner
# # ======================================================================


# async def _run_pipeline_async(
#     topic: str,
#     as_of: str,
#     model_name: str,
#     temperature: float,
#     user_research_mode: Optional[str],
# ) -> None:
#     """Async generator that streams intermediate states."""
#     backend.llm = ChatOllama(model=model_name, temperature=temperature, timeout=60)
#     backend.llm_async = ChatOllama(model=model_name, temperature=temperature, timeout=60)

#     initial: backend.State = {
#         "topic": topic,
#         "as_of": as_of,
#         "sections": [],
#         "thinking_traces": [],
#         "user_research_mode": user_research_mode,
#     }

#     async for event in backend.app.astream(initial):
#         # Each event is a dict like {"node_name": state_after_node}
#         for node_name, state_snapshot in event.items():
#             # Merge the snapshot into session state
#             # (simple dict update – deep merge may be needed but LangGraph returns full state)
#             if isinstance(state_snapshot, dict):
#                 st.session_state.last_state = state_snapshot
#             break  # Only one state per event in LangGraph's default stream mode
#     # After stream ends, the final state is already captured
#     st.session_state.is_running = False
#     st.session_state.chat_history.append(
#         {"topic": topic, "timestamp": datetime.now().isoformat()}
#     )

# def _bg_run_pipeline(
#     topic: str,
#     as_of: str,
#     model_name: str,
#     temperature: float,
#     user_research_mode: Optional[str],
# ) -> None:
#     """Background thread entry point."""
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     try:
#         loop.run_until_complete(
#             _run_pipeline_async(topic, as_of, model_name, temperature, user_research_mode)
#         )
#     except Exception as exc:
#         st.session_state.pipeline_error = str(exc)
#         st.session_state.logs.append(f"FATAL ERROR: {exc}")
#     finally:
#         st.session_state.is_running = False
#         loop.close()


# # def _bg_run_pipeline(
# #     topic: str,
# #     as_of: str,
# #     model_name: str,
# #     temperature: float,
# #     user_research_mode: Optional[str],
# # ) -> None:
# #     """
# #     Run the LangGraph pipeline in a background thread.
# #     Writes results back to st.session_state when complete.
# #     This function owns its own event loop.
# #     """
# #     loop = asyncio.new_event_loop()
# #     asyncio.set_event_loop(loop)
# #     try:
# #         backend.llm = ChatOllama(model=model_name, temperature=temperature, timeout=60)
# #         backend.llm_async = ChatOllama(model=model_name, temperature=temperature, timeout=60)

# #         initial: backend.State = {
# #             "topic": topic,
# #             "as_of": as_of,
# #             "sections": [],
# #             "thinking_traces": [],
# #             "user_research_mode": user_research_mode,
# #         }
# #         result = loop.run_until_complete(backend.app.ainvoke(initial))

# #         st.session_state.last_state = result
# #         st.session_state.current_stage = "complete"
# #         if "completed_stages" not in st.session_state:
# #             st.session_state.completed_stages = set()
# #         st.session_state.completed_stages.update(STAGE_KEYS)
# #         st.session_state.chat_history.append(
# #             {"topic": topic, "timestamp": datetime.now().isoformat()}
# #         )
# #     except Exception as exc:
# #         st.session_state.pipeline_error = str(exc)
# #         st.session_state.logs.append(f"FATAL ERROR: {exc}")
# #         st.session_state.current_stage = "error"
# #     finally:
# #         st.session_state.is_running = False
# #         loop.close()


# # ======================================================================
# # Stage progress renderer
# # ======================================================================

# def render_stage_progress() -> None:
#     """
#     Render an inline progress ribbon showing completed / active / pending stages.
#     Uses only built-in Streamlit primitives (no st_autorefresh dependency).
#     """
#     current = st.session_state.get("current_stage", "")
#     completed = st.session_state.get("completed_stages", set())
#     is_running = st.session_state.get("is_running", False)
#     error = st.session_state.get("pipeline_error", "")

#     if error:
#         st.error(f"Pipeline error: {error}")
#         return

#     if not is_running and not st.session_state.get("last_state"):
#         return

#     cols = st.columns(len(STAGES))
#     for i, (stage, col) in enumerate(zip(STAGES, cols)):
#         key = stage["key"]
#         icon = stage["icon"]
#         label = stage["label"]

#         if key in completed:
#             col.markdown(
#                 f"<div style='text-align:center;color:#22c55e;font-size:11px'>"
#                 f"{icon}<br><b>{label}</b><br>✔</div>",
#                 unsafe_allow_html=True,
#             )
#         elif key == current and is_running:
#             col.markdown(
#                 f"<div style='text-align:center;color:#f59e0b;font-size:11px'>"
#                 f"{icon}<br><b>{label}</b><br>⏳</div>",
#                 unsafe_allow_html=True,
#             )
#         else:
#             col.markdown(
#                 f"<div style='text-align:center;color:#9ca3af;font-size:11px'>"
#                 f"{icon}<br>{label}<br>·</div>",
#                 unsafe_allow_html=True,
#             )


# # ======================================================================
# # Tab content helpers
# # ======================================================================

# def _normalise_plan(raw_plan) -> Optional[backend.Plan]:
#     if raw_plan is None:
#         return None
#     if isinstance(raw_plan, backend.Plan):
#         return raw_plan
#     if isinstance(raw_plan, dict):
#         try:
#             return backend.Plan(**raw_plan)
#         except Exception:
#             return None
#     return None


# def render_workflow_graph() -> None:
#     """§2.1 Flowchart – faithfully reproduced."""
#     st.markdown(
#         """
# ```mermaid
# flowchart TD
#     Start([Start: User provides topic]) --> Router["Router Node\\n(Ollama/Mistral)"]
#     Router --> Decision{Research needed?}
#     Decision -- No --> Orchestrator["Orchestrator Node\\n(Ollama/Mistral)"]
#     Decision -- Yes --> Research["Research Node\\n(DuckDuckGo + Ollama)"]
#     Research --> Orchestrator
#     Orchestrator --> Fanout[Fan-out: For each task]
#     Fanout --> Worker["Worker Node\\n(Ollama/Mistral)"]

#     Worker --> Merge[Merge Content]
#     Merge --> DecideImages["Decide Images\\n(Ollama/Mistral)"]
#     DecideImages --> GenerateImages["Generate Images\\n(Hugging Face Stable Diffusion)"]
#     GenerateImages --> PlaceImages[Place Images in Markdown]
#     PlaceImages --> Save[Save final .md file]
#     Save --> End([End])

#     subgraph ReducerSubgraph [Reducer Subgraph]
#         Merge
#         DecideImages
#         GenerateImages
#         PlaceImages
#     end

#     Research -- No results --> Orchestrator
#     GenerateImages -- Failure --> PlaceImages
# ```
#         """
#     )


# def tab_blog_preview(fs: dict) -> None:
#     st.subheader("Final Blog Post")
#     final_content = fs.get("final") or ""
#     if final_content:
#         st.markdown(final_content)
#         slug = (fs.get("topic") or "blog")[:40].replace(" ", "_")
#         st.download_button(
#             "⬇️ Download Markdown",
#             data=final_content,
#             file_name=f"{slug}.md",
#             mime="text/markdown",
#         )
#     elif st.session_state.get("is_running"):
#         st.info("⏳ Blog post is being generated…")
#     else:
#         st.info("No content generated yet.")


# def tab_workflow_graph() -> None:
#     st.subheader("Workflow Graph  (§2.1 Flowchart)")
#     render_workflow_graph()
#     st.caption("Faithful reproduction of §2.1 from Visual_Representation.md.")


# def tab_router(fs: dict) -> None:
#     router_decision = fs.get("router_decision")
#     if not router_decision:
#         if st.session_state.get("is_running"):
#             st.info("⏳ Router has not run yet…")
#         else:
#             st.info("No router decision available.")
#         return

#     st.subheader("🚦 Router Decision")
#     c1, c2 = st.columns(2)
#     reason = router_decision.get("reason") or "N/A"
#     with c1:
#         st.metric("Mode", router_decision.get("mode", "N/A"))
#         st.metric("Needs Research", str(router_decision.get("needs_research", False)))
#     with c2:
#         st.metric("Reason", (reason[:60] + "…") if len(reason) > 60 else reason)

#     queries = router_decision.get("queries") or []
#     if queries:
#         st.markdown("**Search Queries sent to DuckDuckGo:**")
#         for q in queries:
#             st.markdown(f"- `{q}`")
#     with st.expander("🔍 Full RouterDecision JSON"):
#         st.json(router_decision)

#     # Show WRE output files if they exist
#     output_dir = backend.RESEARCH_OUTPUT_DIR
#     if output_dir.exists():
#         files = sorted(output_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
#         if files:
#             st.markdown("---")
#             st.markdown("**📂 Research Output Files** (`research_output/`)")
#             for f in files[:9]:  # show last 9 (3 sets of JSON/MD/HTML)
#                 size_kb = f.stat().st_size / 1024
#                 st.markdown(f"- `{f.name}` &nbsp; ({size_kb:.1f} KB)")

# if not backend.WRE_AVAILABLE:
#     st.warning(
#         "⚠️ Web Research Engine dependencies are missing. "
#         "Install them with:\n"
#         "`pip install ddgs requests beautifulsoup4`\n"
#         "Without these, research will be skipped and no output files will be generated."
#     )

# def tab_research(fs: dict) -> None:
#     evidence = fs.get("evidence") or []
#     if not evidence:
#         if st.session_state.get("is_running") and st.session_state.get("current_stage") in (
#             "router", "research", "synthesis"
#         ):
#             st.info("⏳ Research is in progress…")
#         else:
#             st.info(
#                 "No research evidence collected. Expected in closed_book mode "
#                 "or when DuckDuckGo returns no results."
#             )
#         return

#     st.subheader("📚 Research Evidence  (DuckDuckGo + Ollama synthesis)")
#     df = pd.DataFrame(evidence)
#     if not df.empty:
#         cols = [c for c in ["title", "url", "published_at", "source"] if c in df.columns]
#         df_display = df[cols].copy()
#         if "title" in df_display.columns:
#             df_display["title"] = df_display["title"].apply(
#                 lambda x: (str(x)[:65] + "…") if len(str(x)) > 65 else str(x)
#             )
#         st.dataframe(df_display, use_container_width=True)
#     with st.expander("📄 Raw EvidencePack JSON"):
#         st.json(evidence)


# # Show research output files
# output_dir = backend.RESEARCH_OUTPUT_DIR
# if output_dir.exists():
#     files = sorted(output_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
#     if files:
#         st.markdown("---")
#         st.markdown("**📂 Research Output Files** (`research_output/`)")
#         for f in files[:12]:  # show recent files
#             size_kb = f.stat().st_size / 1024
#             st.markdown(f"- `{f.name}` &nbsp; ({size_kb:.1f} KB)")
#     else:
#         st.info("No research output files found. The research engine may have produced no results, or dependencies are missing.")
# else:
#     st.info("No research output directory – research may have been skipped.")


# def tab_plan(fs: dict) -> None:
#     plan_obj = _normalise_plan(fs.get("plan"))
#     if not plan_obj:
#         if st.session_state.get("is_running"):
#             st.info("⏳ Plan is being generated by the Orchestrator…")
#         else:
#             st.info("No plan generated.")
#         return

#     st.subheader("📋 Blog Plan")
#     st.markdown(f"**Title:** {plan_obj.blog_title}")
#     c1, c2, c3 = st.columns(3)
#     c1.metric("Audience", plan_obj.audience)
#     c2.metric("Tone", plan_obj.tone)
#     c3.metric("Kind", plan_obj.blog_kind)
#     if plan_obj.constraints:
#         st.markdown("**Constraints:** " + ", ".join(plan_obj.constraints))
#     st.markdown("---")
#     st.markdown(f"### Tasks  ({len(plan_obj.tasks)} sections)")
#     for i, task in enumerate(plan_obj.tasks, 1):
#         with st.expander(f"Task {i}: {task.title}  ·  {task.target_words} words"):
#             st.markdown(f"**Goal:** {task.goal}")
#             st.markdown("**Bullets:**")
#             for b in task.bullets:
#                 st.markdown(f"- {b}")
#             if task.tags:
#                 st.markdown(f"**Tags:** {', '.join(task.tags)}")
#             ca, cb, cc = st.columns(3)
#             ca.markdown(f"Research: {'✅' if task.requires_research else '❌'}")
#             cb.markdown(f"Citations: {'✅' if task.requires_citations else '❌'}")
#             cc.markdown(f"Code: {'✅' if task.requires_code else '❌'}")


# def tab_workers(fs: dict) -> None:
#     sections = fs.get("sections") or []
#     plan_obj = _normalise_plan(fs.get("plan"))

#     if not sections:
#         if st.session_state.get("is_running"):
#             st.info("⏳ Workers are writing sections in parallel…")
#         else:
#             st.info("No worker sections generated.")
#         return

#     st.subheader(f"👷 Worker Sections  ({len(sections)} of "
#                  f"{len(plan_obj.tasks) if plan_obj else '?'} complete)")

#     task_dict = {t.id: t for t in plan_obj.tasks} if plan_obj else {}
#     for task_id, md in sorted(sections, key=lambda x: x[0]):
#         task = task_dict.get(task_id)
#         label = task.title if task else f"Task {task_id}"
#         with st.expander(f"📝 {label}", expanded=False):
#             st.markdown(md)

#     if st.session_state.get("is_running") and plan_obj and len(sections) < len(plan_obj.tasks):
#         remaining = len(plan_obj.tasks) - len(sections)
#         st.info(f"⏳ {remaining} section(s) still being written…")


# def tab_images(fs: dict) -> None:
#     image_specs = fs.get("image_specs") or []
#     md_with_placeholders = fs.get("md_with_placeholders")

#     if not image_specs:
#         if st.session_state.get("is_running"):
#             stage = st.session_state.get("current_stage", "")
#             if stage in ("workers", "merge", "decide_img", "gen_img", "place_img"):
#                 st.info("⏳ Image stage in progress…")
#             else:
#                 st.info("⏳ Waiting for image generation stage…")
#         else:
#             st.info("No images were requested for this blog post.")
#         return

#     st.subheader("🖼️ Image Generation  (Hugging Face Stable Diffusion)")
#     success = sum(1 for s in image_specs if not s.get("error"))
#     failed = len(image_specs) - success
#     c1, c2 = st.columns(2)
#     c1.metric("Generated", success)
#     c2.metric("Failed", failed, delta=None)

#     for spec in image_specs:
#         status_icon = "❌" if spec.get("error") else "✅"
#         with st.expander(
#             f"{status_icon} {spec.get('placeholder', '?')} → {spec.get('filename', '')}"
#         ):
#             if spec.get("error"):
#                 st.error(f"Error: {spec['error']}")
#             st.markdown(f"**Alt:** {spec.get('alt', '')}")
#             st.markdown(f"**Caption:** {spec.get('caption', '')}")
#             st.markdown(f"**HF SD Prompt:** {spec.get('prompt', '')}")
#             st.caption(
#                 f"Size: {spec.get('size', '1024x1024')}  |  "
#                 f"Quality: {spec.get('quality', 'medium')}"
#             )

#     if md_with_placeholders:
#         with st.expander("📄 Markdown with placeholders (pre-substitution)"):
#             st.text(md_with_placeholders)


# def tab_reasoning(fs: dict) -> None:
#     """
#     Display Qwen3 reasoning traces captured from <think>…</think> blocks.
#     Only shown when the model actually produced reasoning output.
#     """
#     traces: List[dict] = fs.get("thinking_traces") or []
#     valid = [t for t in traces if t.get("thinking")]

#     if not valid:
#         if st.session_state.get("is_running"):
#             st.info("⏳ Waiting for worker sections to produce reasoning traces…")
#         else:
#             st.info(
#                 "No reasoning traces captured.  \n"
#                 "Qwen3 reasoning is only emitted when the model uses thinking mode. "
#                 "If you are using a non-thinking model variant this tab will remain empty."
#             )
#         return

#     st.subheader(f"🧠 Qwen3 Reasoning Traces  ({len(valid)} section(s))")
#     st.caption(
#         "These are the internal `<think>…</think>` blocks produced by Qwen3 "
#         "before generating each blog section. They are stripped from the final blog output."
#     )
#     st.divider()
#     for t in valid:
#         with st.expander(f"💭 Task {t['task_id']}: {t['title']}", expanded=False):
#             st.markdown(t["thinking"])


# def tab_logs() -> None:
#     logs: List[str] = st.session_state.get("logs") or []
#     is_running = st.session_state.get("is_running", False)

#     st.subheader(f"📜 Execution Logs  ({len(logs)} entries)")
#     if is_running:
#         st.caption("🔴 Live — auto-refreshing every second")
#     if logs:
#         for entry in reversed(logs[-200:]):
#             st.code(entry, language="text")
#     else:
#         st.info("No log entries yet.")


# # ======================================================================
# # Page setup
# # ======================================================================
# st.set_page_config(
#     page_title="BWA Blog Writing Agent",
#     page_icon="✍️",
#     layout="wide",
# )

# # ── Session-state defaults ────────────────────────────────────────────────────
# _defaults = {
#     "logs": [],
#     "last_state": None,
#     "chat_history": [],
#     "is_running": False,
#     "current_stage": "",
#     "completed_stages": set(),
#     "pipeline_error": "",
# }
# for k, v in _defaults.items():
#     if k not in st.session_state:
#         st.session_state[k] = v

# setup_logging()

# # ======================================================================
# # Sidebar
# # ======================================================================
# with st.sidebar:
#     st.header("⚙️ Configuration")

#     available_models = get_ollama_models()
#     default_idx = (
#         available_models.index("qwen3:4b") if "qwen3:4b" in available_models else 0
#     )
#     selected_model = st.selectbox("Ollama Model", options=available_models, index=default_idx)
#     temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)

#     research_mode_map: Dict[str, Optional[str]] = {
#         "Auto (let router decide)": None,
#         "Standard (closed_book)": "closed_book",
#         "Hybrid": "hybrid",
#         "Research (open_book)": "open_book",
#     }
#     user_research_mode = research_mode_map[
#         st.radio("Research Mode", options=list(research_mode_map.keys()), index=0)
#     ]

#     as_of_date = st.date_input("As-of Date", value=datetime.now().date())

#     st.markdown("---")
#     st.header("🎛️ Actions")

#     run_button = st.button(
#         "🚀 Generate Blog Post",
#         type="primary",
#         disabled=st.session_state.is_running,
#     )

#     if st.button("🧹 Clear Results & Logs", disabled=st.session_state.is_running):
#         for k in ["logs", "last_state", "current_stage", "pipeline_error"]:
#             st.session_state[k] = [] if k == "logs" else (None if k == "last_state" else "")
#         st.session_state.completed_stages = set()
#         st.rerun()

#     st.markdown("---")
#     if st.session_state.chat_history:
#         st.subheader("📜 History")
#         for entry in reversed(st.session_state.chat_history[-8:]):
#             st.caption(f"[{entry['timestamp'][:16]}] {entry['topic'][:38]}")

# # ======================================================================
# # Main area – topic input
# # ======================================================================
# st.title("✍️ BWA Blog Writing Agent")
# st.caption(
#     "Router → DuckDuckGo + Ollama Research → Orchestrator → "
#     "Workers (parallel) → Merge → Decide Images → HF SD → Place Images"
# )

# topic = st.text_area(
#     "Blog Topic",
#     placeholder="e.g.  LangGraph vs AutoGen  |  Weekly AI news  |  How to build RAG pipelines",
#     height=90,
#     disabled=st.session_state.is_running,
# )

# # ── Input validation / launch ─────────────────────────────────────────────────
# if run_button:
#     if not topic.strip():
#         st.warning("Please enter a topic before generating.")
#     else:
#         # Reset all progress state
#         st.session_state.logs = []
#         st.session_state.last_state = None
#         st.session_state.current_stage = "router"
#         st.session_state.completed_stages = set()
#         st.session_state.pipeline_error = ""
#         st.session_state.is_running = True

#         # Launch pipeline in background thread so UI stays responsive
#         t = threading.Thread(
#             target=_bg_run_pipeline,
#             args=(
#                 topic.strip(),
#                 as_of_date.isoformat(),
#                 selected_model,
#                 temperature,
#                 user_research_mode,
#             ),
#             daemon=True,
#         )
#         t.start()
#         st.rerun()   # immediately re-render so tabs appear

# # ======================================================================
# # Real-time status bar  (always rendered while running)
# # ======================================================================
# if st.session_state.is_running or st.session_state.last_state:
#     st.divider()

# if st.session_state.is_running:
#     current_key = st.session_state.get("current_stage", "")
#     current_stage_info = next((s for s in STAGES if s["key"] == current_key), None)
#     label = (
#         f"{current_stage_info['icon']} Currently running: **{current_stage_info['label']}**"
#         if current_stage_info else "⏳ Initialising pipeline…"
#     )
#     st.info(label)
#     render_stage_progress()

# elif st.session_state.last_state:
#     if st.session_state.get("pipeline_error"):
#         st.error(f"Pipeline failed: {st.session_state.pipeline_error}")
#     else:
#         st.success("✅ Blog post generated successfully!")
#     render_stage_progress()

# # ======================================================================
# # Tabs – ALWAYS mounted when running or when results exist
# # ======================================================================
# show_tabs = st.session_state.is_running or bool(st.session_state.last_state)

# if show_tabs:
#     # Use partial_state during execution; fall back to last_state when done
#     fs: dict = st.session_state.last_state or {}

#     tabs = st.tabs([
#         "📄 Blog Preview",
#         "📊 Workflow Graph",
#         "🚦 Router",
#         "📚 Research",
#         "📋 Plan",
#         "👷 Workers",
#         "🖼️ Images",
#         "🧠 Reasoning",
#         "📜 Logs",
#     ])

#     with tabs[0]:
#         tab_blog_preview(fs)

#     with tabs[1]:
#         tab_workflow_graph()

#     with tabs[2]:
#         tab_router(fs)

#     with tabs[3]:
#         tab_research(fs)

#     with tabs[4]:
#         tab_plan(fs)

#     with tabs[5]:
#         tab_workers(fs)

#     with tabs[6]:
#         tab_images(fs)

#     with tabs[7]:
#         tab_reasoning(fs)

#     with tabs[8]:
#         tab_logs()

# else:
#     st.info("👆 Enter a topic above and click **Generate Blog Post** to start.")

# # ======================================================================
# # Auto-refresh while running  (1-second polling loop)
# # Streamlit will re-run this script each second, pulling updated
# # session_state written by the background thread.
# # ======================================================================
# if st.session_state.is_running:
#     time.sleep(1)
#     st.rerun()
