# """
# app/ui.py
# ---------
# Streamlit frontend for the blog-agent generation system.

# Layout
# ------
# • Sidebar  – model config display + "Clear Cache" button.
# • Main     – topic input → background LangGraph run → live progress →
#              section previews → image gallery → download panel.

# Threading model
# ---------------
# `compiled_graph.stream()` runs in a daemon thread; it communicates
# back to the Streamlit script thread exclusively through a `queue.Queue`
# so we never call any `st.*` command from a non-script thread.
# """

# from __future__ import annotations

# import queue
# import threading
# import time
# import uuid
# from pathlib import Path

# import streamlit as st
# import structlog

# from app.config import BLOGS_DIR, IMAGES_DIR
# from graph.graph_builder import compiled_graph
# from memory import cache  # used by "Clear Cache" button

# # ---------------------------------------------------------------------------
# # Logger
# # ---------------------------------------------------------------------------
# logger = structlog.get_logger(__name__)

# # ---------------------------------------------------------------------------
# # Page config  (must be the very first Streamlit call)
# # ---------------------------------------------------------------------------
# st.set_page_config(
#     page_title="Blog Agent",
#     layout="wide",
#     page_icon="✍️",
# )

# # ---------------------------------------------------------------------------
# # Sidebar
# # ---------------------------------------------------------------------------
# with st.sidebar:
#     st.title("Blog Agent ⚙️")
#     st.divider()

#     # -- Model config --------------------------------------------------------
#     st.subheader("Model Configuration")
#     try:
#         from app.config import OLLAMA_MODEL
#         st.text(f"Model : {OLLAMA_MODEL}")
#     except ImportError:
#         st.text("Model : (not configured)")

#     # -- SearxNG status ------------------------------------------------------
#     st.subheader("SearxNG Status")
#     try:
#         from app.config import SEARXNG_URL
#         st.text(f"Endpoint : {SEARXNG_URL}")
#         st.success("SearxNG configured ✅")
#     except ImportError:
#         st.warning("SearxNG URL not set ⚠️")

#     st.divider()

#     # -- Clear cache ---------------------------------------------------------
#     if st.button("🗑️ Clear Cache"):
#         try:
#             cache.clear_expired()
#             st.success("Expired cache entries cleared.")
#         except Exception as exc:  # noqa: BLE001
#             st.error(f"Cache clear failed: {exc}")

# # ---------------------------------------------------------------------------
# # Main panel header
# # ---------------------------------------------------------------------------
# st.title("✍️ Blog Agent")
# st.markdown(
#     "Generate a fully researched, illustrated blog post on **any topic** "
#     "using a multi-agent LangGraph pipeline."
# )
# st.divider()

# # ---------------------------------------------------------------------------
# # Topic input
# # ---------------------------------------------------------------------------
# topic_input: str = st.text_input(
#     "Enter a blog topic",
#     placeholder="e.g. The future of quantum computing in 2030",
# )

# generate_btn = st.button("🚀 Generate Blog", type="primary", disabled=not topic_input)

# # ---------------------------------------------------------------------------
# # Generation logic
# # ---------------------------------------------------------------------------
# if generate_btn and topic_input:
#     run_id: str = str(uuid.uuid4())[:8]
#     logger.info("blog_generation.started", run_id=run_id, topic=topic_input)

#     progress_queue: queue.Queue = queue.Queue()

#     # -- Initial state passed to the graph -----------------------------------
#     initial_state: dict = {
#         "topic": topic_input,
#         "run_id": run_id,
#         "research_required": False,
#         "blog_plan": None,
#         "research_results": [],
#         "section_drafts": [],
#         "generated_images": [],
#         "citation_registry": {},
#         "final_blog_md": "",
#         "final_blog_html": "",
#         "error": None,
#     }

#     # -- Background thread ---------------------------------------------------
#     def run_graph() -> None:
#         """Stream the LangGraph pipeline; push updates onto progress_queue."""
#         try:
#             config = {"configurable": {"thread_id": run_id}}
#             for chunk in compiled_graph.stream(initial_state, config=config):
#                 # chunk is {node_name: state_update}
#                 for node_name, state_update in chunk.items():
#                     progress_queue.put((node_name, state_update))
#             progress_queue.put(("__DONE__", {}))
#         except Exception as exc:  # noqa: BLE001
#             logger.error("blog_generation.error", run_id=run_id, error=str(exc))
#             progress_queue.put(("__ERROR__", {"error": str(exc)}))

#     thread = threading.Thread(target=run_graph, daemon=True)
#     thread.start()

#     # -- Live progress UI (runs in the script thread) ------------------------
#     final_blog_md: str = ""
#     final_blog_html: str = ""
#     section_map: dict[str, str] = {}   # section_id → draft content

#     with st.status("⏳ Generating blog…", expanded=True) as status_box:
#         progress_placeholder = st.empty()
#         sections_placeholder = st.empty()

#         while True:
#             time.sleep(0.1)

#             # Drain all available items from the queue each tick
#             new_items: list[tuple[str, dict]] = []
#             while True:
#                 try:
#                     new_items.append(progress_queue.get_nowait())
#                 except queue.Empty:
#                     break

#             for node_name, update in new_items:
#                 # ---- Router ------------------------------------------------
#                 if node_name == "router_node":
#                     progress_placeholder.info("🔍 Routing topic…")

#                 # ---- Planner -----------------------------------------------
#                 elif node_name == "planner_node":
#                     blog_plan = update.get("blog_plan")
#                     if blog_plan and hasattr(blog_plan, "blog_title"):
#                         progress_placeholder.info(
#                             f"📋 Planning blog structure…\n\n"
#                             f"**Title:** {blog_plan.blog_title}"
#                         )
#                     else:
#                         progress_placeholder.info("📋 Planning blog structure…")

#                 # ---- Researcher --------------------------------------------
#                 elif node_name == "researcher_node":
#                     progress_placeholder.info("🔬 Researching sections…")

#                 # ---- Writer ------------------------------------------------
#                 elif node_name == "writer_node":
#                     # Accumulate section drafts
#                     drafts = update.get("section_drafts", [])
#                     if isinstance(drafts, list):
#                         for draft in drafts:
#                             if hasattr(draft, "section_id") and hasattr(draft, "content"):
#                                 section_map[draft.section_id] = draft.content
#                             elif isinstance(draft, dict):
#                                 sid = draft.get("section_id", "unknown")
#                                 section_map[sid] = draft.get("content", "")
#                     # Also check single section_id key (some graph schemas)
#                     section_id = update.get("section_id", "")
#                     content = update.get("content", "")
#                     if section_id:
#                         section_map[section_id] = content

#                     if section_id:
#                         progress_placeholder.info(f"✍️ Writing section: **{section_id}**…")
#                     else:
#                         progress_placeholder.info("✍️ Writing sections…")

#                     # Render live section preview
#                     with sections_placeholder.container():
#                         st.markdown("#### 📄 Live Section Preview")
#                         for sid, body in section_map.items():
#                             with st.expander(f"Section: `{sid}`", expanded=False):
#                                 st.markdown(body if body else "_Writing…_")

#                 # ---- Image agent -------------------------------------------
#                 elif node_name == "image_agent_node":
#                     progress_placeholder.info("🎨 Generating images…")

#                 # ---- Reducer (final assembly) -------------------------------
#                 elif node_name == "reducer_node":
#                     final_blog_md = update.get("final_blog_md", "")
#                     final_blog_html = update.get("final_blog_html", "")
#                     progress_placeholder.success("✅ Blog generation complete!")
#                     status_box.update(label="✅ Blog generated!", state="complete")
#                     break  # inner for-loop

#                 # ---- Sentinel: done ----------------------------------------
#                 elif node_name == "__DONE__":
#                     status_box.update(label="✅ Done!", state="complete")
#                     break

#                 # ---- Sentinel: error ---------------------------------------
#                 elif node_name == "__ERROR__":
#                     err_msg = update.get("error", "Unknown error")
#                     st.error(f"❌ Pipeline error: {err_msg}")
#                     logger.error("ui.pipeline_error", error=err_msg)
#                     status_box.update(label="❌ Generation failed", state="error")
#                     break
#             else:
#                 # No sentinel received yet → keep polling
#                 continue
#             # Sentinel received → exit polling loop
#             break

#     # -- Final blog preview --------------------------------------------------
#     if final_blog_md:
#         st.divider()
#         st.subheader("📝 Final Blog Preview")
#         st.markdown(final_blog_md)

#     # -- Image gallery -------------------------------------------------------
#     image_paths: list[Path] = sorted(
#         p for p in Path(IMAGES_DIR).glob(f"{run_id}*.png")
#     )
#     if image_paths:
#         st.divider()
#         st.subheader("🖼️ Generated Images")
#         cols = st.columns(min(len(image_paths), 3))
#         for idx, img_path in enumerate(image_paths):
#             with cols[idx % len(cols)]:
#                 st.image(str(img_path), use_column_width=True, caption=img_path.name)

#     # -- Download panel ------------------------------------------------------
#     st.divider()
#     st.subheader("⬇️ Download")

#     blogs_dir = Path(BLOGS_DIR)
#     md_files = sorted(blogs_dir.glob(f"*{run_id}*.md"))
#     html_files = sorted(blogs_dir.glob(f"*{run_id}*.html"))

#     col_md, col_html = st.columns(2)

#     with col_md:
#         if md_files:
#             md_bytes = md_files[0].read_bytes()
#             st.download_button(
#                 label="📄 Download Markdown (.md)",
#                 data=md_bytes,
#                 file_name=md_files[0].name,
#                 mime="text/markdown",
#             )
#         elif final_blog_md:
#             # Fallback: serve the in-memory content
#             st.download_button(
#                 label="📄 Download Markdown (.md)",
#                 data=final_blog_md.encode("utf-8"),
#                 file_name=f"blog_{run_id}.md",
#                 mime="text/markdown",
#             )
#         else:
#             st.info("Markdown file not available yet.")

#     with col_html:
#         if html_files:
#             html_bytes = html_files[0].read_bytes()
#             st.download_button(
#                 label="🌐 Download HTML (.html)",
#                 data=html_bytes,
#                 file_name=html_files[0].name,
#                 mime="text/html",
#             )
#         elif final_blog_html:
#             st.download_button(
#                 label="🌐 Download HTML (.html)",
#                 data=final_blog_html.encode("utf-8"),
#                 file_name=f"blog_{run_id}.html",
#                 mime="text/html",
#             )
#         else:
#             st.info("HTML file not available yet.")





































"""Streamlit frontend for the blog agent with real-time progress.

This module implements a full-featured user interface that allows users to
input a topic, trigger the LangGraph pipeline, and observe progress live.
It handles concurrent graph execution in a background thread, streams state
updates to the UI, displays partial results, and provides download links
for the final blog.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from pathlib import Path

import streamlit as st
import structlog

# Local imports
from app.config import BLOGS_DIR, IMAGES_DIR, OLLAMA_MODEL, SEARXNG_URL
from graph.graph_builder import compiled_graph
from memory import cache  # used by "Clear Cache" button

# --------------------------------------------------------------------------
# Logger
# --------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# --------------------------------------------------------------------------
# Page config – must be the very first Streamlit call
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Blog Agent",
    layout="wide",
    page_icon="✍️",
)

# --------------------------------------------------------------------------
# Session state initialisation
# --------------------------------------------------------------------------
if "generating" not in st.session_state:
    st.session_state.generating = False
if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "final_md" not in st.session_state:
    st.session_state.final_md = ""
if "final_html" not in st.session_state:
    st.session_state.final_html = ""
if "section_drafts" not in st.session_state:
    st.session_state.section_drafts = {}

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("Blog Agent ⚙️")
    st.divider()

    # -- Model config --------------------------------------------------------
    st.subheader("Model Configuration")
    try:
        st.text(f"Model : {OLLAMA_MODEL}")
    except ImportError:
        st.text("Model : (not configured)")

    # -- SearxNG status ------------------------------------------------------
    st.subheader("SearxNG Status")
    try:
        st.text(f"Endpoint : {SEARXNG_URL}")
        st.success("SearxNG configured ✅")
    except ImportError:
        st.warning("SearxNG URL not set ⚠️")

    st.divider()

    # -- Clear cache ---------------------------------------------------------
    if st.button("🗑️ Clear Cache"):
        try:
            cache.clear_expired()
            st.success("Expired cache entries cleared.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Cache clear failed: {exc}")

# --------------------------------------------------------------------------
# Main panel header
# --------------------------------------------------------------------------
st.title("✍️ Blog Agent")
st.markdown(
    "Generate a fully researched, illustrated blog post on **any topic** "
    "using a multi‑agent LangGraph pipeline."
)
st.divider()

# --------------------------------------------------------------------------
# Topic input
# --------------------------------------------------------------------------
topic_input: str = st.text_input(
    "Enter a blog topic",
    placeholder="e.g. The future of quantum computing in 2030",
)

generate_btn = st.button(
    "🚀 Generate Blog",
    type="primary",
    disabled=st.session_state.generating or not topic_input,
)

# --------------------------------------------------------------------------
# Generation logic
# --------------------------------------------------------------------------
if generate_btn and topic_input:
    # Prepare run
    st.session_state.generating = True
    st.session_state.run_id = str(uuid.uuid4())[:8]
    st.session_state.final_md = ""
    st.session_state.final_html = ""
    st.session_state.section_drafts = {}

    run_id = st.session_state.run_id
    logger.info("blog_generation.started", run_id=run_id, topic=topic_input)

    progress_queue: queue.Queue = queue.Queue()

    # -- Initial state passed to the graph -----------------------------------
    initial_state: dict = {
        "topic": topic_input,
        "run_id": run_id,
        "research_required": False,
        "blog_plan": None,
        "research_results": [],
        "section_drafts": [],
        "generated_images": [],
        "citation_registry": {},
        "final_blog_md": "",
        "final_blog_html": "",
        "error": None,
    }

    # -- Background thread ---------------------------------------------------
    def run_graph() -> None:
        """Stream the LangGraph pipeline; push updates onto progress_queue."""
        try:
            config = {"configurable": {"thread_id": run_id}}
            for chunk in compiled_graph.stream(initial_state, config=config):
                # chunk is {node_name: state_update}
                for node_name, state_update in chunk.items():
                    progress_queue.put((node_name, state_update))
            progress_queue.put(("__DONE__", {}))
        except Exception as exc:  # noqa: BLE001
            logger.exception("blog_generation.error", run_id=run_id, error=str(exc))
            progress_queue.put(("__ERROR__", {"error": str(exc)}))

    thread = threading.Thread(target=run_graph, daemon=True)
    thread.start()

    # -- Live progress UI (runs in the script thread) ------------------------
    with st.status("⏳ Generating blog…", expanded=True) as status_box:
        progress_placeholder = st.empty()
        sections_placeholder = st.empty()

        # Poll queue until completion or error
        while True:
            # Drain all available items each tick
            try:
                node_name, update = progress_queue.get(timeout=0.2)
            except queue.Empty:
                # No new data, continue polling
                continue

            # ---- Router ------------------------------------------------
            if node_name == "router_node":
                progress_placeholder.info("🔍 Routing topic…")

            # ---- Planner -----------------------------------------------
            elif node_name == "planner_node":
                blog_plan = update.get("blog_plan")
                if blog_plan and hasattr(blog_plan, "blog_title"):
                    progress_placeholder.info(
                        f"📋 Planning blog structure…\n\n"
                        f"**Title:** {blog_plan.blog_title}"
                    )
                else:
                    progress_placeholder.info("📋 Planning blog structure…")

            # ---- Researcher --------------------------------------------
            elif node_name == "researcher_node":
                progress_placeholder.info("🔬 Researching sections…")

            # ---- Writer ------------------------------------------------
            elif node_name == "writer_node":
                # Accumulate section drafts
                drafts = update.get("section_drafts", [])
                if isinstance(drafts, list):
                    for draft in drafts:
                        if hasattr(draft, "section_id") and hasattr(draft, "content"):
                            st.session_state.section_drafts[draft.section_id] = draft.content
                        elif isinstance(draft, dict):
                            sid = draft.get("section_id", "unknown")
                            st.session_state.section_drafts[sid] = draft.get("content", "")
                # Also check single section_id key (some graph schemas)
                section_id = update.get("section_id", "")
                content = update.get("content", "")
                if section_id:
                    st.session_state.section_drafts[section_id] = content

                if section_id:
                    progress_placeholder.info(f"✍️ Writing section: **{section_id}**…")
                else:
                    progress_placeholder.info("✍️ Writing sections…")

                # Render live section preview
                with sections_placeholder.container():
                    st.markdown("#### 📄 Live Section Preview")
                    for sid, body in st.session_state.section_drafts.items():
                        with st.expander(f"Section: `{sid}`", expanded=False):
                            st.markdown(body if body else "_Writing…_")

            # ---- Image agent -------------------------------------------
            elif node_name == "image_agent_node":
                progress_placeholder.info("🎨 Generating images…")

            # ---- Reducer (final assembly) -------------------------------
            elif node_name == "reducer_node":
                st.session_state.final_md = update.get("final_blog_md", "")
                st.session_state.final_html = update.get("final_blog_html", "")
                progress_placeholder.success("✅ Blog generation complete!")
                status_box.update(label="✅ Blog generated!", state="complete")
                break  # exit polling loop

            # ---- Sentinel: done ----------------------------------------
            elif node_name == "__DONE__":
                status_box.update(label="✅ Done!", state="complete")
                break

            # ---- Sentinel: error ---------------------------------------
            elif node_name == "__ERROR__":
                err_msg = update.get("error", "Unknown error")
                st.error(f"❌ Pipeline error: {err_msg}")
                logger.error("ui.pipeline_error", error=err_msg)
                status_box.update(label="❌ Generation failed", state="error")
                break

            # Mark task as processed (optional, not strictly needed)
            progress_queue.task_done()

    # -- Final blog preview --------------------------------------------------
    if st.session_state.final_md:
        st.divider()
        st.subheader("📝 Final Blog Preview")
        st.markdown(st.session_state.final_md)

    # -- Image gallery -------------------------------------------------------
    image_paths: list[Path] = sorted(Path(IMAGES_DIR).glob(f"{run_id}*.png"))
    if image_paths:
        st.divider()
        st.subheader("🖼️ Generated Images")
        cols = st.columns(min(len(image_paths), 3))
        for idx, img_path in enumerate(image_paths):
            with cols[idx % len(cols)]:
                st.image(str(img_path), use_column_width=True, caption=img_path.name)

    # -- Download panel ------------------------------------------------------
    st.divider()
    st.subheader("⬇️ Download")

    md_files = sorted(Path(BLOGS_DIR).glob(f"*{run_id}*.md"))
    html_files = sorted(Path(BLOGS_DIR).glob(f"*{run_id}*.html"))

    col_md, col_html = st.columns(2)

    with col_md:
        if md_files:
            md_bytes = md_files[0].read_bytes()
            st.download_button(
                label="📄 Download Markdown (.md)",
                data=md_bytes,
                file_name=md_files[0].name,
                mime="text/markdown",
            )
        elif st.session_state.final_md:
            st.download_button(
                label="📄 Download Markdown (.md)",
                data=st.session_state.final_md.encode("utf-8"),
                file_name=f"blog_{run_id}.md",
                mime="text/markdown",
            )
        else:
            st.info("Markdown file not available yet.")

    with col_html:
        if html_files:
            html_bytes = html_files[0].read_bytes()
            st.download_button(
                label="🌐 Download HTML (.html)",
                data=html_bytes,
                file_name=html_files[0].name,
                mime="text/html",
            )
        elif st.session_state.final_html:
            st.download_button(
                label="🌐 Download HTML (.html)",
                data=st.session_state.final_html.encode("utf-8"),
                file_name=f"blog_{run_id}.html",
                mime="text/html",
            )
        else:
            st.info("HTML file not available yet.")

    # Reset generation flag
    st.session_state.generating = False
    st.rerun()  # Force a rerun to re-enable the generate button and clear temporary UI state

# If generation is not active, show a placeholder (optional)
else:
    if st.session_state.generating:
        # This case should not happen because button is disabled, but for completeness
        st.info("Generation in progress...")

