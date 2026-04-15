import queue
import threading
import time
import uuid
from typing import Any, Dict

import streamlit as st
import structlog

from config.settings import BLOGS_DIR, IMAGES_DIR, LOGS_DIR
from graph.workflow import compiled_graph

logger = structlog.get_logger(__name__)


def _new_initial_state(topic: str, run_id: str) -> Dict[str, Any]:
    return {
        "topic": topic,
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


def _serialize_section_drafts(drafts: Any) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not isinstance(drafts, list):
        return result
    for draft in drafts:
        if hasattr(draft, "section_id") and hasattr(draft, "content"):
            result[draft.section_id] = draft.content
        elif isinstance(draft, dict):
            sid = draft.get("section_id")
            if sid:
                result[sid] = draft.get("content", "")
    return result


def _start_graph(topic: str) -> None:
    run_id = str(uuid.uuid4())[:8]
    st.session_state.run_id = run_id
    st.session_state.section_drafts = {}
    st.session_state.generated_images = []
    st.session_state.final_blog_md = ""
    st.session_state.final_blog_html = ""
    st.session_state.status = "Starting blog generation..."
    st.session_state.error_message = ""
    st.session_state.progress_queue = queue.Queue()
    st.session_state.graph_thread = threading.Thread(
        target=_run_graph,
        args=(topic, run_id, st.session_state.progress_queue),
        daemon=True,
    )
    st.session_state.graph_thread.start()
    st.session_state.is_running = True


def _run_graph(topic: str, run_id: str, progress_queue: queue.Queue) -> None:
    initial_state = _new_initial_state(topic, run_id)
    config = {"configurable": {"thread_id": run_id}}
    try:
        for chunk in compiled_graph.stream(initial_state, config=config):
            for node_name, state_update in chunk.items():
                progress_queue.put((node_name, state_update))
        progress_queue.put(("__DONE__", {}))
    except Exception as exc:
        logger.exception("ui.graph_failed", error=str(exc), run_id=run_id)
        progress_queue.put(("__ERROR__", {"error": str(exc)}))


def _render_progress() -> None:
    status_placeholder = st.empty()
    output_placeholder = st.empty()
    image_placeholder = st.empty()

    while True:
        try:
            node_name, update = st.session_state.progress_queue.get(timeout=0.25)
        except queue.Empty:
            if not st.session_state.graph_thread.is_alive():
                break
            time.sleep(0.1)
            continue

        if node_name == "__DONE__":
            st.session_state.status = "Blog generation complete."
            break

        if node_name == "__ERROR__":
            st.session_state.error_message = update.get("error", "Unknown error")
            st.session_state.status = "Blog generation failed."
            break

        if node_name == "router_node":
            st.session_state.status = "Routing topic and checking safety..."
        elif node_name == "planner_node":
            st.session_state.status = "Planning blog structure..."
        elif node_name == "researcher_node":
            st.session_state.status = "Researching sources..."
        elif node_name == "writer_node":
            section_id = update.get("section_id", "")
            st.session_state.status = f"Writing section {section_id or '...'}..."
            st.session_state.section_drafts.update(_serialize_section_drafts(update.get("section_drafts", [])))
        elif node_name == "editor_node":
            section_id = update.get("section_id", "")
            st.session_state.status = f"Editing section {section_id or '...'}..."
            st.session_state.section_drafts.update(_serialize_section_drafts(update.get("section_drafts", [])))
        elif node_name == "image_agent_node":
            image_path = update.get("generated_images", [])
            if isinstance(image_path, list) and image_path:
                st.session_state.generated_images.extend([img.image_path for img in image_path if hasattr(img, "image_path")])
            st.session_state.status = "Generating images..."
        elif node_name == "citation_manager_node":
            st.session_state.status = "Resolving citations..."
        elif node_name == "reducer_node":
            st.session_state.status = "Assembling final blog..."
            st.session_state.final_blog_md = update.get("final_blog_md", st.session_state.final_blog_md)
            st.session_state.final_blog_html = update.get("final_blog_html", st.session_state.final_blog_html)

        status_placeholder.info(st.session_state.status)
        with output_placeholder.container():
            if st.session_state.section_drafts:
                st.markdown("#### Live section drafts")
                for section_id, content in st.session_state.section_drafts.items():
                    with st.expander(f"{section_id}", expanded=False):
                        st.markdown(content or "_Writing..._")
        with image_placeholder.container():
            if st.session_state.generated_images:
                st.markdown("#### Generated images")
                for image_path in st.session_state.generated_images:
                    st.image(image_path, use_column_width=True)

    status_placeholder.success(st.session_state.status)

    if st.session_state.error_message:
        st.error(st.session_state.error_message)


def main() -> None:
    st.set_page_config(page_title="BWAgent", page_icon="📝", layout="wide")
    st.title("BWAgent — Autonomous Blog Generation")

    st.sidebar.header("Settings")
    st.sidebar.write("Configure your local services and generation preferences in `.env`.")
    st.sidebar.text(f"Output: {BLOGS_DIR}")
    st.sidebar.text(f"Images: {IMAGES_DIR}")
    st.sidebar.text(f"Logs: {LOGS_DIR}")

    topic = st.text_area("Blog topic", "The future of local AI content creation", height=120)
    if st.button("Generate Blog"):
        if not topic.strip():
            st.warning("Please enter a blog topic before generating.")
        else:
            _start_graph(topic.strip())

    if st.session_state.get("is_running") and st.session_state.get("progress_queue") is not None:
        _render_progress()

    if st.session_state.get("final_blog_md"):
        st.header("Final Blog Markdown")
        st.download_button(
            label="Download Markdown",
            data=st.session_state.final_blog_md,
            file_name=f"blog_{st.session_state.run_id}.md",
            mime="text/markdown",
        )
        st.markdown(st.session_state.final_blog_md)

    if st.session_state.get("generated_images"):
        st.header("Generated Images")
        for image_path in st.session_state.generated_images:
            st.image(image_path, use_column_width=True)


if __name__ == "__main__":
    main()
