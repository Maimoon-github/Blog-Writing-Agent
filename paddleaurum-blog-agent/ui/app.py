# ui/app.py
import streamlit as st
import asyncio
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from graph.graph_builder import build_graph
from config.settings import CHECKPOINT_DB_PATH
import pandas as pd

# Page config
st.set_page_config(
    page_title="PaddleAurum Blog Review",
    page_icon="📝",
    layout="wide"
)

# Initialize session state
if "selected_session" not in st.session_state:
    st.session_state.selected_session = None
if "graph_app" not in st.session_state:
    st.session_state.graph_app = None
if "current_state" not in st.session_state:
    st.session_state.current_state = None

# Load graph once
@st.cache_resource
def load_graph():
    app, _ = build_graph(credentials_already_loaded=False)
    return app

def load_interrupted_sessions():
    """Query SQLite checkpointer for sessions with pending interrupts."""
    memory = SqliteSaver.from_conn_string(CHECKPOINT_DB_PATH)
    # Get all configs with checkpoints
    configs = memory.list(None)
    interrupted = []
    for config in configs:
        checkpoint = memory.get_tuple(config)
        if checkpoint and checkpoint.checkpoint.get("pending_sends") is not None:
            # Has interrupt
            thread_id = config["configurable"]["thread_id"]
            # Load state to get title info
            app = st.session_state.graph_app
            try:
                state = asyncio.run(app.aget_state(config))
                if state and state.values.get("final_output"):
                    title = state.values["final_output"].get("title_tag", "Untitled")
                else:
                    title = state.values.get("title_tag", "Untitled")
            except:
                title = "Unknown"
            interrupted.append({
                "thread_id": thread_id,
                "title": title,
                "config": config
            })
    return interrupted

def main():
    st.title("🧠 PaddleAurum Blog Review Dashboard")
    st.markdown("Review articles paused for human approval and resume the pipeline.")

    # Load graph
    if st.session_state.graph_app is None:
        with st.spinner("Loading graph..."):
            st.session_state.graph_app = load_graph()
    app = st.session_state.graph_app

    # Sidebar: list interrupted sessions
    st.sidebar.header("Pending Reviews")
    sessions = load_interrupted_sessions()
    if not sessions:
        st.sidebar.info("No sessions currently paused.")
    else:
        session_options = {f"{s['title']} ({s['thread_id'][:8]})": s for s in sessions}
        selected_label = st.sidebar.selectbox(
            "Select a session",
            options=list(session_options.keys()),
            key="session_selector"
        )
        if selected_label:
            selected = session_options[selected_label]
            st.session_state.selected_session = selected
            # Load state
            config = selected["config"]
            with st.spinner("Loading state..."):
                state = asyncio.run(app.aget_state(config))
                st.session_state.current_state = state.values
                st.session_state.current_config = config

    # Main area: show selected session
    if st.session_state.selected_session and st.session_state.current_state:
        state = st.session_state.current_state
        config = st.session_state.current_config

        col1, col2 = st.columns([2, 1])
        with col1:
            from ui.article_preview import render_article_preview
            render_article_preview(state)
        with col2:
            from ui.seo_dashboard import render_seo_dashboard
            render_seo_dashboard(state)

        from ui.approval_controls import render_approval_controls
        render_approval_controls(app, config, state)

    else:
        st.info("Select a session from the sidebar to begin review.")

if __name__ == "__main__":
    main()