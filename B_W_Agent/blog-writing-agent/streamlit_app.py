"""streamlit_app.py
Final UI layer — STEP 11 of the roadmap.
Clean, production-ready Streamlit interface for the full 8-agent pipeline.
"""

# -----------------------------------------------------------------------
# MUST be set before ANY import of transformers or diffusers.
# Suppresses the hundreds of `Accessing __path__` aliasing warnings that
# fire at module-init time in transformers 4.45+ / diffusers combinations.
# Root-fix: pin transformers==4.44.2 in requirements.txt
# -----------------------------------------------------------------------
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
from pathlib import Path

from graph import run_pipeline_sync
from state import CrewState

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Autonomous Blog Agent",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📰 Autonomous Blog Generation Agent")
st.markdown("**8-agent parallel pipeline** • Mistral 7B + Stable Diffusion v1.4 + SearxNG • 100% local • No API keys")

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Pipeline")
    st.caption("Router → Planner → Parallel Workers → Citation Manager → Reducer")
    st.info("Expected runtime: 3–8 minutes on consumer GPU")

    if st.button("🗑️ Clear session & restart", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = None
if "last_topic" not in st.session_state:
    st.session_state.last_topic = ""

# ----------------------------------------------------------------------
# Main input
# ----------------------------------------------------------------------
topic = st.text_input(
    "What should we write about today?",
    placeholder="e.g. The rise of autonomous AI agents in 2026",
    value=st.session_state.last_topic,
    label_visibility="visible"
)

if st.button("🚀 Generate Full Blog Post", type="primary", use_container_width=True):
    if not topic or not topic.strip():
        st.error("⚠️ Please enter a blog topic.")
        st.stop()

    st.session_state.last_topic = topic.strip()

    status_router   = st.status("🔀 **Router** — classifying intent...", expanded=True)
    status_planner  = st.status("📋 **Planner** — generating BlogPlan...", expanded=True)
    status_parallel = st.status("⚡ **Parallel workers** — researching, writing, editing & generating images...", expanded=True)
    status_citation = st.status("📎 **Citation Manager** — building references...", expanded=True)
    status_reducer  = st.status("🔗 **Reducer** — assembling final blog with images & citations...", expanded=True)

    try:
        with st.spinner("Running full 8-agent pipeline..."):
            final_state: CrewState = run_pipeline_sync(topic.strip())

        status_router.update(label="✅ Router complete", state="complete")
        status_planner.update(label="✅ Planner complete", state="complete")
        status_parallel.update(
            label=f"✅ Parallel workers complete ({len(final_state.get('completed_sections', []))} sections)",
            state="complete"
        )
        status_citation.update(label="✅ Citation Manager complete", state="complete")
        status_reducer.update(label="✅ Final blog assembled", state="complete")

        st.session_state.pipeline_state = final_state
        st.success("🎉 Blog post generated successfully!")

    except Exception as e:
        st.error(f"Pipeline error: {e}")
        status_router.update(label="❌ Failed", state="error")
        status_planner.update(label="❌ Failed", state="error")

# ----------------------------------------------------------------------
# Results display
# ----------------------------------------------------------------------
if st.session_state.pipeline_state:
    state: CrewState = st.session_state.pipeline_state
    plan = state.get("plan") or {}

    st.divider()
    st.subheader(plan.get("blog_title", "Generated Blog Post"))

    if state.get("final_markdown"):
        st.markdown(state["final_markdown"])
    else:
        st.warning("No markdown content generated.")

    st.divider()

    images = state.get("generated_images", [])
    if images:
        st.subheader("🖼️ Generated Images")
        cols = st.columns(min(4, len(images)))
        for i, img in enumerate(images):
            with cols[i % len(cols)]:
                if img.file_path and img.file_path != "PLACEHOLDER" and Path(img.file_path).exists():
                    st.image(img.file_path, caption=img.alt_text, use_column_width=True)
                else:
                    st.caption(f"📍 {img.alt_text} (placeholder)")

    st.subheader("📥 Download Outputs")
    col_md, col_html = st.columns(2)

    with col_md:
        if state.get("final_markdown"):
            st.download_button(
                label="📄 Download .md",
                data=state["final_markdown"],
                file_name=Path(state.get("output_path", "blog_post.md")).name,
                mime="text/markdown"
            )

    with col_html:
        if state.get("final_html"):
            st.download_button(
                label="🌐 Download .html",
                data=state["final_html"],
                file_name=Path(state.get("output_path", "blog_post.html")).name,
                mime="text/html"
            )

    st.caption(f"✅ Saved to: `{state.get('output_path', 'outputs/blogs/')}`")

    if st.button("Generate another blog post"):
        st.session_state.pipeline_state = None
        st.rerun()