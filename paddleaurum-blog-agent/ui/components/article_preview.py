# ui/article_preview.py
import streamlit as st

def render_article_preview(state):
    """Render the article markdown with metadata."""
    final_output = state.get("final_output") or {}
    markdown = final_output.get("markdown") or state.get("draft_article") or ""

    st.header("📄 Article Preview")
    with st.expander("Show/hide full article", expanded=True):
        st.markdown(markdown, unsafe_allow_html=True)

    # Show metadata
    st.subheader("Article Metadata")
    cols = st.columns(4)
    cols[0].metric("Title Tag", state.get("title_tag", "N/A")[:50] + "...")
    cols[1].metric("Word Count", final_output.get("word_count", 0))
    cols[2].metric("SEO Score", f"{state.get('seo_score', 0)}/100")
    cols[3].metric("Images Resolved", sum(1 for img in (state.get("image_manifest") or []) if img.get("url")))