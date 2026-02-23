# ui/seo_dashboard.py
import streamlit as st
from graph.state import Severity

def render_seo_dashboard(state):
    st.header("🔍 SEO Audit")

    seo_score = state.get("seo_score", 0)
    st.metric("SEO Score", f"{seo_score}/100")

    issues = state.get("seo_issues", [])
    if issues:
        # Count by severity
        sev_counts = {
            Severity.CRITICAL: 0,
            Severity.WARNING: 0,
            Severity.INFO: 0
        }
        for issue in issues:
            sev = issue.get("severity")
            if sev in sev_counts:
                sev_counts[sev] += 1
        cols = st.columns(3)
        cols[0].metric("Critical", sev_counts[Severity.CRITICAL])
        cols[1].metric("Warnings", sev_counts[Severity.WARNING])
        cols[2].metric("Info", sev_counts[Severity.INFO])

        with st.expander("View detailed issues"):
            for issue in issues:
                severity = issue["severity"].value.upper()
                st.warning(f"[{severity}] {issue['field']}: {issue['message']}")
                st.caption(f"Fix: {issue['suggestion']}")
    else:
        st.success("No SEO issues reported.")

    suggestions = state.get("seo_suggestions", [])
    if suggestions:
        with st.expander("Revision suggestions"):
            for i, s in enumerate(suggestions, 1):
                st.markdown(f"{i}. {s}")

    iteration = state.get("revision_iteration", 0)
    st.info(f"Revision iteration: {iteration}")