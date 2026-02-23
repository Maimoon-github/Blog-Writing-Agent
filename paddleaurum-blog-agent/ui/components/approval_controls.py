# ui/approval_controls.py
import streamlit as st
import asyncio
from langgraph.types import Command

def render_approval_controls(app, config, state):
    st.header("✅ Approval Decision")

    with st.form(key="approval_form"):
        feedback = st.text_area(
            "Feedback (required if not approved)",
            placeholder="Enter revision instructions or leave blank if approved.",
            key="feedback_input"
        )
        col1, col2 = st.columns(2)
        approve = col1.form_submit_button("✅ Approve", type="primary")
        reject = col2.form_submit_button("🔄 Request Revision")

    if approve or reject:
        approved = approve
        if not approved and not feedback.strip():
            st.error("Please provide feedback when requesting a revision.")
            return

        # Prepare resume value
        resume_value = {
            "approved": approved,
            "feedback": feedback.strip()
        }

        # Resume the graph
        try:
            with st.spinner("Resuming pipeline..."):
                # Use asyncio to invoke the graph with Command
                async def resume():
                    # app.invoke expects Command as input
                    # We need to call with Command(resume=resume_value)
                    # Note: app.invoke expects the input to be Command
                    # We'll pass it as Command(resume=resume_value)
                    result = await app.ainvoke(
                        Command(resume=resume_value),
                        config=config
                    )
                    return result

                result = asyncio.run(resume())
                st.success("Pipeline resumed successfully!")
                # Optionally update state
                st.session_state.current_state = result
                st.rerun()
        except Exception as e:
            st.error(f"Error resuming pipeline: {e}")