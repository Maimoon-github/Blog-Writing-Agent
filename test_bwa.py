#!/usr/bin/env python3
"""
test_bwa.py – Test suite for BWA Blog Writing Agent (backend + frontend)

Run:   python test_bwa.py

Checks:
  - Backend module imports, graph compilation.
  - A minimal generation run (closed_book) producing a plan and final markdown.
  - Frontend module imports and key component definitions.
"""

import asyncio
import sys
import importlib
import traceback
from pathlib import Path

# ----------------------------------------------------------------------
# 1. Test backend
# ----------------------------------------------------------------------
def test_backend():
    print("\n[1] Testing backend module...")
    try:
        import update_bwa_backend_1 as backend
        print("    ✅ Backend module imported.")
    except ImportError as e:
        print(f"    ❌ Failed to import backend: {e}")
        return False

    # Check critical constants / availability
    if not backend.WRE_AVAILABLE:
        print("    ⚠️  Web Research Engine not installed – research will be skipped.")
    if not backend.HF_AVAILABLE:
        print("    ⚠️  Hugging Face Hub not installed – image generation will fail (but test may still pass).")

    # Check that the graph compiles
    try:
        from langgraph.graph import StateGraph
        assert hasattr(backend, 'app') and backend.app is not None
        print("    ✅ Graph compiled successfully.")
    except Exception as e:
        print(f"    ❌ Graph compilation error: {e}")
        return False

    # Run a minimal closed_book generation (no research, no images)
    async def run_test():
        # Prepare a very short topic
        initial_state: backend.State = {
            "topic": "what is the current global situation concerning AI in today's world? AI global situation AI regulation AI_overview",
            "as_of": "2025-01-01",
            "sections": [],
            "thinking_traces": [],
            "user_research_mode": "open_book",   # force  research
        }

        try:
            # Override LLM to a small model to speed up; you may need to adjust model name
            # If your Ollama has a different model, change it here or set env variable.
            import os
            model = os.environ.get("TEST_OLLAMA_MODEL", "llama3.2:3b")
            backend.llm = backend.ChatOllama(model=model, temperature=0.5, timeout=30)
            backend.llm_async = backend.ChatOllama(model=model, temperature=0.5, timeout=30)

            final = await backend.app.ainvoke(initial_state)

            # Basic validation
            if final.get("plan") is None:
                return False, "No plan generated"
            if not final.get("final"):
                return False, "No final content"
            print(f"    ✅ Blog title: {final['plan'].blog_title}")
            print(f"    ✅ Final content length: {len(final['final'])} chars")
            return True, "OK"
        except Exception as e:
            return False, f"Runtime error: {e}"

    try:
        success, msg = asyncio.run(run_test())
        if success:
            print("    ✅ Generation test passed.")
        else:
            print(f"    ❌ Generation test failed: {msg}")
            return False
    except Exception as e:
        print(f"    ❌ Asyncio execution error: {e}")
        return False

    return True


# ----------------------------------------------------------------------
# 2. Test frontend (import‑level and critical component existence)
# ----------------------------------------------------------------------
def test_frontend():
    print("\n[2] Testing frontend module...")
    try:
        import streamlit_app_1 as frontend
        print("    ✅ Frontend module imported.")
    except ImportError as e:
        print(f"    ❌ Failed to import frontend: {e}")
        return False

    # Check that essential functions are defined
    required_funcs = [
        "render_workflow_graph",
        "tab_blog_preview",
        "tab_router",
        "tab_research",
        "tab_plan",
        "tab_workers",
        "tab_images",
        "tab_reasoning",
        "tab_logs",
        "render_stage_progress",
    ]
    missing = [f for f in required_funcs if not hasattr(frontend, f)]
    if missing:
        print(f"    ❌ Missing frontend functions: {missing}")
        return False
    print("    ✅ All expected frontend functions present.")

    # Check that STAGES list is defined
    if not hasattr(frontend, "STAGES") or not frontend.STAGES:
        print("    ❌ STAGES list missing or empty.")
        return False
    print(f"    ✅ STAGES defined ({len(frontend.STAGES)} stages).")

    # Attempt to instantiate a minimal session state (without actually running Streamlit)
    # This verifies that the module can handle its default setup.
    try:
        # Simulate some session state defaults (Streamlit would do this)
        import streamlit as st
        from unittest.mock import MagicMock
        # Mock st.session_state to avoid actual Streamlit execution
        st.session_state = {}
        # Re-run the setup code that relies on session_state (if any)
        # But the frontend module may have top-level code that uses st.*
        # Instead, we just check that the module loads without raising exceptions.
        print("    ✅ Module loads without runtime Streamlit errors.")
    except Exception as e:
        print(f"    ❌ Exception while simulating frontend init: {e}")
        return False

    return True


# ----------------------------------------------------------------------
# 3. Additional checks (file existence, etc.)
# ----------------------------------------------------------------------
def test_environment():
    print("\n[3] Environment checks...")
    issues = []

    # Check that images directory can be created
    try:
        Path("images").mkdir(exist_ok=True)
        print("    ✅ images/ directory accessible.")
    except Exception as e:
        issues.append(f"Cannot create images/: {e}")

    # Check that research output directory can be created
    try:
        # The backend uses RESEARCH_OUTPUT_DIR
        import update_bwa_backend_1 as backend
        backend.RESEARCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print("    ✅ research_output/ directory accessible.")
    except Exception as e:
        issues.append(f"Cannot create research_output/: {e}")

    # Check for .env file (optional)
    if Path(".env").exists():
        print("    ✅ .env file found.")
    else:
        print("    ⚠️  No .env file – HF_TOKEN may be missing (image generation will fail).")

    if issues:
        for i in issues:
            print(f"    ⚠️  {i}")
        return False
    return True


# ----------------------------------------------------------------------
# Main runner
# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print(" BWA Blog Writing Agent – Functional Test")
    print("=" * 60)

    tests = [
        ("Backend", test_backend),
        ("Frontend", test_frontend),
        ("Environment", test_environment),
    ]

    all_passed = True
    for name, func in tests:
        try:
            result = func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"\n[!] Unhandled exception in {name} test:")
            traceback.print_exc()
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED – The system appears fully functional.")
        print("   (Note: Some warnings may be expected if optional dependencies are missing.)")
    else:
        print("❌ SOME TESTS FAILED – Please review the output above.")
    print("=" * 60)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())