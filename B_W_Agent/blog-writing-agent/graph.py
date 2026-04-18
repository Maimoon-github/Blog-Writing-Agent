"""graph.py
Central orchestration pipeline — 8-agent blog generation system.
Exactly matches roadmap.html STEP 10 + idea.md supervisor → parallel → termination flow.
Uses CrewAI v1.14.2 kickoff_async + asyncio.gather + built-in SQLite checkpointing.
"""

import asyncio
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from state import CrewState                  # ← FIXED: correct source of TypedDict
from state import CrewState as CrewStateTypedDict  # TypedDict reference
from agents.router import router_node
from agents.planner import planner_node, dispatch_crews
from agents.citation_manager import citation_manager_node
from agents.reducer import reducer_node

# ----------------------------------------------------------------------
# Checkpointing (SQLite — v1.14.2 recommended resumability pattern)
# ----------------------------------------------------------------------
CHECKPOINT_DB = Path("checkpoints/pipeline.db")
CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)

def _save_checkpoint(state: Dict[str, Any], stage: str) -> None:
    """Save full CrewState snapshot after each major tier."""
    try:
        conn = sqlite3.connect(str(CHECKPOINT_DB))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stage TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                state_json TEXT NOT NULL
            )
        """)
        conn.execute(
            "INSERT INTO checkpoints (stage, timestamp, state_json) VALUES (?, ?, ?)",
            (
                stage,
                datetime.now().isoformat(),
                json.dumps(state, default=str, ensure_ascii=False)
            )
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[graph] ⚠️ Checkpoint write failed for stage '{stage}': {e}")


# ----------------------------------------------------------------------
# Main async orchestration pipeline
# ----------------------------------------------------------------------
async def run_pipeline(topic: str) -> CrewState:
    """Full end-to-end pipeline: supervisor → parallel workers → termination."""
    # 1. Initialize CrewState (single source of truth)
    state: CrewState = {
        "topic": topic.strip(),
        "research_required": True,   # safe default
        "plan": None,
        "research_results": [],
        "completed_sections": [],
        "generated_images": [],
        "citation_registry": {},
        "final_markdown": "",
        "final_html": "",
        "output_path": ""
    }

    try:
        # === SUPERVISOR LAYER (sequential) ===
        # Router
        router_update = router_node(state)
        state.update(router_update)
        _save_checkpoint(state, "after_router")

        # Planner → produces BlogPlan + list of Crew objects
        planner_update = planner_node(state)
        state.update(planner_update)
        _save_checkpoint(state, "after_planner")

        plan = state.get("plan")
        crews: List = planner_update.get("crews", [])

        if not plan or len(crews) == 0:
            raise ValueError(f"Planner produced no crews (got {len(crews)}). Plan: {plan}")

        # === PARALLEL WORKER LAYER ===
        print(f"[graph] 🚀 Dispatching {len(crews)} parallel worker crews...")
        raw_results = await asyncio.gather(
            *[c.kickoff_async() for c in crews],
            return_exceptions=True
        )

        # Merge partial results from all crews (CrewAI aggregation pattern)
        for result in raw_results:
            if isinstance(result, Exception):
                print(f"[graph] Crew failed (non-fatal): {result}")
                continue
            if isinstance(result, dict):
                for key in ("research_results", "completed_sections", "generated_images"):
                    if key in result and isinstance(result[key], list):
                        state.setdefault(key, []).extend(result[key])

        _save_checkpoint(state, "after_parallel_workers")

        # === TERMINATION LAYER (sequential) ===
        # Citation Manager
        citation_update = citation_manager_node(state)
        if citation_update:
            state.update(citation_update)
        _save_checkpoint(state, "after_citation_manager")

        # Reducer / Assembler (final output)
        reducer_update = reducer_node(state)
        if reducer_update:
            state.update(reducer_update)
        _save_checkpoint(state, "final")

        print(f"[graph] ✅ Pipeline completed → {state.get('output_path', 'NO_OUTPUT')}")
        return state

    except Exception as e:
        print(f"[graph] ❌ Critical failure: {e}")
        _save_checkpoint(state, f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        raise


# ----------------------------------------------------------------------
# Synchronous wrapper for Streamlit (non-async UI)
# ----------------------------------------------------------------------
def run_pipeline_sync(topic: str) -> CrewState:
    """Blocking entry point called by streamlit_app.py"""
    return asyncio.run(run_pipeline(topic))