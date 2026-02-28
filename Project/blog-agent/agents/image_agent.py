"""
agents/image_agent.py
─────────────────────
LangGraph worker node for parallel image generation.

Dispatched by `Send()` once per blog section plus once for the feature image.
Each invocation receives its own isolated state slice (section_id + image_prompt)
injected by the dispatcher, making the node fully stateless and safe to run
concurrently across all sections in a single LangGraph superstep.
"""

from __future__ import annotations

import structlog
from pathlib import Path  # noqa: F401 – kept for type-hint clarity; IMAGES_DIR is a Path

from graph.state import GeneratedImage, GraphState
from tools.image_gen import generate_image
from app.config import IMAGES_DIR

logger = structlog.get_logger(__name__)


def image_agent_node(state: GraphState) -> dict:
    """Worker node: generate one image for a given blog section (or feature image).

    This function is dispatched via ``LangGraph Send()`` once per section plus
    once for the feature image.  Each call receives its own state slice, so
    every invocation is independent and can run in parallel inside a single
    LangGraph superstep.

    Parameters
    ----------
    state:
        A ``GraphState`` slice injected by the dispatcher.  Expected keys:
        - ``section_id``  – unique identifier for the section / "feature".
        - ``image_prompt`` – textual description used to generate the image.
        - ``run_id``       – optional pipeline run identifier (defaults to
                             ``"default"`` when absent).

    Returns
    -------
    dict
        ``{"generated_images": [GeneratedImage]}`` – a single-element list so
        that the parent graph's ``operator.add`` reducer can accumulate results
        from all parallel workers without overwriting each other.
    """
    # ── 1. Extract fields from state ─────────────────────────────────────────
    section_id: str = state["section_id"]
    image_prompt: str = state["image_prompt"]
    run_id: str = state.get("run_id", "default")  # type: ignore[call-overload]

    # ── 2. Build deterministic output path ───────────────────────────────────
    output_path: str = str(IMAGES_DIR / f"{run_id}_{section_id}.png")

    # ── 3. Log start (truncate prompt to 60 chars to keep logs readable) ─────
    truncated_prompt: str = image_prompt[:60] + ("…" if len(image_prompt) > 60 else "")
    logger.info(
        "image_generation.started",
        section_id=section_id,
        run_id=run_id,
        prompt_preview=truncated_prompt,
        output_path=output_path,
    )

    # ── 4. Generate the image ─────────────────────────────────────────────────
    result_path: str = generate_image(prompt=image_prompt, output_path=output_path)

    # ── 5. Build the result object ────────────────────────────────────────────
    generated_image = GeneratedImage(
        section_id=section_id,
        image_path=result_path,
        prompt=image_prompt,
    )

    # ── 6. Log completion ─────────────────────────────────────────────────────
    logger.info(
        "image_generation.completed",
        section_id=section_id,
        run_id=run_id,
        result_path=result_path,
    )

    # ── 7. Return result for reducer accumulation ─────────────────────────────
    # The parent graph state must declare `generated_images` with an
    # `operator.add` reducer so that results from all parallel Send() workers
    # are appended rather than overwritten.
    return {"generated_images": [generated_image]}