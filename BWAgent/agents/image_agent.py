"""Image generation node for BWAgent.

This module implements a worker node that generates an image for a single blog
section (or the feature image) using a Stable Diffusion pipeline. The node is
designed to be dispatched concurrently via LangGraph's `Send()` for each section
and once for the feature image.

Both synchronous and asynchronous versions are provided. The async version uses
`asyncio.to_thread` to offload the blocking generation call to a thread pool,
keeping the event loop responsive during parallel execution.
"""

import asyncio
import structlog
from pathlib import Path
from typing import Any, Dict, Optional

# Import from the project's configuration and utilities
from config.settings import IMAGES_DIR, SD_IMAGE_HEIGHT, SD_IMAGE_WIDTH, SD_INFERENCE_STEPS
from graph.state import GeneratedImage, GraphState
from tools.image_utils import generate_image_async

logger = structlog.get_logger(__name__)

DEFAULT_RUN_ID = "blog"
PROMPT_PREVIEW_LENGTH = 60


def _normalize_output_path(run_id: str, section_id: str) -> str:
    """Create a deterministic output path and ensure parent directory exists."""
    output_path = IMAGES_DIR / f"{run_id}_{section_id}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path)


# ---------------------------------------------------------------------------
# Asynchronous version (recommended for LangGraph parallel dispatch)
# ---------------------------------------------------------------------------
async def image_agent_node_async(state: GraphState) -> Dict[str, Any]:
    """
    Asynchronous LangGraph worker node for generating an image.

    This function is dispatched via `LangGraph Send()` once per section plus
    once for the feature image. Each call receives its own state slice, so
    every invocation is independent and can run in parallel inside a single
    LangGraph superstep.

    Args:
        state: GraphState slice injected by the dispatcher. Expected keys:
            - section_id (str): unique identifier for the section / "feature".
            - image_prompt (str): textual description for the image.
            - run_id (str, optional): pipeline run identifier; defaults to "blog".

    Returns:
        Dictionary with:
            - "generated_images": list containing one GeneratedImage (or empty on failure)
            - optionally "error" if critical failure occurred.

    Notes:
        - The parent graph state must declare `generated_images` with an
          `operator.add` reducer so that results from all parallel Send() workers
          are appended rather than overwritten.
        - This function uses `generate_image_async` which internally handles
          thread‑safe execution of the Stable Diffusion pipeline.
    """
    # 1. Extract required fields with validation
    section_id = state.get("section_id")
    image_prompt = state.get("image_prompt")
    run_id = state.get("run_id", DEFAULT_RUN_ID)

    missing = []
    if not section_id:
        missing.append("section_id")
    if not image_prompt:
        missing.append("image_prompt")

    if missing:
        logger.warning(
            "image_agent_node_async.missing_fields",
            missing_fields=missing,
            run_id=run_id,
        )
        return {
            "error": f"Missing required fields: {', '.join(missing)}",
            "generated_images": [],
        }

    # 2. Construct output path
    output_path = _normalize_output_path(run_id, section_id)

    # 3. Log start (truncate prompt for readability)
    truncated_prompt = (
        image_prompt[:PROMPT_PREVIEW_LENGTH] + "…"
        if len(image_prompt) > PROMPT_PREVIEW_LENGTH
        else image_prompt
    )
    logger.info(
        "image_agent_node_async.started",
        section_id=section_id,
        run_id=run_id,
        prompt_preview=truncated_prompt,
        output_path=output_path,
    )

    # 4. Generate the image (async, but may still be blocking; called directly)
    try:
        result_path = await generate_image_async(
            prompt=image_prompt,
            output_path=output_path,
            width=SD_IMAGE_WIDTH,
            height=SD_IMAGE_HEIGHT,
            num_inference_steps=SD_INFERENCE_STEPS,
        )
    except Exception as e:
        logger.error(
            "image_agent_node_async.generation_failed",
            section_id=section_id,
            run_id=run_id,
            error=str(e),
            exc_info=True,
        )
        return {"error": f"Image generation failed: {e}", "generated_images": []}

    # 5. Build result object
    generated_image = GeneratedImage(
        section_id=section_id,
        image_path=result_path,
        prompt=image_prompt,
    )

    # 6. Log completion
    logger.info(
        "image_agent_node_async.completed",
        section_id=section_id,
        run_id=run_id,
        result_path=result_path,
    )

    # 7. Return result for reducer accumulation
    return {"generated_images": [generated_image]}


# ---------------------------------------------------------------------------
# Synchronous version (for non‑async graphs or simple testing)
# ---------------------------------------------------------------------------
def image_agent_node_sync(state: GraphState) -> Dict[str, Any]:
    """
    Synchronous LangGraph worker node for generating an image.

    This version calls the async function via `asyncio.run()` – suitable only
    for environments where an event loop is not already running (e.g., in a
    simple script or when using LangGraph's sync execution). For production
    use with parallel dispatch, prefer the async version.

    Args and return value are the same as `image_agent_node_async`.
    """
    # Since we cannot call async from sync directly without a loop,
    # we create a new event loop. This is safe only if no other loop is running.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop – we can create one
        return asyncio.run(image_agent_node_async(state))
    else:
        # A loop is already running – we cannot call run() again.
        # In this case, the caller should use the async version directly.
        logger.error(
            "image_agent_node_sync.called_with_existing_loop",
            msg="Use image_agent_node_async when an event loop is already running",
        )
        return {"error": "Event loop already running; use async version", "generated_images": []}


# ---------------------------------------------------------------------------
# Compatibility alias (the node expected by the main graph)
# ---------------------------------------------------------------------------
# By default, export the async version as the primary node.
# The main graph can import `image_agent_node` and use it in an async pipeline.
image_agent_node = image_agent_node_async


# ---------------------------------------------------------------------------
# Thread‑safety note
# ---------------------------------------------------------------------------
# The underlying `generate_image_async` is expected to handle concurrency
# (e.g., using a lock or thread‑local pipeline). This node does not add
# additional synchronization beyond what is provided by the async runtime.