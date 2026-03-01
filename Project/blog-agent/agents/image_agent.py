"""LangGraph image generation node for parallel section/feature images.

This module implements a worker node that generates an image for a single blog
section (or the feature image) using a Stable Diffusion pipeline. The node is
designed to be dispatched concurrently via LangGraph's `Send()` for each section
and once for the feature image.

Both synchronous and asynchronous versions are provided. The async version uses
`asyncio.to_thread` to offload the blocking `generate_image` call to a thread
pool, keeping the event loop responsive during parallel execution.
"""

import asyncio
import structlog
from pathlib import Path
from typing import Dict, Any, Optional

from graph.state import GraphState, GeneratedImage
from tools.image_gen import generate_image
from app.config import IMAGES_DIR

logger = structlog.get_logger(__name__)

DEFAULT_RUN_ID = "default"
PROMPT_PREVIEW_LENGTH = 60


# ---------------------------------------------------------------------------
# Synchronous version
# ---------------------------------------------------------------------------
def image_agent_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph worker node (synchronous) that generates an image for a section or feature.

    This function is dispatched via `LangGraph Send()` once per section plus
    once for the feature image. Each call receives its own state slice, so
    every invocation is independent and can run in parallel inside a single
    LangGraph superstep.

    Args:
        state: GraphState slice injected by the dispatcher. Expected keys:
            - section_id (str): unique identifier for the section / "feature".
            - image_prompt (str): textual description for the image.
            - run_id (str, optional): pipeline run identifier; defaults to "default".

    Returns:
        Dictionary with:
            - "generated_images": list containing one GeneratedImage (or empty on failure)
            - optionally "error" if critical failure occurred.

    Notes:
        - The parent graph state must declare `generated_images` with an
          `operator.add` reducer so that results from all parallel Send() workers
          are appended rather than overwritten.
        - This function assumes `generate_image` is thread-safe when called concurrently.
    """
    # 1. Extract required fields with validation
    section_id = state.get("section_id")
    image_prompt = state.get("image_prompt")
    run_id = state.get("run_id", DEFAULT_RUN_ID)

    # Validate required fields
    missing = []
    if not section_id:
        missing.append("section_id")
    if not image_prompt:
        missing.append("image_prompt")

    if missing:
        logger.warning(
            "image_agent_node.missing_fields",
            missing_fields=missing,
            run_id=run_id,
        )
        return {
            "error": f"Missing required fields: {', '.join(missing)}",
            "generated_images": [],
        }

    # 2. Construct deterministic output path
    output_path = IMAGES_DIR / f"{run_id}_{section_id}.png"

    # Ensure parent directory exists (create if needed)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(
            "image_agent_node.directory_creation_failed",
            section_id=section_id,
            run_id=run_id,
            output_dir=str(output_path.parent),
            error=str(e),
        )
        return {"error": f"Could not create output directory: {e}", "generated_images": []}

    # 3. Log start (truncate prompt for readability)
    truncated_prompt = (
        image_prompt[:PROMPT_PREVIEW_LENGTH] + "…"
        if len(image_prompt) > PROMPT_PREVIEW_LENGTH
        else image_prompt
    )
    logger.info(
        "image_agent_node.started",
        section_id=section_id,
        run_id=run_id,
        prompt_preview=truncated_prompt,
        output_path=str(output_path),
    )

    # 4. Generate the image
    try:
        result_path = generate_image(prompt=image_prompt, output_path=str(output_path))
    except Exception as e:
        logger.error(
            "image_agent_node.generation_failed",
            section_id=section_id,
            run_id=run_id,
            error=str(e),
            exc_info=True,
        )
        return {"error": f"Image generation failed: {e}", "generated_images": []}

    # 5. Build the result object
    generated_image = GeneratedImage(
        section_id=section_id,
        image_path=result_path,
        prompt=image_prompt,
    )

    # 6. Log completion
    logger.info(
        "image_agent_node.completed",
        section_id=section_id,
        run_id=run_id,
        result_path=result_path,
    )

    # 7. Return result for reducer accumulation
    return {"generated_images": [generated_image]}


# ---------------------------------------------------------------------------
# Asynchronous version
# ---------------------------------------------------------------------------
async def image_agent_node_async(state: GraphState) -> Dict[str, Any]:
    """
    Asynchronous LangGraph worker node for generating an image.

    Identical to `image_agent_node`, but uses `asyncio.to_thread` to run the
    blocking `generate_image` call in a thread pool, allowing the event loop
    to handle other tasks concurrently.

    Args and return value are the same as `image_agent_node`.

    Notes:
        - This version is suitable for use in an async LangGraph pipeline.
        - It respects the same error handling and logging as the sync version.
    """
    # 1. Extract required fields with validation
    section_id = state.get("section_id")
    image_prompt = state.get("image_prompt")
    run_id = state.get("run_id", DEFAULT_RUN_ID)

    # Validate required fields
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

    # 2. Construct deterministic output path
    output_path = IMAGES_DIR / f"{run_id}_{section_id}.png"

    # Ensure parent directory exists (async version still uses sync mkdir,
    # but it's fast and we run it in the event loop – acceptable for FS ops)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(
            "image_agent_node_async.directory_creation_failed",
            section_id=section_id,
            run_id=run_id,
            output_dir=str(output_path.parent),
            error=str(e),
        )
        return {"error": f"Could not create output directory: {e}", "generated_images": []}

    # 3. Log start
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
        output_path=str(output_path),
    )

    # 4. Generate the image in a thread pool
    try:
        # Run the blocking `generate_image` in a separate thread
        result_path = await asyncio.to_thread(
            generate_image, prompt=image_prompt, output_path=str(output_path)
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

    # 5. Build the result object
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
# Thread‑safety note
# ---------------------------------------------------------------------------
# The underlying `generate_image` function may use a singleton Stable Diffusion
# pipeline. If that pipeline is not thread‑safe, concurrent calls from multiple
# threads (as used by `asyncio.to_thread` or the sync version's own concurrency)
# could lead to crashes or corrupted outputs. It is the responsibility of the
# `generate_image` implementation to handle such concurrency (e.g., by using a
# lock or a thread‑local pipeline). This node does not introduce additional
# synchronization beyond what is already provided by the thread pool.