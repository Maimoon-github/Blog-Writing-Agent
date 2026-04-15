"""Image generation node for BWAgent."""

import asyncio
import structlog
from typing import Any, Dict

from config.settings import IMAGES_DIR, SD_IMAGE_HEIGHT, SD_IMAGE_WIDTH, SD_INFERENCE_STEPS
from graph.state import GeneratedImage, GraphState
from tools.image_utils import generate_image_async

logger = structlog.get_logger(__name__)


def _normalize_output_path(run_id: str, section_id: str) -> str:
    output_path = IMAGES_DIR / f"{run_id}_{section_id}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path)


async def image_agent_node(state: GraphState) -> Dict[str, Any]:
    section_id = state.get("section_id")
    image_prompt = state.get("image_prompt")
    run_id = state.get("run_id", "blog")

    if not section_id or not image_prompt:
        logger.error("image_agent_node.missing_fields", section_id=section_id)
        return {"error": "Missing section_id or image_prompt", "generated_images": []}

    output_path = _normalize_output_path(run_id, section_id)
    logger.info("image_agent_node.started", section_id=section_id, run_id=run_id)

    try:
        result_path = await generate_image_async(
            prompt=image_prompt,
            output_path=output_path,
            width=SD_IMAGE_WIDTH,
            height=SD_IMAGE_HEIGHT,
            num_inference_steps=SD_INFERENCE_STEPS,
        )
    except Exception as exc:
        logger.error("image_agent_node.failed", section_id=section_id, error=str(exc))
        return {"error": str(exc), "generated_images": []}

    generated_image = GeneratedImage(section_id=section_id, image_path=result_path, prompt=image_prompt)
    logger.info("image_agent_node.completed", section_id=section_id, image_path=result_path)
    return {"generated_images": [generated_image]}
