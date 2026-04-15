"""Stable Diffusion image utilities for BWAgent."""

import asyncio
import math
import random
import time
from pathlib import Path
from typing import Optional

import structlog
import torch
from diffusers import StableDiffusionPipeline

from config.settings import SD_DEVICE, SD_IMAGE_HEIGHT, SD_IMAGE_WIDTH, SD_INFERENCE_STEPS, SD_MODEL_ID, SD_SAFETY_CHECKER

logger = structlog.get_logger(__name__)
_pipeline: Optional[StableDiffusionPipeline] = None
_load_lock = asyncio.Lock()
MIN_DIMENSION = 64
DIMENSION_MULTIPLE = 8
MAX_OOM_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2.0


def _validate_dimensions(width: int, height: int) -> tuple[int, int]:
    width = max(MIN_DIMENSION, (width // DIMENSION_MULTIPLE) * DIMENSION_MULTIPLE)
    height = max(MIN_DIMENSION, (height // DIMENSION_MULTIPLE) * DIMENSION_MULTIPLE)
    return width, height


def _enable_memory_optimizations(pipeline: StableDiffusionPipeline) -> None:
    pipeline.enable_attention_slicing()
    pipeline.enable_vae_slicing()
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except ImportError:
        logger.debug("image_utils.xformers_unavailable")


async def load_pipeline() -> StableDiffusionPipeline:
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    async with _load_lock:
        if _pipeline is not None:
            return _pipeline
        logger.info("image_utils.loading_pipeline", model_id=SD_MODEL_ID, device=SD_DEVICE)
        if SD_DEVICE == "cuda":
            torch_dtype = torch.float16
        elif SD_DEVICE == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float32

        safety_checker = None if not SD_SAFETY_CHECKER else None
        def _load():
            return StableDiffusionPipeline.from_pretrained(
                SD_MODEL_ID,
                torch_dtype=torch_dtype,
                safety_checker=safety_checker,
            )

        try:
            loop = asyncio.get_event_loop()
            pipeline = await loop.run_in_executor(None, _load)
            pipeline = pipeline.to(SD_DEVICE)
            _enable_memory_optimizations(pipeline)
            _pipeline = pipeline
            logger.info("image_utils.pipeline_loaded", device=SD_DEVICE, dtype=str(torch_dtype))
            return _pipeline
        except Exception as exc:
            logger.error("image_utils.load_failed", error=str(exc))
            raise RuntimeError(f"Failed to load Stable Diffusion pipeline: {exc}") from exc


def generate_image(
    prompt: str,
    output_path: str,
    width: int = SD_IMAGE_WIDTH,
    height: int = SD_IMAGE_HEIGHT,
    num_inference_steps: int = SD_INFERENCE_STEPS,
) -> str:
    if not prompt or not prompt.strip():
        logger.warning("image_utils.empty_prompt")
    out_path = Path(output_path)
    if out_path.is_dir():
        raise ValueError(f"output_path is a directory: {output_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = _validate_dimensions(width, height)

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        pipeline = loop.run_until_complete(load_pipeline())
        loop.close()
    except Exception as exc:
        logger.error("image_utils.pipeline_unavailable", error=str(exc))
        raise RuntimeError("Stable Diffusion pipeline unavailable") from exc

    current_width, current_height = width, height
    delay = 1.0

    for attempt in range(MAX_OOM_RETRIES):
        try:
            logger.info("image_utils.generation_attempt", attempt=attempt + 1, width=current_width, height=current_height)
            result = pipeline(prompt=prompt, width=current_width, height=current_height, num_inference_steps=num_inference_steps)
            image = result.images[0]
            image.save(output_path)
            logger.info("image_utils.generated", output_path=output_path)
            return str(output_path)
        except torch.cuda.OutOfMemoryError as oom:
            if attempt == MAX_OOM_RETRIES - 1:
                torch.cuda.empty_cache()
                logger.error("image_utils.oom_failed", error=str(oom))
                raise RuntimeError(f"OOM after {MAX_OOM_RETRIES} retries") from oom
            current_width = max(MIN_DIMENSION, (current_width // 2) // DIMENSION_MULTIPLE * DIMENSION_MULTIPLE)
            current_height = max(MIN_DIMENSION, (current_height // 2) // DIMENSION_MULTIPLE * DIMENSION_MULTIPLE)
            logger.warning("image_utils.oom_retry", new_width=current_width, new_height=current_height)
            torch.cuda.empty_cache()
            time.sleep(delay)
            delay *= RETRY_BACKOFF_FACTOR
        except Exception as exc:
            logger.error("image_utils.generation_error", error=str(exc))
            raise RuntimeError(f"Image generation failed: {exc}") from exc

    raise RuntimeError("Image generation failed after all retries")


async def generate_image_async(
    prompt: str,
    output_path: str,
    width: int = SD_IMAGE_WIDTH,
    height: int = SD_IMAGE_HEIGHT,
    num_inference_steps: int = SD_INFERENCE_STEPS,
) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_image, prompt, output_path, width, height, num_inference_steps)
