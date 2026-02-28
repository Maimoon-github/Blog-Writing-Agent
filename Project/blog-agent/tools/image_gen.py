"""
tools/image_gen.py
------------------
Manages the Stable Diffusion v1.4 image generation pipeline for blog-agent.

Provides:
  - load_pipeline()   : loads and caches a StableDiffusionPipeline singleton.
  - generate_image()  : runs inference and saves the result as a PNG file.
"""

import os
import pathlib

import structlog
import torch
from diffusers import StableDiffusionPipeline

from app.config import (
    SD_DEVICE,
    SD_IMAGE_HEIGHT,
    SD_IMAGE_WIDTH,
    SD_INFERENCE_STEPS,
    SD_MODEL_ID,
)

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Singleton cache — populated on first call to load_pipeline()
# ---------------------------------------------------------------------------
_pipeline: StableDiffusionPipeline | None = None


# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------
def load_pipeline() -> StableDiffusionPipeline:
    """Load the Stable Diffusion pipeline, caching it as a module-level singleton.

    Returns
    -------
    StableDiffusionPipeline
        The loaded (and possibly cached) pipeline instance.
    """
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    logger.info("loading_stable_diffusion_pipeline", model_id=SD_MODEL_ID, device=SD_DEVICE)

    torch_dtype = torch.float16 if SD_DEVICE == "cuda" else torch.float32

    pipeline = StableDiffusionPipeline.from_pretrained(
        SD_MODEL_ID,
        torch_dtype=torch_dtype,
    )
    pipeline = pipeline.to(SD_DEVICE)
    pipeline.enable_attention_slicing()

    _pipeline = pipeline

    logger.info(
        "stable_diffusion_pipeline_loaded",
        model_id=SD_MODEL_ID,
        device=SD_DEVICE,
        dtype=str(torch_dtype),
    )

    return _pipeline


# ---------------------------------------------------------------------------
# Image generator
# ---------------------------------------------------------------------------
def generate_image(
    prompt: str,
    output_path: str,
    width: int = SD_IMAGE_WIDTH,
    height: int = SD_IMAGE_HEIGHT,
    num_inference_steps: int = SD_INFERENCE_STEPS,
) -> str:
    """Generate an image from *prompt* and save it as a PNG at *output_path*.

    Parameters
    ----------
    prompt:
        The text prompt used to guide image generation.
    output_path:
        Destination file path for the generated PNG image.
    width:
        Output image width in pixels (default: ``SD_IMAGE_WIDTH``).
    height:
        Output image height in pixels (default: ``SD_IMAGE_HEIGHT``).
    num_inference_steps:
        Number of denoising steps (default: ``SD_INFERENCE_STEPS``).

    Returns
    -------
    str
        The resolved *output_path* string where the image was saved.

    Notes
    -----
    If a ``torch.cuda.OutOfMemoryError`` is encountered, the function
    automatically retries at half resolution (``width // 2``, ``height // 2``).
    """
    log = logger.bind(prompt=prompt, output_path=output_path, width=width, height=height)
    log.info("image_generation_started", num_inference_steps=num_inference_steps)

    pipeline = load_pipeline()

    def _run(w: int, h: int) -> object:
        return pipeline(
            prompt=prompt,
            width=w,
            height=h,
            num_inference_steps=num_inference_steps,
        )

    def _save(image, path: str) -> None:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        image.save(path)

    try:
        result = _run(width, height)
        image = result.images[0]
        _save(image, output_path)

    except torch.cuda.OutOfMemoryError:
        half_w, half_h = width // 2, height // 2
        logger.warning(
            "oom_retrying_at_half_resolution",
            original_width=width,
            original_height=height,
            retry_width=half_w,
            retry_height=half_h,
        )
        result = _run(half_w, half_h)
        image = result.images[0]
        _save(image, output_path)

    log.info("image_generation_completed", output_path=output_path)

    return str(output_path)