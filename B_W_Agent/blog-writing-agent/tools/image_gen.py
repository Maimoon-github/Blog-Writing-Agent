"""
tools/image_gen.py

Stable Diffusion v1.4 wrapper for the Autonomous AI Blog Generation System.
Loads the diffusion pipeline exactly once at module import time with correct
device/dtype detection, then exposes a synchronous generate_image() callable
from ThreadPoolExecutor contexts.

Stack: diffusers 0.37.1, torch, pathlib, logging
"""

import logging
import os
from pathlib import Path
from typing import Tuple

import torch
from diffusers import StableDiffusionPipeline

# ---------------------------------------------------------------------------
# Import tunables from central config — NEVER hardcode here
# ---------------------------------------------------------------------------
from config import (
    SD_MODEL_ID,
    USE_GPU,
    IMG_SIZE,
    IMG_STEPS,
    OUTPUT_IMAGES,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

# ---------------------------------------------------------------------------
# Hardware detection (executed once at import time)
# ---------------------------------------------------------------------------
_cuda_available = torch.cuda.is_available()
DEVICE: str = "cuda" if (_cuda_available and USE_GPU) else "cpu"
DTYPE: torch.dtype = torch.float16 if DEVICE == "cuda" else torch.float32

logger.info("[image_gen] CUDA available: %s | USE_GPU config: %s | Selected device: %s",
            _cuda_available, USE_GPU, DEVICE)

# ---------------------------------------------------------------------------
# Global singleton pipeline — loaded once, reused forever
# ---------------------------------------------------------------------------
# Production pattern: load at module level to avoid repeated expensive
# initialisation inside ThreadPoolExecutor workers or parallel agent crews.
# ---------------------------------------------------------------------------
logger.info("[image_gen] Loading SD pipeline: %s (dtype=%s, device=%s)",
            SD_MODEL_ID, DTYPE, DEVICE)

try:
    _pipeline: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
        SD_MODEL_ID,
        torch_dtype=DTYPE,
        safety_checker=None,          # disabled for local/offline use
        requires_safety_checker=False,
    )
    _pipeline = _pipeline.to(DEVICE)
    logger.info("[image_gen] Pipeline loaded successfully on %s", DEVICE)
except Exception as exc:
    logger.error("[image_gen] FATAL: Failed to load Stable Diffusion pipeline: %s", exc)
    _pipeline = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_image(prompt: str, section_id: str) -> str:
    """
    Generate a 512×512 (GPU) or 384×384 (CPU) PNG from *prompt* and save it
    to a deterministic path derived from *section_id*.

    Parameters
    ----------
    prompt : str
        Optimised image prompt (from Planner / Image Agent).
    section_id : str
        Identifer such as "feature", "section_1", "section_2". Used to build
        the deterministic filename.

    Returns
    -------
    str
        Absolute path to the saved PNG file.

    Raises
    ------
    RuntimeError
        If the pipeline failed to load at import time.
    """
    if _pipeline is None:
        raise RuntimeError(
            "Stable Diffusion pipeline is not available (failed at import). "
            "Check logs and ensure model ID is correct."
        )

    # Deterministic path — prevents race conditions when multiple agents
    # generate images concurrently.
    out_dir = Path(OUTPUT_IMAGES)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"{section_id}.png"

    # Avoid regenerating if file already exists (idempotent for retries)
    if file_path.exists():
        logger.info("[image_gen] Image already exists for %s — skipping generation.", section_id)
        return str(file_path.resolve())

    logger.info("[image_gen] Generating image for %s (%s steps, %s)",
                section_id, IMG_STEPS, IMG_SIZE)

    try:
        # width/height come from config.IMG_SIZE tuple (512,512) GPU or (384,384) CPU.
        width, height = IMG_SIZE
        result = _pipeline(
            prompt,
            num_inference_steps=IMG_STEPS,
            width=width,
            height=height,
            guidance_scale=7.5,
        )
        image = result.images[0]
        image.save(file_path)
        logger.info("[image_gen] Saved %s", file_path)
    except torch.cuda.OutOfMemoryError as oom:
        logger.error("[image_gen] CUDA OOM for %s — consider reducing IMG_SIZE/IMG_STEPS in config.py: %s",
                     section_id, oom)
        raise
    except Exception as exc:
        logger.error("[image_gen] Generation failed for %s: %s", section_id, exc)
        raise

    return str(file_path.resolve())


def get_image_metadata(section_id: str) -> Tuple[str, str]:
    """
    Return (file_path, alt_text) for a given section_id.
    Useful for the Reducer agent when assembling final Markdown.

    Returns
    -------
    Tuple[str, str]
        (absolute_path, alt_text) where alt_text defaults to the section_id.
    """
    out_dir = Path(OUTPUT_IMAGES)
    file_path = out_dir / f"{section_id}.png"
    alt_text = f"Illustration for {section_id.replace('_', ' ')}"
    return str(file_path.resolve()), alt_text


# ---------------------------------------------------------------------------
# Self-test (run with: python -m tools.image_gen)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_prompt = (
        "A futuristic robot writing a blog post at a holographic desk, "
        "digital art, highly detailed, 8k, cinematic lighting"
    )
    test_id = "feature"
    print(f"[test] Generating test image for '{test_id}' ...")
    generated_path = generate_image(test_prompt, test_id)
    assert os.path.isfile(generated_path), f"File not found: {generated_path}"
    print(f"[test] SUCCESS — image saved to: {generated_path}")