"""
tools/image_gen.py

Stable Diffusion v1.4 wrapper for the Autonomous AI Blog Generation System.
Loads the diffusion pipeline exactly once at module import time with correct
device/dtype detection, then exposes a synchronous generate_image() callable
from ThreadPoolExecutor contexts.

Also exports image_gen_tool (LangChain Tool) for use in Planner/Image Agent crews.

Stack: diffusers 0.29.2, torch (CUDA build), pathlib, logging
"""

# -----------------------------------------------------------------------
# MUST be set before ANY import of transformers or diffusers.
# Suppresses `Accessing __path__` aliasing warnings that fire at module-init
# time in transformers 4.45+ / diffusers combinations.
# Root-fix: pin transformers==4.44.2 + diffusers==0.29.2 in requirements.txt
# -----------------------------------------------------------------------
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import logging
from pathlib import Path
from typing import Tuple

import torch
from diffusers import StableDiffusionPipeline
from langchain_core.tools import Tool

from config import (
    SD_MODEL_ID,
    USE_GPU,
    IMG_SIZE,
    IMG_STEPS,
    OUTPUT_IMAGES,
)

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

# ----------------------------------------------------------------------
# Hardware detection — HARD GUARD (no silent fallback)
# ----------------------------------------------------------------------
_cuda_available = torch.cuda.is_available()

logger.info(
    "[image_gen] CUDA available: %s | USE_GPU config: %s",
    _cuda_available, USE_GPU
)

if USE_GPU and not _cuda_available:
    raise RuntimeError(
        "\n\n[image_gen] ❌ USE_GPU=True but torch.cuda.is_available() returned False.\n"
        "The installed PyTorch is a CPU-only build — image generation on CPU is ~50-100x slower\n"
        "and will break the 6-minute pipeline SLA.\n\n"
        "Fix options:\n"
        "  1. Reinstall PyTorch with CUDA support (recommended for RTX 3060+):\n"
        "       pip uninstall torch torchvision torchaudio -y\n"
        "       pip install torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cu121\n"
        "     Then verify: python -c \"import torch; print(torch.cuda.is_available())\"\n\n"
        "  2. If your machine has no NVIDIA GPU, set USE_GPU=False in config.py\n"
        "     (pipeline will run on CPU; expect 30-60 min per image).\n\n"
        "  3. If nvidia-smi works but CUDA is still False, your NVIDIA driver is too old.\n"
        "     CUDA 12.1 requires driver >= 525.x.\n"
    )

DEVICE: str = "cuda" if USE_GPU else "cpu"
DTYPE: torch.dtype = torch.float16 if DEVICE == "cuda" else torch.float32

logger.info("[image_gen] Selected device: %s (dtype=%s)", DEVICE, DTYPE)

# ----------------------------------------------------------------------
# Global singleton pipeline — loaded once, reused forever
# ----------------------------------------------------------------------
logger.info("[image_gen] Loading SD pipeline: %s (dtype=%s, device=%s)",
            SD_MODEL_ID, DTYPE, DEVICE)

try:
    _pipeline: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
        SD_MODEL_ID,
        torch_dtype=DTYPE,
        safety_checker=None,
        requires_safety_checker=False,
    )
    _pipeline = _pipeline.to(DEVICE)
    logger.info("[image_gen] Pipeline loaded successfully on %s", DEVICE)
except Exception as exc:
    logger.error("[image_gen] FATAL: Failed to load Stable Diffusion pipeline: %s", exc)
    _pipeline = None  # type: ignore[assignment]


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def generate_image(prompt: str, section_id: str) -> str:
    """
    Generate a PNG from *prompt* and save to a deterministic path.
    Returns absolute path to the saved PNG.
    Idempotent: reuses the file if it already exists.
    """
    if _pipeline is None:
        raise RuntimeError(
            "Stable Diffusion pipeline is not available (failed at import). "
            "Check logs and ensure model ID is correct."
        )

    out_dir = Path(OUTPUT_IMAGES)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"{section_id}.png"

    if file_path.exists():
        logger.info("[image_gen] Image already exists for %s — skipping generation.", section_id)
        return str(file_path.resolve())

    logger.info("[image_gen] Generating image for %s (%s steps, %s)",
                section_id, IMG_STEPS, IMG_SIZE)

    try:
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
        logger.error(
            "[image_gen] CUDA OOM for %s — reduce IMG_SIZE/IMG_STEPS in config.py: %s",
            section_id, oom
        )
        raise
    except Exception as exc:
        logger.error("[image_gen] Generation failed for %s: %s", section_id, exc)
        raise

    return str(file_path.resolve())


def get_image_metadata(section_id: str) -> Tuple[str, str]:
    """Return (file_path, alt_text) for a given section_id."""
    out_dir = Path(OUTPUT_IMAGES)
    file_path = out_dir / f"{section_id}.png"
    alt_text = f"Illustration for {section_id.replace('_', ' ')}"
    return str(file_path.resolve()), alt_text


# ----------------------------------------------------------------------
# LangChain Tool wrapper (required by planner.py + CrewAI agents)
# ----------------------------------------------------------------------

image_gen_tool = Tool(
    name="generate_image",
    func=generate_image,
    description=(
        "Generates a high-quality image using Stable Diffusion v1.4. "
        "Takes a detailed prompt and section_id. Saves the PNG to outputs/images/ "
        "with deterministic filename and returns the absolute file path. "
        "GPU/CPU aware, idempotent (reuses existing images)."
    ),
    return_direct=False,
)


# ----------------------------------------------------------------------
# Exports
# ----------------------------------------------------------------------
__all__ = ["image_gen_tool", "generate_image", "get_image_metadata"]


# ----------------------------------------------------------------------
# Self-test (run with: python -m tools.image_gen)
# ----------------------------------------------------------------------
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
    