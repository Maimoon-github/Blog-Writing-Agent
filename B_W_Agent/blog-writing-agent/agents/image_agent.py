"""agents/image_agent.py
Lightweight Image Agent Node (roadmap Step 9).
Runs Stable Diffusion generation inside ThreadPoolExecutor (off async event loop).
Returns ImageResult or Tier-3 placeholder on failure.
Pure Python node for flat asyncio.gather in graph.py.
CrewAI v1.14.2 + verified 2026 non-blocking pattern.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

from config import IMG_SIZE
from schemas import Section, ImageResult
from tools.image_gen import generate_image


# ----------------------------------------------------------------------
# Main lightweight node (called via asyncio.gather in graph.py)
# ----------------------------------------------------------------------
def image_agent_node(section: Section) -> Dict[str, Any]:
    """image_agent_node(section: Section) -> {"generated_images": [ImageResult]}"""
    
    if not isinstance(section, Section):
        raise ValueError("image_agent_node: input must be a valid Section object")
    
    if not section.id or not section.image_prompt:
        raise ValueError(
            f"image_agent_node: section missing id or image_prompt. "
            f"Got id={getattr(section, 'id', 'None')}, prompt={getattr(section, 'image_prompt', 'None')}"
        )

    # Run heavy SD inference off the main async event loop
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            future = executor.submit(
                generate_image,
                section.image_prompt,
                section.id
            )
            # 180-second safety timeout (prevents hanging the entire gather)
            file_path: str = future.result(timeout=180)

        # Success path
        image_result = ImageResult(
            section_id=section.id,
            prompt=section.image_prompt,
            file_path=file_path,
            alt_text=f"Visual illustration for section: {section.title}",
            size=IMG_SIZE
        )

    except Exception as e:
        # Tier-3 graceful fallback – Reducer will replace PLACEHOLDER cleanly
        print(f"[image_agent_node] Generation failed for section '{section.id}': {type(e).__name__} - {e}")
        image_result = ImageResult(
            section_id=section.id,
            prompt=section.image_prompt,
            file_path="PLACEHOLDER",
            alt_text=f"Placeholder for: {section.title} (image generation failed)",
            size=(400, 400)
        )

    return {
        "generated_images": [image_result]
    }