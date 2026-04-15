"""Reducer node for BWAgent."""

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List

import markdown
import structlog

from config.settings import BLOGS_DIR
from graph.state import GeneratedImage, GraphState, SectionDraft

logger = structlog.get_logger(__name__)


def _resolve_citation_markers(content: str, registry: Dict[str, str], section_id: str) -> str:
    missing = 0

    def replacer(match: re.Match) -> str:
        nonlocal missing
        key = match.group(0)
        if key in registry:
            return f"[{key}]({registry[key]})"
        missing += 1
        logger.warning("reducer.unresolved_citation", section_id=section_id, key=key)
        return ""

    return re.sub(r"\[SOURCE_\d+\]", replacer, content)


async def reducer_node(state: GraphState) -> Dict[str, Any]:
    blog_plan = state.get("blog_plan")
    if not blog_plan:
        logger.error("reducer_node.missing_blog_plan")
        return {"error": "Missing blog_plan", "final_blog_md": "", "final_blog_html": ""}

    section_drafts: List[SectionDraft] = state.get("section_drafts", [])
    generated_images: List[GeneratedImage] = state.get("generated_images", [])
    citation_registry: Dict[str, str] = state.get("citation_registry", {})
    run_id = state.get("run_id", "blog")

    image_by_section = {image.section_id: image.image_path for image in generated_images}
    section_order = {section.id: idx for idx, section in enumerate(blog_plan.sections)}
    sorted_drafts = sorted(section_drafts, key=lambda draft: section_order.get(draft.section_id, float("inf")))

    feature_image_md = ""
    if image_by_section.get("feature"):
        feature_image_md = f"![Feature image]({image_by_section['feature']})\n\n"

    sections_md: List[str] = []
    for draft in sorted_drafts:
        content = draft.content
        placeholder = f"[IMAGE_PLACEHOLDER_{draft.section_id}]"
        if image_by_section.get(draft.section_id):
            image_md = f"![{draft.title}]({image_by_section[draft.section_id]})"
        else:
            image_md = ""
        content = content.replace(placeholder, image_md)
        content = _resolve_citation_markers(content, citation_registry, draft.section_id)
        sections_md.append(f"## {draft.title}\n\n{content}")

    unique_urls = list(dict.fromkeys(citation_registry.values()))
    references_md = ""
    if unique_urls:
        references_md = "\n## References\n\n" + "\n".join(f"{idx + 1}. {url}" for idx, url in enumerate(unique_urls))

    full_md = f"# {blog_plan.blog_title}\n\n{feature_image_md}{'\n\n'.join(sections_md)}{references_md}".strip()

    try:
        html_content = await asyncio.to_thread(markdown.markdown, full_md, extensions=["tables", "fenced_code", "codehilite"])
    except Exception as exc:
        logger.error("reducer_node.html_failed", error=str(exc))
        html_content = ""

    try:
        BLOGS_DIR.mkdir(parents=True, exist_ok=True)
        md_path = BLOGS_DIR / f"{run_id}.md"
        html_path = BLOGS_DIR / f"{run_id}.html"
        await asyncio.to_thread(md_path.write_text, full_md, encoding="utf-8")
        await asyncio.to_thread(html_path.write_text, html_content, encoding="utf-8")
        logger.info("reducer_node.saved", md_path=str(md_path), html_path=str(html_path))
    except Exception as exc:
        logger.error("reducer_node.save_failed", error=str(exc))

    return {"final_blog_md": full_md, "final_blog_html": html_content}
