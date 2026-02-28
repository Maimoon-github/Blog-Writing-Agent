"""
agents/reducer.py
-----------------
LangGraph final assembler node for blog-agent.

Assembles the complete blog post from all component outputs stored in GraphState,
converts it to HTML, and persists both artefacts to BLOGS_DIR.
"""

from __future__ import annotations

import re
from pathlib import Path

import markdown
import structlog

from app.config import BLOGS_DIR
from graph.state import GraphState

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _replace_image_placeholder(
    content: str,
    section_id: str,
    title: str,
    image_by_section: dict[str, str],
) -> str:
    """Replace ``[IMAGE_PLACEHOLDER_<section_id>]`` with a Markdown image tag.

    If no image exists for the section the placeholder is silently removed.
    """
    placeholder = f"[IMAGE_PLACEHOLDER_{section_id}]"
    image_path = image_by_section.get(section_id)

    if image_path:
        replacement = f"![{title}]({image_path})"
    else:
        replacement = ""

    return content.replace(placeholder, replacement)


def _replace_source_markers(
    content: str,
    citation_registry: dict[str, str],
) -> str:
    """Replace every ``[SOURCE_N]`` marker with a Markdown inline link.

    Unknown markers are removed rather than left as-is to keep the output clean.
    """
    def _substitutor(match: re.Match) -> str:  # noqa: ANN202
        key = match.group(0)          # e.g. "[SOURCE_3]"
        url = citation_registry.get(key)
        if url:
            return f"[{key}]({url})"
        # Remove unresolvable markers
        return ""

    return re.sub(r"\[SOURCE_\d+\]", _substitutor, content)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def reducer_node(state: GraphState) -> dict:  # type: ignore[return]
    """Assemble the final blog post and persist it to disk.

    Steps
    -----
    1. Determine section order from ``blog_plan``.
    2. Sort ``section_drafts`` accordingly.
    3. Build lookup maps for images and citations.
    4. Locate the feature image (``section_id == "feature"``).
    5. Process each draft: replace image placeholders and source markers.
    6. Build a ``## References`` appendix from unique citation URLs.
    7. Assemble the full Markdown document.
    8. Convert to HTML via the ``markdown`` library.
    9. Save ``.md`` and ``.html`` files to ``BLOGS_DIR``.
    10. Return ``{"final_blog_md": ..., "final_blog_html": ...}``.
    """

    # ------------------------------------------------------------------
    # 1. Section order index
    # ------------------------------------------------------------------
    blog_plan = state["blog_plan"]
    section_order: dict[str, int] = {
        s.id: i for i, s in enumerate(blog_plan.sections)
    }

    # ------------------------------------------------------------------
    # 2. Sort section drafts
    # ------------------------------------------------------------------
    sorted_drafts = sorted(
        state.get("section_drafts", []),  # type: ignore[arg-type]
        key=lambda draft: section_order.get(draft.section_id, 999),
    )

    # ------------------------------------------------------------------
    # 3. Build lookup maps
    # ------------------------------------------------------------------
    generated_images = state.get("generated_images", [])  # type: ignore[assignment]

    image_by_section: dict[str, str] = {
        img.section_id: img.image_path
        for img in generated_images
    }

    citation_registry: dict[str, str] = state.get("citation_registry", {})  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # 4. Feature image
    # ------------------------------------------------------------------
    feature_img_md = ""
    for img in generated_images:
        if img.section_id == "feature":
            feature_img_md = f"![Feature Image]({img.image_path})"
            break

    # ------------------------------------------------------------------
    # 5 & 6. Process each draft into Markdown sections
    # ------------------------------------------------------------------
    sections_md: list[str] = []

    for draft in sorted_drafts:
        content: str = draft.content

        # 5a. Replace image placeholder (or remove it if no image)
        content = _replace_image_placeholder(
            content,
            section_id=draft.section_id,
            title=draft.title,
            image_by_section=image_by_section,
        )

        # 5b. Replace [SOURCE_N] markers with Markdown inline links
        content = _replace_source_markers(content, citation_registry)

        # 5c. Prepend the section H2 heading
        content = f"## {draft.title}\n\n{content}"

        sections_md.append(content)

    # ------------------------------------------------------------------
    # 7. References section
    # ------------------------------------------------------------------
    # dict.fromkeys preserves insertion order while deduplicating
    unique_urls: list[str] = list(dict.fromkeys(citation_registry.values()))
    references_md = (
        "## References\n\n"
        + "\n".join(f"{i + 1}. {url}" for i, url in enumerate(unique_urls))
    )

    # ------------------------------------------------------------------
    # 8. Assemble full Markdown
    # ------------------------------------------------------------------
    full_md = (
        f"# {blog_plan.blog_title}\n\n"
        f"{feature_img_md}\n\n"
        + "\n\n".join(sections_md)
        + "\n\n"
        + references_md
    )

    # ------------------------------------------------------------------
    # 9. Convert to HTML
    # ------------------------------------------------------------------
    html_content: str = markdown.markdown(
        full_md,
        extensions=["tables", "fenced_code"],
    )

    # ------------------------------------------------------------------
    # 10. Persist artefacts
    # ------------------------------------------------------------------
    run_id: str = state.get("run_id", "blog")  # type: ignore[assignment]

    BLOGS_DIR.mkdir(parents=True, exist_ok=True)

    md_path: Path = BLOGS_DIR / f"{run_id}.md"
    html_path: Path = BLOGS_DIR / f"{run_id}.html"

    md_path.write_text(full_md, encoding="utf-8")
    html_path.write_text(html_content, encoding="utf-8")

    # ------------------------------------------------------------------
    # 11. Log
    # ------------------------------------------------------------------
    logger.info(
        "reducer_node: blog artefacts saved",
        md_path=str(md_path),
        html_path=str(html_path),
        sections=len(sections_md),
        citations=len(unique_urls),
    )

    # ------------------------------------------------------------------
    # 12. Return
    # ------------------------------------------------------------------
    return {
        "final_blog_md": full_md,
        "final_blog_html": html_content,
    }