# """
# agents/reducer.py
# -----------------
# LangGraph final assembler node for blog-agent.

# Assembles the complete blog post from all component outputs stored in GraphState,
# converts it to HTML, and persists both artefacts to BLOGS_DIR.
# """

# from __future__ import annotations

# import re
# from pathlib import Path

# import markdown
# import structlog

# from app.config import BLOGS_DIR
# from graph.state import GraphState

# logger = structlog.get_logger(__name__)


# # ---------------------------------------------------------------------------
# # Helper utilities
# # ---------------------------------------------------------------------------

# def _replace_image_placeholder(
#     content: str,
#     section_id: str,
#     title: str,
#     image_by_section: dict[str, str],
# ) -> str:
#     """Replace ``[IMAGE_PLACEHOLDER_<section_id>]`` with a Markdown image tag.

#     If no image exists for the section the placeholder is silently removed.
#     """
#     placeholder = f"[IMAGE_PLACEHOLDER_{section_id}]"
#     image_path = image_by_section.get(section_id)

#     if image_path:
#         replacement = f"![{title}]({image_path})"
#     else:
#         replacement = ""

#     return content.replace(placeholder, replacement)


# def _replace_source_markers(
#     content: str,
#     citation_registry: dict[str, str],
# ) -> str:
#     """Replace every ``[SOURCE_N]`` marker with a Markdown inline link.

#     Unknown markers are removed rather than left as-is to keep the output clean.
#     """
#     def _substitutor(match: re.Match) -> str:  # noqa: ANN202
#         key = match.group(0)          # e.g. "[SOURCE_3]"
#         url = citation_registry.get(key)
#         if url:
#             return f"[{key}]({url})"
#         # Remove unresolvable markers
#         return ""

#     return re.sub(r"\[SOURCE_\d+\]", _substitutor, content)


# # ---------------------------------------------------------------------------
# # Node
# # ---------------------------------------------------------------------------

# def reducer_node(state: GraphState) -> dict:  # type: ignore[return]
#     """Assemble the final blog post and persist it to disk.

#     Steps
#     -----
#     1. Determine section order from ``blog_plan``.
#     2. Sort ``section_drafts`` accordingly.
#     3. Build lookup maps for images and citations.
#     4. Locate the feature image (``section_id == "feature"``).
#     5. Process each draft: replace image placeholders and source markers.
#     6. Build a ``## References`` appendix from unique citation URLs.
#     7. Assemble the full Markdown document.
#     8. Convert to HTML via the ``markdown`` library.
#     9. Save ``.md`` and ``.html`` files to ``BLOGS_DIR``.
#     10. Return ``{"final_blog_md": ..., "final_blog_html": ...}``.
#     """

#     # ------------------------------------------------------------------
#     # 1. Section order index
#     # ------------------------------------------------------------------
#     blog_plan = state["blog_plan"]
#     section_order: dict[str, int] = {
#         s.id: i for i, s in enumerate(blog_plan.sections)
#     }

#     # ------------------------------------------------------------------
#     # 2. Sort section drafts
#     # ------------------------------------------------------------------
#     sorted_drafts = sorted(
#         state.get("section_drafts", []),  # type: ignore[arg-type]
#         key=lambda draft: section_order.get(draft.section_id, 999),
#     )

#     # ------------------------------------------------------------------
#     # 3. Build lookup maps
#     # ------------------------------------------------------------------
#     generated_images = state.get("generated_images", [])  # type: ignore[assignment]

#     image_by_section: dict[str, str] = {
#         img.section_id: img.image_path
#         for img in generated_images
#     }

#     citation_registry: dict[str, str] = state.get("citation_registry", {})  # type: ignore[assignment]

#     # ------------------------------------------------------------------
#     # 4. Feature image
#     # ------------------------------------------------------------------
#     feature_img_md = ""
#     for img in generated_images:
#         if img.section_id == "feature":
#             feature_img_md = f"![Feature Image]({img.image_path})"
#             break

#     # ------------------------------------------------------------------
#     # 5 & 6. Process each draft into Markdown sections
#     # ------------------------------------------------------------------
#     sections_md: list[str] = []

#     for draft in sorted_drafts:
#         content: str = draft.content

#         # 5a. Replace image placeholder (or remove it if no image)
#         content = _replace_image_placeholder(
#             content,
#             section_id=draft.section_id,
#             title=draft.title,
#             image_by_section=image_by_section,
#         )

#         # 5b. Replace [SOURCE_N] markers with Markdown inline links
#         content = _replace_source_markers(content, citation_registry)

#         # 5c. Prepend the section H2 heading
#         content = f"## {draft.title}\n\n{content}"

#         sections_md.append(content)

#     # ------------------------------------------------------------------
#     # 7. References section
#     # ------------------------------------------------------------------
#     # dict.fromkeys preserves insertion order while deduplicating
#     unique_urls: list[str] = list(dict.fromkeys(citation_registry.values()))
#     references_md = (
#         "## References\n\n"
#         + "\n".join(f"{i + 1}. {url}" for i, url in enumerate(unique_urls))
#     )

#     # ------------------------------------------------------------------
#     # 8. Assemble full Markdown
#     # ------------------------------------------------------------------
#     full_md = (
#         f"# {blog_plan.blog_title}\n\n"
#         f"{feature_img_md}\n\n"
#         + "\n\n".join(sections_md)
#         + "\n\n"
#         + references_md
#     )

#     # ------------------------------------------------------------------
#     # 9. Convert to HTML
#     # ------------------------------------------------------------------
#     html_content: str = markdown.markdown(
#         full_md,
#         extensions=["tables", "fenced_code"],
#     )

#     # ------------------------------------------------------------------
#     # 10. Persist artefacts
#     # ------------------------------------------------------------------
#     run_id: str = state.get("run_id", "blog")  # type: ignore[assignment]

#     BLOGS_DIR.mkdir(parents=True, exist_ok=True)

#     md_path: Path = BLOGS_DIR / f"{run_id}.md"
#     html_path: Path = BLOGS_DIR / f"{run_id}.html"

#     md_path.write_text(full_md, encoding="utf-8")
#     html_path.write_text(html_content, encoding="utf-8")

#     # ------------------------------------------------------------------
#     # 11. Log
#     # ------------------------------------------------------------------
#     logger.info(
#         "reducer_node: blog artefacts saved",
#         md_path=str(md_path),
#         html_path=str(html_path),
#         sections=len(sections_md),
#         citations=len(unique_urls),
#     )

#     # ------------------------------------------------------------------
#     # 12. Return
#     # ------------------------------------------------------------------
#     return {
#         "final_blog_md": full_md,
#         "final_blog_html": html_content,
#     }




































"""LangGraph node that assembles the final blog post from components and saves to disk.

This node is the final step in the blog generation pipeline. It combines section drafts,
images, and citations into a complete Markdown document, converts it to HTML,
and persists both formats to disk. It handles missing data gracefully and logs
every step with structured logging.

Both synchronous and asynchronous versions are provided for compatibility with
different execution contexts. The async version offloads blocking I/O and
Markdown conversion to thread pools.
"""

import asyncio
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

import markdown
import structlog

from graph.state import GraphState, SectionDraft, GeneratedImage
from app.config import BLOGS_DIR

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Synchronous version (for LangGraph sync pipelines)
# ---------------------------------------------------------------------------
def reducer_node(state: GraphState) -> Dict[str, Any]:
    """
    Assemble the final blog post from all components and save to disk (synchronous).

    Args:
        state: The final graph state, expected to contain:
            - blog_plan: BlogPlan (with blog_title and sections)
            - section_drafts: List[SectionDraft]
            - generated_images: List[GeneratedImage]
            - citation_registry: Dict[str, str]  # mapping "[SOURCE_N]" -> URL
            - run_id: str (optional, defaults to "blog")

    Returns:
        A dictionary with keys:
            - 'final_blog_md': str (Markdown content)
            - 'final_blog_html': str (HTML content)
        In case of critical errors, returns an empty dict with an 'error' key.
    """
    # ----------------------------------------------------------------------
    # 1. Extract required data with validation
    # ----------------------------------------------------------------------
    blog_plan = state.get("blog_plan")
    if not blog_plan:
        logger.error("reducer_node.missing_blog_plan")
        return {"error": "Missing blog_plan", "final_blog_md": "", "final_blog_html": ""}

    section_drafts: List[SectionDraft] = state.get("section_drafts", [])
    generated_images: List[GeneratedImage] = state.get("generated_images", [])
    citation_registry: Dict[str, str] = state.get("citation_registry", {})
    run_id: str = state.get("run_id", "blog")

    # Log assembly start
    logger.info(
        "reducer_node.starting",
        run_id=run_id,
        section_count=len(section_drafts),
        image_count=len(generated_images),
        citation_count=len(citation_registry),
    )

    # ----------------------------------------------------------------------
    # 2. Build lookup structures
    # ----------------------------------------------------------------------
    # Section order from blog plan (preserve original plan ordering)
    try:
        section_order = {s.id: idx for idx, s in enumerate(blog_plan.sections)}
    except AttributeError:
        logger.error("reducer_node.invalid_blog_plan", blog_plan=type(blog_plan).__name__)
        return {"error": "Invalid blog_plan structure", "final_blog_md": "", "final_blog_html": ""}

    # Image lookup by section_id
    image_by_section = {img.section_id: img.image_path for img in generated_images}

    # ----------------------------------------------------------------------
    # 3. Sort section drafts according to blog plan
    # ----------------------------------------------------------------------
    try:
        sorted_drafts = sorted(
            section_drafts,
            key=lambda draft: section_order.get(draft.section_id, float('inf')),
        )
    except Exception as e:
        logger.error("reducer_node.sort_failed", error=str(e))
        sorted_drafts = section_drafts  # fallback to original order

    logger.debug("reducer_node.sorted_drafts", draft_ids=[d.section_id for d in sorted_drafts])

    # ----------------------------------------------------------------------
    # 4. Process feature image (if any)
    # ----------------------------------------------------------------------
    feature_img_md = ""
    for img in generated_images:
        if img.section_id == "feature":
            feature_img_md = f"![Feature Image]({img.image_path})"
            break
    if not feature_img_md:
        logger.debug("reducer_node.no_feature_image")

    # ----------------------------------------------------------------------
    # 5. Process each section draft
    # ----------------------------------------------------------------------
    sections_md: List[str] = []
    missing_images = 0
    missing_citations = 0

    for draft in sorted_drafts:
        content = draft.content

        # 5a. Replace image placeholder
        placeholder = f"[IMAGE_PLACEHOLDER_{draft.section_id}]"
        image_path = image_by_section.get(draft.section_id)
        if image_path:
            replacement = f"![{draft.title}]({image_path})"
        else:
            replacement = ""
            if placeholder in content:
                missing_images += 1
                logger.debug(
                    "reducer_node.missing_section_image",
                    section_id=draft.section_id,
                )
        content = content.replace(placeholder, replacement)

        # 5b. Replace citation markers
        def replace_citation(match: re.Match) -> str:
            key = match.group(0)  # e.g. "[SOURCE_3]"
            url = citation_registry.get(key)
            if url:
                return f"[{key}]({url})"
            nonlocal missing_citations
            missing_citations += 1
            logger.debug(
                "reducer_node.unresolved_citation",
                section_id=draft.section_id,
                citation_key=key,
            )
            return ""  # remove unresolvable marker

        content = re.sub(r'\[SOURCE_\d+\]', replace_citation, content)

        # 5c. Prepend section heading
        section_md = f"## {draft.title}\n\n{content}"
        sections_md.append(section_md)

    # ----------------------------------------------------------------------
    # 6. Build references section from unique URLs
    # ----------------------------------------------------------------------
    unique_urls: List[str] = list(dict.fromkeys(citation_registry.values()))  # deduplicate, preserve order
    if unique_urls:
        refs = "## References\n\n" + "\n".join(f"{i+1}. {url}" for i, url in enumerate(unique_urls))
    else:
        refs = ""
        logger.debug("reducer_node.no_references")

    # ----------------------------------------------------------------------
    # 7. Assemble full Markdown
    # ----------------------------------------------------------------------
    full_md = (
        f"# {blog_plan.blog_title}\n\n"
        f"{feature_img_md}\n\n"
        f"{''.join(sections_md)}\n\n"
        f"{refs}"
    ).strip()

    # ----------------------------------------------------------------------
    # 8. Convert Markdown to HTML
    # ----------------------------------------------------------------------
    try:
        html_content = markdown.markdown(
            full_md,
            extensions=['tables', 'fenced_code', 'codehilite']
        )
    except Exception as e:
        logger.error("reducer_node.markdown_conversion_failed", error=str(e))
        html_content = ""  # Return empty HTML but keep Markdown

    # ----------------------------------------------------------------------
    # 9. Save files to disk
    # ----------------------------------------------------------------------
    try:
        BLOGS_DIR.mkdir(parents=True, exist_ok=True)
        md_path = BLOGS_DIR / f"{run_id}.md"
        html_path = BLOGS_DIR / f"{run_id}.html"

        md_path.write_text(full_md, encoding='utf-8')
        html_path.write_text(html_content, encoding='utf-8')

        logger.info(
            "reducer_node.files_saved",
            md_path=str(md_path.absolute()),
            html_path=str(html_path.absolute()),
        )
    except Exception as e:
        logger.error("reducer_node.file_write_failed", error=str(e))
        # Continue – still return content

    # ----------------------------------------------------------------------
    # 10. Final log and return
    # ----------------------------------------------------------------------
    logger.info(
        "reducer_node.completed",
        run_id=run_id,
        sections_processed=len(sections_md),
        unique_urls=len(unique_urls),
        missing_images=missing_images,
        missing_citations=missing_citations,
    )

    return {
        "final_blog_md": full_md,
        "final_blog_html": html_content,
    }


# ---------------------------------------------------------------------------
# Asynchronous version (for LangGraph async pipelines)
# ---------------------------------------------------------------------------
async def reducer_node_async(state: GraphState) -> Dict[str, Any]:
    """
    Asynchronous version of the reducer node.

    Offloads blocking I/O (file writes) and CPU‑intensive operations
    (Markdown conversion) to a thread pool to avoid blocking the event loop.
    All other processing (string manipulation, dict lookups) runs in the
    event loop as it is fast and non‑blocking.

    Args and return value are identical to `reducer_node`.
    """
    # 1. Extract required data with validation (same as sync)
    blog_plan = state.get("blog_plan")
    if not blog_plan:
        logger.error("reducer_node_async.missing_blog_plan")
        return {"error": "Missing blog_plan", "final_blog_md": "", "final_blog_html": ""}

    section_drafts: List[SectionDraft] = state.get("section_drafts", [])
    generated_images: List[GeneratedImage] = state.get("generated_images", [])
    citation_registry: Dict[str, str] = state.get("citation_registry", {})
    run_id: str = state.get("run_id", "blog")

    logger.info(
        "reducer_node_async.starting",
        run_id=run_id,
        section_count=len(section_drafts),
        image_count=len(generated_images),
        citation_count=len(citation_registry),
    )

    # 2. Build lookup structures (fast, stay in event loop)
    try:
        section_order = {s.id: idx for idx, s in enumerate(blog_plan.sections)}
    except AttributeError:
        logger.error("reducer_node_async.invalid_blog_plan", blog_plan=type(blog_plan).__name__)
        return {"error": "Invalid blog_plan structure", "final_blog_md": "", "final_blog_html": ""}

    image_by_section = {img.section_id: img.image_path for img in generated_images}

    # 3. Sort drafts (fast)
    try:
        sorted_drafts = sorted(
            section_drafts,
            key=lambda draft: section_order.get(draft.section_id, float('inf')),
        )
    except Exception as e:
        logger.error("reducer_node_async.sort_failed", error=str(e))
        sorted_drafts = section_drafts

    # 4. Process feature image (fast)
    feature_img_md = ""
    for img in generated_images:
        if img.section_id == "feature":
            feature_img_md = f"![Feature Image]({img.image_path})"
            break

    # 5. Process each section draft (string operations, fast)
    sections_md: List[str] = []
    missing_images = 0
    missing_citations = 0

    for draft in sorted_drafts:
        content = draft.content

        # Replace image placeholder
        placeholder = f"[IMAGE_PLACEHOLDER_{draft.section_id}]"
        image_path = image_by_section.get(draft.section_id)
        if image_path:
            replacement = f"![{draft.title}]({image_path})"
        else:
            replacement = ""
            if placeholder in content:
                missing_images += 1
        content = content.replace(placeholder, replacement)

        # Replace citation markers
        def replace_citation(match: re.Match) -> str:
            key = match.group(0)
            url = citation_registry.get(key)
            if url:
                return f"[{key}]({url})"
            missing_citations += 1
            return ""

        content = re.sub(r'\[SOURCE_\d+\]', replace_citation, content)
        section_md = f"## {draft.title}\n\n{content}"
        sections_md.append(section_md)

    # 6. Build references (fast)
    unique_urls: List[str] = list(dict.fromkeys(citation_registry.values()))
    if unique_urls:
        refs = "## References\n\n" + "\n".join(f"{i+1}. {url}" for i, url in enumerate(unique_urls))
    else:
        refs = ""

    # 7. Assemble full Markdown (fast)
    full_md = (
        f"# {blog_plan.blog_title}\n\n"
        f"{feature_img_md}\n\n"
        f"{''.join(sections_md)}\n\n"
        f"{refs}"
    ).strip()

    # 8. Convert Markdown to HTML (CPU‑intensive, run in thread pool)
    try:
        html_content = await asyncio.to_thread(
            markdown.markdown,
            full_md,
            extensions=['tables', 'fenced_code', 'codehilite']
        )
    except Exception as e:
        logger.error("reducer_node_async.markdown_conversion_failed", error=str(e))
        html_content = ""

    # 9. Save files to disk (blocking I/O, run in thread pool)
    try:
        # Ensure directory exists (fast, but we do it synchronously; could also thread)
        BLOGS_DIR.mkdir(parents=True, exist_ok=True)
        md_path = BLOGS_DIR / f"{run_id}.md"
        html_path = BLOGS_DIR / f"{run_id}.html"

        # Write both files concurrently
        await asyncio.gather(
            asyncio.to_thread(md_path.write_text, full_md, encoding='utf-8'),
            asyncio.to_thread(html_path.write_text, html_content, encoding='utf-8'),
        )

        logger.info(
            "reducer_node_async.files_saved",
            md_path=str(md_path.absolute()),
            html_path=str(html_path.absolute()),
        )
    except Exception as e:
        logger.error("reducer_node_async.file_write_failed", error=str(e))

    # 10. Final log and return
    logger.info(
        "reducer_node_async.completed",
        run_id=run_id,
        sections_processed=len(sections_md),
        unique_urls=len(unique_urls),
        missing_images=missing_images,
        missing_citations=missing_citations,
    )

    return {
        "final_blog_md": full_md,
        "final_blog_html": html_content,
    }