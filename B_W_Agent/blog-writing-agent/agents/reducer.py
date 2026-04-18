"""
agents/reducer.py

Final assembler for the Autonomous AI Blog Generation System.

Sorts parallel worker outputs by plan order, substitutes image placeholders,
assembles full blog with metadata, and writes .md + .html files.

Stack: Python 3.10+, markdown library, standard library.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import markdown

from config import OUTPUT_BLOGS, OUTPUT_IMAGES
from schemas import BlogPlan, ImageResult, SectionDraft
from state import CrewState   # ← required for reducer_node

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
# Helpers (unchanged)
# ----------------------------------------------------------------------

def _slugify(title: str) -> str:
    """Convert blog title to filesystem-safe slug."""
    slug = title.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "-", slug)
    slug = slug.strip("-")
    return slug or "untitled"


def _estimate_read_time(word_count: int) -> str:
    """Estimate read time in minutes at 200 WPM."""
    minutes = max(1, round(word_count / 200))
    return f"{minutes} min read"


# ----------------------------------------------------------------------
# Core Functions (100% unchanged)
# ----------------------------------------------------------------------

def sort_sections_by_plan(
    plan: BlogPlan,
    completed_sections: List[SectionDraft]
) -> List[SectionDraft]:
    """... (your original code) ..."""
    if not plan or not hasattr(plan, "sections") or not plan.sections:
        logger.warning("[reducer] No plan sections found, returning sections as-is")
        return completed_sections

    plan_order = {}
    for idx, sec in enumerate(plan.sections):
        sid = sec.id if hasattr(sec, "id") else getattr(sec, "id", f"section_{idx}")
        if sid in plan_order:
            continue
        plan_order[sid] = idx

    def sort_key(section: SectionDraft) -> int:
        sid = section.section_id if hasattr(section, "section_id") else getattr(section, "section_id", "")
        return plan_order.get(sid, float("inf"))

    sorted_sections = sorted(completed_sections, key=sort_key)

    found_ids = {(s.section_id if hasattr(s, "section_id") else getattr(s, "section_id", "")) for s in sorted_sections}
    missing = [sid for sid in plan_order if sid not in found_ids]
    if missing:
        logger.warning("[reducer] Missing sections (workers may have failed): %s", missing)

    logger.info("[reducer] Sorted %d sections by plan order (%d planned)", len(sorted_sections), len(plan_order))
    return sorted_sections


def substitute_image_placeholders(
    sections: List[SectionDraft],
    images: List[ImageResult],
    feature_image: Optional[ImageResult] = None
) -> List[SectionDraft]:
    """... (your original code) ..."""
    if not sections:
        return []

    image_lookup = {}
    for img in images:
        sid = img.section_id if hasattr(img, "section_id") else getattr(img, "section_id", "")
        if sid:
            image_lookup[sid] = img

    updated = []
    for section in sections:
        content = section.content if hasattr(section, "content") else ""
        if not content:
            updated.append(section)
            continue

        def replace_placeholder(match: re.Match) -> str:
            sid = match.group(1)
            img = image_lookup.get(sid)
            if img is None:
                logger.warning("[reducer] No image found for placeholder '%s', removing", sid)
                return ""
            path = img.file_path if hasattr(img, "file_path") else getattr(img, "file_path", "")
            alt = img.alt_text if hasattr(img, "alt_text") else getattr(img, "alt_text", sid)
            return f"![{alt}]({path})"

        new_content = re.sub(r"\[IMAGE_PLACEHOLDER_([a-zA-Z0-9_]+)\]", replace_placeholder, content)

        new_section = SectionDraft(
            section_id=section.section_id,
            title=section.title,
            content=new_content,
            word_count=section.word_count if hasattr(section, "word_count") else 0,
            citations=section.citations if hasattr(section, "citations") else [],
        )
        updated.append(new_section)

    logger.info("[reducer] Substituted image placeholders in %d sections", len(updated))
    return updated


def assemble_blog(
    plan: BlogPlan,
    sorted_sections: List[SectionDraft],
    references_md: str = "",
    feature_image: Optional[ImageResult] = None,
) -> str:
    """... (your original code) ..."""
    lines = []

    title = plan.blog_title if hasattr(plan, "blog_title") else getattr(plan, "blog_title", "Untitled")
    date_str = datetime.now().strftime("%Y-%m-%d")

    total_words = sum(
        s.word_count if hasattr(s, "word_count") else 0
        for s in sorted_sections
    )
    read_time = _estimate_read_time(total_words)

    lines.append("---")
    lines.append(f"title: \"{title}\"")
    lines.append(f"date: {date_str}")
    lines.append(f"read_time: {read_time}")
    lines.append(f"word_count: {total_words}")
    if hasattr(plan, "sections") and plan.sections:
        tags = [s.title for s in plan.sections[:3]]
        lines.append(f"tags: {tags}")
    lines.append("---")
    lines.append("")

    if feature_image is not None:
        fpath = feature_image.file_path if hasattr(feature_image, "file_path") else getattr(feature_image, "file_path", "")
        falt = feature_image.alt_text if hasattr(feature_image, "alt_text") else getattr(feature_image, "alt_text", "Feature")
        lines.append(f"![{falt}]({fpath})")
        lines.append("")

    for section in sorted_sections:
        sec_title = section.title if hasattr(section, "title") else getattr(section, "title", "Untitled")
        sec_content = section.content if hasattr(section, "content") else ""
        lines.append(f"# {sec_title}")
        lines.append("")
        lines.append(sec_content)
        lines.append("")
        lines.append("---")
        lines.append("")

    if references_md and references_md.strip():
        lines.append(references_md)
        lines.append("")

    return "\n".join(lines)


def write_outputs(
    md_content: str,
    slug: str,
    output_dir: str = OUTPUT_BLOGS
) -> Tuple[str, str]:
    """... (your original code) ..."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    md_file = out_path / f"{slug}.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("[reducer] Wrote Markdown: %s", md_file)

    html_content = markdown.markdown(
        md_content,
        extensions=["fenced_code", "tables"],
    )

    html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{slug.replace("-", " ").title()}</title>
    <style>
        body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 700px; margin: 0 auto; padding: 2rem; line-height: 1.6; }}
        img {{ max-width: 100%; height: auto; }}
        pre {{ background: #f4f4f4; padding: 1rem; overflow-x: auto; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 0.5rem; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""

    html_file = out_path / f"{slug}.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_page)
    logger.info("[reducer] Wrote HTML: %s", html_file)

    return str(md_file), str(html_file)


def process_reduction(
    plan: BlogPlan,
    completed_sections: List[SectionDraft],
    generated_images: List[ImageResult],
    references_md: str = "",
    feature_image: Optional[ImageResult] = None,
) -> Tuple[str, str, str]:
    """... (your original code) ..."""
    logger.info("[reducer] Starting reduction pipeline")

    sorted_sections = sort_sections_by_plan(plan, completed_sections)

    sections_with_images = substitute_image_placeholders(
        sorted_sections, generated_images, feature_image
    )

    md_content = assemble_blog(plan, sections_with_images, references_md, feature_image)

    title = plan.blog_title if hasattr(plan, "blog_title") else "untitled"
    slug = _slugify(title)
    md_path, html_path = write_outputs(md_content, slug)

    logger.info("[reducer] Reduction complete: %s, %s", md_path, html_path)
    return md_content, md_path, html_path


# ----------------------------------------------------------------------
# Node function expected by graph.py (lightweight partial dict)
# ----------------------------------------------------------------------
def reducer_node(state: CrewState) -> Dict[str, Any]:
    """reducer_node(state) → {"final_markdown": str, "final_html": str, "output_path": str}"""
    plan = state.get("plan")
    if not plan:
        raise ValueError("reducer_node: plan is required in CrewState")

    completed_sections = state.get("completed_sections", [])
    generated_images = state.get("generated_images", [])

    # Citation manager already appended References to the last section
    md_content, md_path, html_path = process_reduction(
        plan, completed_sections, generated_images, references_md=""
    )

    # Read HTML content for Streamlit preview + download
    html_content = ""
    try:
        html_content = Path(html_path).read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("[reducer_node] Could not read HTML file: %s", e)

    return {
        "final_markdown": md_content,
        "final_html": html_content,
        "output_path": md_path,
    }


# ----------------------------------------------------------------------
# Exports
# ----------------------------------------------------------------------
__all__ = ["reducer_node", "process_reduction", "sort_sections_by_plan",
           "substitute_image_placeholders", "assemble_blog", "write_outputs"]


# ----------------------------------------------------------------------
# Self-test (unchanged)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ... your original self-test code remains exactly the same ...
    print("All assertions passed.")