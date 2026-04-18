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
from typing import Dict, List, Optional, Tuple

import markdown

from config import OUTPUT_BLOGS, OUTPUT_IMAGES
from schemas import BlogPlan, ImageResult, SectionDraft

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
# Helpers
# ---------------------------------------------------------------------------

def _slugify(title: str) -> str:
    """Convert blog title to filesystem-safe slug."""
    # Lowercase, replace spaces/special chars with hyphens, collapse multiple hyphens
    slug = title.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "-", slug)
    slug = slug.strip("-")
    return slug or "untitled"


def _estimate_read_time(word_count: int) -> str:
    """Estimate read time in minutes at 200 WPM."""
    minutes = max(1, round(word_count / 200))
    return f"{minutes} min read"


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def sort_sections_by_plan(
    plan: BlogPlan,
    completed_sections: List[SectionDraft]
) -> List[SectionDraft]:
    """
    Sort completed sections to match the order defined in plan.sections.

    CrewAI parallel workers do NOT guarantee output ordering.
    We build an index map from plan.section_ids and sort accordingly.
    """
    if not plan or not hasattr(plan, "sections") or not plan.sections:
        logger.warning("[reducer] No plan sections found, returning sections as-is")
        return completed_sections

    # Build {section_id: index} from plan order
    plan_order = {}
    for idx, sec in enumerate(plan.sections):
        sid = sec.id if hasattr(sec, "id") else sec.get("id", f"section_{idx}")
        if sid in plan_order:
            logger.warning("[reducer] Duplicate section_id '%s' in plan, using first occurrence", sid)
            continue
        plan_order[sid] = idx

    def sort_key(section: SectionDraft) -> int:
        sid = section.section_id if hasattr(section, "section_id") else section.get("section_id", "")
        return plan_order.get(sid, float("inf"))

    sorted_sections = sorted(completed_sections, key=sort_key)

    # Log any missing sections
    found_ids = {
        (s.section_id if hasattr(s, "section_id") else s.get("section_id", ""))
        for s in sorted_sections
    }
    missing = [sid for sid in plan_order if sid not in found_ids]
    if missing:
        logger.warning("[reducer] Missing sections (workers may have failed): %s", missing)

    logger.info(
        "[reducer] Sorted %d sections by plan order (%d planned)",
        len(sorted_sections),
        len(plan_order)
    )
    return sorted_sections


def substitute_image_placeholders(
    sections: List[SectionDraft],
    images: List[ImageResult],
    feature_image: Optional[ImageResult] = None
) -> List[SectionDraft]:
    """
    Replace [IMAGE_PLACEHOLDER_{section_id}] tokens with Markdown image syntax.

    Builds lookup from images list. Missing images become empty strings.
    """
    if not sections:
        return []

    # Build lookup: {section_id: ImageResult}
    image_lookup = {}
    for img in images:
        sid = img.section_id if hasattr(img, "section_id") else img.get("section_id", "")
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
            # Use absolute or relative path from ImageResult
            path = img.file_path if hasattr(img, "file_path") else img.get("file_path", "")
            alt = img.alt_text if hasattr(img, "alt_text") else img.get("alt_text", sid)
            return f"![{alt}]({path})"

        # Match [IMAGE_PLACEHOLDER_alphanumeric_underscore]
        new_content = re.sub(r"\[IMAGE_PLACEHOLDER_([a-zA-Z0-9_]+)\]", replace_placeholder, content)

        # Create new SectionDraft (immutable pattern for CrewAI safety)
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
    """
    Assemble final Markdown string from plan, sections, and references.

    Structure:
      ---
      title: ...
      date: ...
      read_time: ...
      tags: ...
      ---
      ![Feature](path)

      # Section 1
      ...
      ---
      # Section 2
      ...

      ## References
      ...
    """
    lines = []

    # YAML frontmatter
    title = plan.blog_title if hasattr(plan, "blog_title") else plan.get("blog_title", "Untitled")
    date_str = datetime.now().strftime("%Y-%m-%d")

    # Calculate total word count
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

    # Feature image
    if feature_image is not None:
        fpath = feature_image.file_path if hasattr(feature_image, "file_path") else feature_image.get("file_path", "")
        falt = feature_image.alt_text if hasattr(feature_image, "alt_text") else feature_image.get("alt_text", "Feature")
        lines.append(f"![{falt}]({fpath})")
        lines.append("")

    # Sections with horizontal rules between them
    for section in sorted_sections:
        sec_title = section.title if hasattr(section, "title") else section.get("title", "Untitled")
        sec_content = section.content if hasattr(section, "content") else ""
        lines.append(f"# {sec_title}")
        lines.append("")
        lines.append(sec_content)
        lines.append("")
        lines.append("---")
        lines.append("")

    # References (if provided)
    if references_md and references_md.strip():
        lines.append(references_md)
        lines.append("")

    return "\n".join(lines)


def write_outputs(
    md_content: str,
    slug: str,
    output_dir: str = OUTPUT_BLOGS
) -> Tuple[str, str]:
    """
    Write Markdown and HTML files to disk.

    Returns:
        Tuple of (md_path, html_path) as strings.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Write Markdown
    md_file = out_path / f"{slug}.md"
    try:
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info("[reducer] Wrote Markdown: %s", md_file)
    except OSError as exc:
        raise RuntimeError(f"Failed to write Markdown file {md_file}: {exc}") from exc

    # Convert to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=["fenced_code", "tables"],
    )

    # Wrap in minimal HTML5 boilerplate
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
    try:
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_page)
        logger.info("[reducer] Wrote HTML: %s", html_file)
    except OSError as exc:
        raise RuntimeError(f"Failed to write HTML file {html_file}: {exc}") from exc

    return str(md_file), str(html_file)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def process_reduction(
    plan: BlogPlan,
    completed_sections: List[SectionDraft],
    generated_images: List[ImageResult],
    references_md: str = "",
    feature_image: Optional[ImageResult] = None,
) -> Tuple[str, str, str]:
    """
    Orchestrate full reduction pipeline.

    Returns:
        Tuple of (markdown_content, md_path, html_path)
    """
    logger.info("[reducer] Starting reduction pipeline")

    # 1. Sort by plan order
    sorted_sections = sort_sections_by_plan(plan, completed_sections)

    # 2. Substitute image placeholders
    sections_with_images = substitute_image_placeholders(
        sorted_sections, generated_images, feature_image
    )

    # 3. Assemble blog
    md_content = assemble_blog(plan, sections_with_images, references_md, feature_image)

    # 4. Write outputs
    title = plan.blog_title if hasattr(plan, "blog_title") else "untitled"
    slug = _slugify(title)
    md_path, html_path = write_outputs(md_content, slug)

    logger.info("[reducer] Reduction complete: %s, %s", md_path, html_path)
    return md_content, md_path, html_path


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import datetime

    # Mock classes for standalone testing
    class MockSection:
        def __init__(self, sid, title, content, word_count=100):
            self.id = sid
            self.title = title
            self.description = "Test"
            self.word_count = word_count
            self.search_query = ""
            self.image_prompt = ""

    class MockBlogPlan:
        def __init__(self):
            self.blog_title = "Test Blog Post"
            self.feature_image_prompt = "A test image"
            self.sections = [
                MockSection("intro", "Introduction", "Intro content"),
                MockSection("body", "Main Body", "Body content"),
                MockSection("conclusion", "Conclusion", "Conclusion content"),
            ]
            self.research_required = True

    class MockSectionDraft:
        def __init__(self, sid, title, content, word_count=100):
            self.section_id = sid
            self.title = title
            self.content = content
            self.word_count = word_count
            self.citations = []

    class MockImageResult:
        def __init__(self, sid, prompt, path, alt=""):
            self.section_id = sid
            self.prompt = prompt
            self.file_path = path
            self.alt_text = alt or f"Image for {sid}"
            self.size = (512, 512)

    # Create test data — sections OUT OF ORDER to test sorting
    plan = MockBlogPlan()
    sections = [
        MockSectionDraft("conclusion", "Conclusion", "This is the conclusion [IMAGE_PLACEHOLDER_conclusion]."),
        MockSectionDraft("intro", "Introduction", "This is the intro [IMAGE_PLACEHOLDER_intro]."),
        MockSectionDraft("body", "Main Body", "This is the body [IMAGE_PLACEHOLDER_body]."),
    ]
    images = [
        MockImageResult("intro", "intro image", "outputs/images/intro.png"),
        MockImageResult("body", "body image", "outputs/images/body.png"),
        MockImageResult("conclusion", "conclusion image", "outputs/images/conclusion.png"),
    ]
    feature = MockImageResult("feature", "feature image", "outputs/images/feature.png", "Feature Image")
    refs = "## References\n\n1. [Example](https://example.com) — An example source.\n"

    # Run pipeline
    md_content, md_path, html_path = process_reduction(
        plan, sections, images, refs, feature
    )

    print("=== MARKDOWN OUTPUT (first 800 chars) ===")
    print(md_content[:800])
    print("\n...\n")

    print("=== FILES WRITTEN ===")
    print(f"Markdown: {md_path}")
    print(f"HTML:     {html_path}")

    # Assertions
    assert "Test Blog Post" in md_content, "Title missing from output"
    assert "## References" in md_content, "References missing"
    assert "![Feature Image]" in md_content, "Feature image missing"
    assert "![Image for intro]" in md_content, "Intro image placeholder not substituted"
    assert "![Image for body]" in md_content, "Body image placeholder not substituted"
    assert "![Image for conclusion]" in md_content, "Conclusion image placeholder not substituted"

    # Verify order: intro should come before body before conclusion
    intro_pos = md_content.find("# Introduction")
    body_pos = md_content.find("# Main Body")
    conclusion_pos = md_content.find("# Conclusion")
    assert intro_pos < body_pos < conclusion_pos, f"Sections out of order: {intro_pos}, {body_pos}, {conclusion_pos}"

    assert Path(md_path).exists(), f"Markdown file not found: {md_path}"
    assert Path(html_path).exists(), f"HTML file not found: {html_path}"

    print("\nAll assertions passed.")
