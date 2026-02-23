# import logging
# import re
# from typing import List, Tuple

# from graph.state import AgentState, SEOIssue, Severity

# logger = logging.getLogger(__name__)

# # Weighted checks — weights sum to 100
# _CHECKS = [
#     ("title_tag_present",       15),
#     ("title_tag_length",         5),
#     ("title_keyword",           10),
#     ("meta_desc_length",         5),
#     ("keyword_first_100_words",  15),
#     ("single_h1",               10),
#     ("h2_count",                 5),
#     ("internal_links",          15),
#     ("alt_text_slots",          10),
#     ("keyword_density",         10),
# ]
# assert sum(w for _, w in _CHECKS) == 100


# def _count_words(text: str) -> int:
#     return len(re.findall(r"\b\w+\b", text))


# def _first_n_words(text: str, n: int) -> str:
#     words = re.findall(r"\b\w+\b", text)
#     return " ".join(words[:n])


# def _keyword_density(text: str, keyword: str) -> float:
#     if not keyword:
#         return 0.0
#     total_words = _count_words(text)
#     if total_words == 0:
#         return 0.0
#     occurrences = len(re.findall(re.escape(keyword.lower()), text.lower()))
#     return (occurrences / total_words) * 100


# def _extract_h1s(text: str) -> List[str]:
#     return re.findall(r"^#{1}\s+(.+)$", text, re.MULTILINE)


# def _extract_h2s(text: str) -> List[str]:
#     return re.findall(r"^#{2}\s+(.+)$", text, re.MULTILINE)


# def _count_internal_link_placeholders(text: str) -> int:
#     return len(re.findall(r"\[INTERNAL LINK:", text))


# def _count_alt_text_slots(text: str) -> int:
#     return len(re.findall(r"\[IMAGE:", text))


# def _run_checks(
#     draft: str,
#     primary_kw: str,
#     title_tag: str,
#     meta_desc: str,
# ) -> Tuple[int, List[SEOIssue], List[str]]:
#     issues: List[SEOIssue] = []
#     suggestions: List[str] = []
#     score = 0

#     # 1. Title tag present (15 pts)
#     if title_tag and len(title_tag.strip()) > 0:
#         score += 15
#     else:
#         issues.append(SEOIssue(
#             field="title_tag",
#             severity=Severity.CRITICAL,
#             message="Title tag is missing or empty.",
#             suggestion="Add a title tag containing the primary keyword.",
#         ))
#         suggestions.append("Add a descriptive title tag (50-60 chars) containing the primary keyword.")

#     # 2. Title tag length 50-60 chars (5 pts)
#     if title_tag:
#         tlen = len(title_tag.strip())
#         if 50 <= tlen <= 60:
#             score += 5
#         else:
#             issues.append(SEOIssue(
#                 field="title_tag",
#                 severity=Severity.WARNING,
#                 message=f"Title tag is {tlen} chars (target: 50-60).",
#                 suggestion=f"{'Shorten' if tlen > 60 else 'Lengthen'} the title to 50-60 characters.",
#             ))
#             suggestions.append(f"Title tag is {tlen} chars — adjust to 50-60 characters.")

#     # 3. Primary keyword in title (10 pts)
#     if title_tag and primary_kw and primary_kw.lower() in title_tag.lower():
#         score += 10
#     elif primary_kw:
#         issues.append(SEOIssue(
#             field="title_tag",
#             severity=Severity.CRITICAL,
#             message=f"Primary keyword '{primary_kw}' not found in title tag.",
#             suggestion=f"Include '{primary_kw}' in the title tag, preferably near the beginning.",
#         ))
#         suggestions.append(f"Include primary keyword '{primary_kw}' in the title tag.")

#     # 4. Meta description 150-160 chars (5 pts)
#     if meta_desc:
#         mlen = len(meta_desc.strip())
#         if 150 <= mlen <= 160:
#             score += 5
#         else:
#             issues.append(SEOIssue(
#                 field="meta_description",
#                 severity=Severity.WARNING,
#                 message=f"Meta description is {mlen} chars (target: 150-160).",
#                 suggestion=f"{'Trim' if mlen > 160 else 'Expand'} meta description to 150-160 characters.",
#             ))
#             suggestions.append(f"Meta description is {mlen} chars — adjust to 150-160 characters.")
#     else:
#         issues.append(SEOIssue(
#             field="meta_description",
#             severity=Severity.WARNING,
#             message="Meta description is missing.",
#             suggestion="Add a meta description of 150-160 characters including the primary keyword.",
#         ))
#         suggestions.append("Write a meta description of 150-160 characters with the primary keyword.")

#     # 5. Primary keyword in first 100 words (15 pts)
#     first_100 = _first_n_words(draft, 100)
#     if primary_kw and primary_kw.lower() in first_100.lower():
#         score += 15
#     else:
#         issues.append(SEOIssue(
#             field="keyword_placement",
#             severity=Severity.CRITICAL,
#             message=f"Primary keyword '{primary_kw}' not found in the first 100 words.",
#             suggestion="Include the primary keyword naturally within the first paragraph.",
#         ))
#         suggestions.append(f"Place primary keyword '{primary_kw}' within the first 100 words of the article.")

#     # 6. Single H1 (10 pts)
#     h1s = _extract_h1s(draft)
#     if len(h1s) == 1:
#         score += 10
#     elif len(h1s) == 0:
#         issues.append(SEOIssue(
#             field="heading_hierarchy",
#             severity=Severity.CRITICAL,
#             message="No H1 heading found in the article.",
#             suggestion="Add exactly one H1 heading containing the primary keyword.",
#         ))
#         suggestions.append("Add exactly one H1 heading (# Title) containing the primary keyword.")
#     else:
#         issues.append(SEOIssue(
#             field="heading_hierarchy",
#             severity=Severity.WARNING,
#             message=f"Found {len(h1s)} H1 headings — only one is allowed.",
#             suggestion="Consolidate to a single H1 heading; convert extras to H2.",
#         ))
#         suggestions.append(f"Reduce to one H1 heading — found {len(h1s)}.")

#     # 7. At least 3 H2 headings (5 pts)
#     h2s = _extract_h2s(draft)
#     if len(h2s) >= 3:
#         score += 5
#     else:
#         issues.append(SEOIssue(
#             field="heading_hierarchy",
#             severity=Severity.WARNING,
#             message=f"Found only {len(h2s)} H2 headings (minimum 3 required).",
#             suggestion="Add more H2 section headings to improve content structure.",
#         ))
#         suggestions.append(f"Add more H2 headings — currently {len(h2s)}, need at least 3.")

#     # 8. Internal link placeholders ≥ 3 (15 pts)
#     internal_count = _count_internal_link_placeholders(draft)
#     if internal_count >= 3:
#         score += 15
#     else:
#         issues.append(SEOIssue(
#             field="internal_linking",
#             severity=Severity.CRITICAL,
#             message=f"Only {internal_count} internal link placeholder(s) found (minimum 3).",
#             suggestion="Add [INTERNAL LINK: anchor text → target] placeholders at natural points in the article.",
#         ))
#         suggestions.append(f"Add at least {3 - internal_count} more [INTERNAL LINK: ...] placeholders.")

#     # 9. Image alt text slots present (10 pts)
#     alt_count = _count_alt_text_slots(draft)
#     if alt_count >= 1:
#         score += 10
#     else:
#         issues.append(SEOIssue(
#             field="image_alt_text",
#             severity=Severity.WARNING,
#             message="No image alt text placeholders [IMAGE: ...] found.",
#             suggestion="Add [IMAGE: descriptive alt text] placeholders where images should appear.",
#         ))
#         suggestions.append("Add at least one [IMAGE: ...] placeholder with descriptive alt text.")

#     # 10. Keyword density 1-2% (10 pts)
#     density = _keyword_density(draft, primary_kw)
#     if 1.0 <= density <= 2.0:
#         score += 10
#     elif density < 1.0:
#         issues.append(SEOIssue(
#             field="keyword_density",
#             severity=Severity.WARNING,
#             message=f"Keyword density is {density:.2f}% (target: 1-2%).",
#             suggestion=f"Use '{primary_kw}' more naturally throughout the article.",
#         ))
#         suggestions.append(f"Increase keyword density for '{primary_kw}' — currently {density:.2f}%, target 1-2%.")
#     else:
#         issues.append(SEOIssue(
#             field="keyword_density",
#             severity=Severity.WARNING,
#             message=f"Keyword density is {density:.2f}% — possible over-optimisation.",
#             suggestion=f"Reduce occurrences of '{primary_kw}' and use LSI synonyms instead.",
#         ))
#         suggestions.append(f"Reduce keyword density for '{primary_kw}' — currently {density:.2f}%, target 1-2%.")

#     return score, issues, suggestions


# async def seo_auditor_node(state: AgentState) -> dict:
#     try:
#         draft: str = state.get("draft_article") or ""
#         if not draft:
#             return {
#                 "seo_score":       0,
#                 "seo_issues":      [SEOIssue(field="draft", severity=Severity.CRITICAL,
#                                              message="No draft article found.", suggestion="Run the writer node first.")],
#                 "seo_suggestions": ["Draft article is missing — run the writer node."],
#             }

#         keyword_map = state.get("keyword_map") or {}
#         primary_kw: str = keyword_map.get("primary", "")
#         title_tag: str  = state.get("title_tag") or ""
#         meta_desc: str  = state.get("meta_description") or ""

#         # Generate title_tag and meta_desc from draft if not yet produced by final_assembler
#         if not title_tag:
#             h1s = _extract_h1s(draft)
#             title_tag = h1s[0][:60] if h1s else draft.split("\n")[0][:60]
#         if not meta_desc:
#             plain = re.sub(r"[#\*\[\]`>_]", "", draft)
#             sentences = re.split(r"(?<=[.!?])\s+", plain.strip())
#             meta_desc = " ".join(sentences[:2])[:160]

#         score, issues, suggestions = _run_checks(draft, primary_kw, title_tag, meta_desc)

#         logger.info(
#             "SEO auditor: score=%d issues=%d (iteration %d)",
#             score, len(issues), state.get("revision_iteration", 0),
#         )

#         return {
#             "seo_score":       score,
#             "seo_issues":      issues,
#             "seo_suggestions": suggestions,
#             "title_tag":       title_tag,
#             "meta_description": meta_desc,
#             "error":           None,
#             "error_node":      None,
#         }

#     except Exception as exc:
#         logger.exception("SEO auditor failed.")
#         return {"error": str(exc), "error_node": "seo_auditor"}






















# @######################################################################























import logging
import re
from typing import List, Tuple

from graph.state import AgentState, SEOIssue, Severity

logger = logging.getLogger(__name__)

# Weighted checks — weights sum to 100
_CHECKS = [
    ("title_tag_present",       15),
    ("title_tag_length",         5),
    ("title_keyword",           10),
    ("meta_desc_length",         5),
    ("keyword_first_100_words",  15),
    ("single_h1",               10),
    ("h2_count",                 5),
    ("internal_links",          15),
    ("alt_text_slots",          10),
    ("keyword_density",         10),
]
assert sum(w for _, w in _CHECKS) == 100


def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _first_n_words(text: str, n: int) -> str:
    words = re.findall(r"\b\w+\b", text)
    return " ".join(words[:n])


def _keyword_density(text: str, keyword: str) -> float:
    if not keyword:
        return 0.0
    total_words = _count_words(text)
    if total_words == 0:
        return 0.0
    occurrences = len(re.findall(re.escape(keyword.lower()), text.lower()))
    return (occurrences / total_words) * 100


def _extract_h1s(text: str) -> List[str]:
    return re.findall(r"^#{1}\s+(.+)$", text, re.MULTILINE)


def _extract_h2s(text: str) -> List[str]:
    return re.findall(r"^#{2}\s+(.+)$", text, re.MULTILINE)


def _count_internal_link_placeholders(text: str) -> int:
    return len(re.findall(r"\[INTERNAL LINK:", text))


def _count_alt_text_slots(text: str) -> int:
    return len(re.findall(r"\[IMAGE:", text))


def _run_checks(
    draft: str,
    primary_kw: str,
    title_tag: str,
    meta_desc: str,
) -> Tuple[int, List[SEOIssue], List[str]]:
    issues: List[SEOIssue] = []
    suggestions: List[str] = []
    score = 0

    # 1. Title tag present (15 pts)
    if title_tag and len(title_tag.strip()) > 0:
        score += 15
    else:
        issues.append(SEOIssue(
            field="title_tag",
            severity=Severity.CRITICAL,
            message="Title tag is missing or empty.",
            suggestion="Add a title tag containing the primary keyword.",
        ))
        suggestions.append("Add a descriptive title tag (50-60 chars) containing the primary keyword.")

    # 2. Title tag length 50-60 chars (5 pts)
    if title_tag:
        tlen = len(title_tag.strip())
        if 50 <= tlen <= 60:
            score += 5
        else:
            issues.append(SEOIssue(
                field="title_tag",
                severity=Severity.WARNING,
                message=f"Title tag is {tlen} chars (target: 50-60).",
                suggestion=f"{'Shorten' if tlen > 60 else 'Lengthen'} the title to 50-60 characters.",
            ))
            suggestions.append(f"Title tag is {tlen} chars — adjust to 50-60 characters.")

    # 3. Primary keyword in title (10 pts)
    if title_tag and primary_kw and primary_kw.lower() in title_tag.lower():
        score += 10
    elif primary_kw:
        issues.append(SEOIssue(
            field="title_tag",
            severity=Severity.CRITICAL,
            message=f"Primary keyword '{primary_kw}' not found in title tag.",
            suggestion=f"Include '{primary_kw}' in the title tag, preferably near the beginning.",
        ))
        suggestions.append(f"Include primary keyword '{primary_kw}' in the title tag.")

    # 4. Meta description 150-160 chars (5 pts)
    if meta_desc:
        mlen = len(meta_desc.strip())
        if 150 <= mlen <= 160:
            score += 5
        else:
            issues.append(SEOIssue(
                field="meta_description",
                severity=Severity.WARNING,
                message=f"Meta description is {mlen} chars (target: 150-160).",
                suggestion=f"{'Trim' if mlen > 160 else 'Expand'} meta description to 150-160 characters.",
            ))
            suggestions.append(f"Meta description is {mlen} chars — adjust to 150-160 characters.")
    else:
        issues.append(SEOIssue(
            field="meta_description",
            severity=Severity.WARNING,
            message="Meta description is missing.",
            suggestion="Add a meta description of 150-160 characters including the primary keyword.",
        ))
        suggestions.append("Write a meta description of 150-160 characters with the primary keyword.")

    # 5. Primary keyword in first 100 words (15 pts)
    first_100 = _first_n_words(draft, 100)
    if primary_kw and primary_kw.lower() in first_100.lower():
        score += 15
    else:
        issues.append(SEOIssue(
            field="keyword_placement",
            severity=Severity.CRITICAL,
            message=f"Primary keyword '{primary_kw}' not found in the first 100 words.",
            suggestion="Include the primary keyword naturally within the first paragraph.",
        ))
        suggestions.append(f"Place primary keyword '{primary_kw}' within the first 100 words of the article.")

    # 6. Single H1 (10 pts)
    h1s = _extract_h1s(draft)
    if len(h1s) == 1:
        score += 10
    elif len(h1s) == 0:
        issues.append(SEOIssue(
            field="heading_hierarchy",
            severity=Severity.CRITICAL,
            message="No H1 heading found in the article.",
            suggestion="Add exactly one H1 heading containing the primary keyword.",
        ))
        suggestions.append("Add exactly one H1 heading (# Title) containing the primary keyword.")
    else:
        issues.append(SEOIssue(
            field="heading_hierarchy",
            severity=Severity.WARNING,
            message=f"Found {len(h1s)} H1 headings — only one is allowed.",
            suggestion="Consolidate to a single H1 heading; convert extras to H2.",
        ))
        suggestions.append(f"Reduce to one H1 heading — found {len(h1s)}.")

    # 7. At least 3 H2 headings (5 pts)
    h2s = _extract_h2s(draft)
    if len(h2s) >= 3:
        score += 5
    else:
        issues.append(SEOIssue(
            field="heading_hierarchy",
            severity=Severity.WARNING,
            message=f"Found only {len(h2s)} H2 headings (minimum 3 required).",
            suggestion="Add more H2 section headings to improve content structure.",
        ))
        suggestions.append(f"Add more H2 headings — currently {len(h2s)}, need at least 3.")

    # 8. Internal link placeholders ≥ 3 (15 pts)
    internal_count = _count_internal_link_placeholders(draft)
    if internal_count >= 3:
        score += 15
    else:
        issues.append(SEOIssue(
            field="internal_linking",
            severity=Severity.CRITICAL,
            message=f"Only {internal_count} internal link placeholder(s) found (minimum 3).",
            suggestion="Add [INTERNAL LINK: anchor text → target] placeholders at natural points in the article.",
        ))
        suggestions.append(f"Add at least {3 - internal_count} more [INTERNAL LINK: ...] placeholders.")

    # 9. Image alt text slots present (10 pts)
    alt_count = _count_alt_text_slots(draft)
    if alt_count >= 1:
        score += 10
    else:
        issues.append(SEOIssue(
            field="image_alt_text",
            severity=Severity.WARNING,
            message="No image alt text placeholders [IMAGE: ...] found.",
            suggestion="Add [IMAGE: descriptive alt text] placeholders where images should appear.",
        ))
        suggestions.append("Add at least one [IMAGE: ...] placeholder with descriptive alt text.")

    # 10. Keyword density 1-2% (10 pts)
    density = _keyword_density(draft, primary_kw)
    if 1.0 <= density <= 2.0:
        score += 10
    elif density < 1.0:
        issues.append(SEOIssue(
            field="keyword_density",
            severity=Severity.WARNING,
            message=f"Keyword density is {density:.2f}% (target: 1-2%).",
            suggestion=f"Use '{primary_kw}' more naturally throughout the article.",
        ))
        suggestions.append(f"Increase keyword density for '{primary_kw}' — currently {density:.2f}%, target 1-2%.")
    else:
        issues.append(SEOIssue(
            field="keyword_density",
            severity=Severity.WARNING,
            message=f"Keyword density is {density:.2f}% — possible over-optimisation.",
            suggestion=f"Reduce occurrences of '{primary_kw}' and use LSI synonyms instead.",
        ))
        suggestions.append(f"Reduce keyword density for '{primary_kw}' — currently {density:.2f}%, target 1-2%.")

    return score, issues, suggestions


async def seo_auditor_node(state: AgentState) -> dict:
    try:
        draft: str = state.get("draft_article") or ""
        if not draft:
            return {
                "seo_score":       0,
                "seo_issues":      [SEOIssue(field="draft", severity=Severity.CRITICAL,
                                             message="No draft article found.", suggestion="Run the writer node first.")],
                "seo_suggestions": ["Draft article is missing — run the writer node."],
            }

        keyword_map = state.get("keyword_map") or {}
        primary_kw: str = keyword_map.get("primary", "")
        title_tag: str  = state.get("title_tag") or ""
        meta_desc: str  = state.get("meta_description") or ""

        # Generate title_tag and meta_desc from draft if not yet produced by final_assembler
        if not title_tag:
            h1s = _extract_h1s(draft)
            title_tag = h1s[0][:60] if h1s else draft.split("\n")[0][:60]
        if not meta_desc:
            plain = re.sub(r"[#\*\[\]`>_]", "", draft)
            sentences = re.split(r"(?<=[.!?])\s+", plain.strip())
            meta_desc = " ".join(sentences[:2])[:160]

        score, issues, suggestions = _run_checks(draft, primary_kw, title_tag, meta_desc)

        logger.info(
            "SEO auditor: score=%d issues=%d (iteration %d)",
            score, len(issues), state.get("revision_iteration", 0),
        )

        return {
            "seo_score":       score,
            "seo_issues":      issues,
            "seo_suggestions": suggestions,
            "title_tag":       title_tag,
            "meta_description": meta_desc,
            "error":           None,
            "error_node":      None,
        }

    except Exception as exc:
        logger.exception("SEO auditor failed.")
        return {"error": str(exc), "error_node": "seo_auditor"}