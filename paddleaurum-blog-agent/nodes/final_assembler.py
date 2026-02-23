# import logging
# import re
# from typing import List

# from graph.state import AgentState, FinalOutput, ImageSlot, SchemaMarkup

# logger = logging.getLogger(__name__)

# _SITE_NAME = "PaddleAurum"
# _MAX_TITLE_LEN = 60
# _MAX_META_LEN  = 160


# def _generate_url_slug(title: str, primary_kw: str) -> str:
#     base = primary_kw if primary_kw else title
#     slug = base.lower().strip()
#     slug = re.sub(r"[^\w\s-]", "", slug)
#     slug = re.sub(r"[\s_]+", "-", slug)
#     slug = re.sub(r"-+", "-", slug).strip("-")
#     return slug[:80]


# def _generate_title_tag(h1: str, primary_kw: str) -> str:
#     candidate = f"{h1} | {_SITE_NAME}"
#     if len(candidate) <= _MAX_TITLE_LEN:
#         return candidate
#     # Trim h1 to fit
#     budget = _MAX_TITLE_LEN - len(f" | {_SITE_NAME}")
#     return f"{h1[:budget].rstrip()} | {_SITE_NAME}"


# def _generate_meta_description(draft: str, primary_kw: str) -> str:
#     plain = re.sub(r"\[.*?\]|[#*`>_\-]", " ", draft)
#     plain = re.sub(r"\s+", " ", plain).strip()
#     sentences = re.split(r"(?<=[.!?])\s+", plain)
#     meta = ""
#     for sentence in sentences:
#         candidate = (meta + " " + sentence).strip() if meta else sentence
#         if len(candidate) <= _MAX_META_LEN:
#             meta = candidate
#         else:
#             break
#     if not meta:
#         meta = plain[:_MAX_META_LEN]
#     return meta.strip()


# def _embed_images(article: str, image_manifest: List[ImageSlot]) -> str:
#     for img in image_manifest:
#         alt_text = img.get("alt_text", img.get("description", "Pickleball image"))
#         url      = img.get("url")
#         credit   = img.get("credit", "")

#         if url:
#             replacement = f"![{alt_text}]({url})"
#             if credit:
#                 replacement += f"\n*Photo credit: {credit}*"
#         else:
#             replacement = f"<!-- IMAGE PLACEHOLDER: {alt_text} -->"

#         # Replace the first unresolved [IMAGE: ...] tag
#         article = re.sub(r"\[IMAGE:[^\]]+\]", replacement, article, count=1)

#     return article


# async def final_assembler_node(state: AgentState) -> dict:
#     try:
#         article: str        = state.get("formatted_article") or state.get("draft_article") or ""
#         image_manifest       = state.get("image_manifest") or []
#         schema_markup        = state.get("schema_markup")
#         sources: List[str]  = state.get("research_sources") or []
#         keyword_map          = state.get("keyword_map") or {}
#         primary_kw: str     = keyword_map.get("primary", "")
#         seo_score: int      = state.get("seo_score") or 0

#         # Embed resolved images
#         article_with_images = _embed_images(article, image_manifest)

#         # Derive meta fields
#         h1s = re.findall(r"^#{1}\s+(.+)$", article_with_images, re.MULTILINE)
#         h1_text = h1s[0].strip() if h1s else state.get("topic", "Pickleball Guide")

#         existing_title = state.get("title_tag") or ""
#         title_tag = existing_title if (existing_title and len(existing_title) <= _MAX_TITLE_LEN) \
#                     else _generate_title_tag(h1_text, primary_kw)

#         existing_meta = state.get("meta_description") or ""
#         meta_description = existing_meta if (existing_meta and 150 <= len(existing_meta) <= _MAX_META_LEN) \
#                            else _generate_meta_description(article_with_images, primary_kw)

#         url_slug = state.get("url_slug") or _generate_url_slug(h1_text, primary_kw)

#         # Format citations list
#         citations = [f"[{i+1}] {url}" for i, url in enumerate(sources)]

#         word_count = len(article_with_images.split())

#         if schema_markup is None:
#             schema_markup = SchemaMarkup(article="{}", faq=None, how_to=None)

#         final_output = FinalOutput(
#             markdown=article_with_images,
#             title_tag=title_tag,
#             meta_description=meta_description,
#             url_slug=url_slug,
#             schema_markup=schema_markup,
#             image_manifest=list(image_manifest),
#             citations=citations,
#             word_count=word_count,
#             seo_score=seo_score,
#         )

#         logger.info(
#             "Final assembler: %d words | title='%s' | slug='%s' | images=%d | citations=%d",
#             word_count, title_tag, url_slug, len(image_manifest), len(citations),
#         )

#         return {
#             "final_output":    final_output,
#             "title_tag":       title_tag,
#             "meta_description": meta_description,
#             "url_slug":        url_slug,
#             "error":           None,
#             "error_node":      None,
#         }

#     except Exception as exc:
#         logger.exception("Final assembler failed.")
#         return {"error": str(exc), "error_node": "final_assembler"}
























# @#######################################################################################


















import logging
import re
from typing import List

from graph.state import AgentState, FinalOutput, ImageSlot, SchemaMarkup

logger = logging.getLogger(__name__)

_SITE_NAME = "PaddleAurum"
_MAX_TITLE_LEN = 60
_MAX_META_LEN  = 160


def _generate_url_slug(title: str, primary_kw: str) -> str:
    base = primary_kw if primary_kw else title
    slug = base.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:80]


def _generate_title_tag(h1: str, primary_kw: str) -> str:
    candidate = f"{h1} | {_SITE_NAME}"
    if len(candidate) <= _MAX_TITLE_LEN:
        return candidate
    # Trim h1 to fit
    budget = _MAX_TITLE_LEN - len(f" | {_SITE_NAME}")
    return f"{h1[:budget].rstrip()} | {_SITE_NAME}"


def _generate_meta_description(draft: str, primary_kw: str) -> str:
    plain = re.sub(r"\[.*?\]|[#*`>_\-]", " ", draft)
    plain = re.sub(r"\s+", " ", plain).strip()
    sentences = re.split(r"(?<=[.!?])\s+", plain)
    meta = ""
    for sentence in sentences:
        candidate = (meta + " " + sentence).strip() if meta else sentence
        if len(candidate) <= _MAX_META_LEN:
            meta = candidate
        else:
            break
    if not meta:
        meta = plain[:_MAX_META_LEN]
    return meta.strip()


def _embed_images(article: str, image_manifest: List[ImageSlot]) -> str:
    for img in image_manifest:
        alt_text = img.get("alt_text", img.get("description", "Pickleball image"))
        url      = img.get("url")
        credit   = img.get("credit", "")

        if url:
            replacement = f"![{alt_text}]({url})"
            if credit:
                replacement += f"\n*Photo credit: {credit}*"
        else:
            replacement = f"<!-- IMAGE PLACEHOLDER: {alt_text} -->"

        # Replace the first unresolved [IMAGE: ...] tag
        article = re.sub(r"\[IMAGE:[^\]]+\]", replacement, article, count=1)

    return article


async def final_assembler_node(state: AgentState) -> dict:
    try:
        article: str        = state.get("formatted_article") or state.get("draft_article") or ""
        image_manifest       = state.get("image_manifest") or []
        schema_markup        = state.get("schema_markup")
        sources: List[str]  = state.get("research_sources") or []
        keyword_map          = state.get("keyword_map") or {}
        primary_kw: str     = keyword_map.get("primary", "")
        seo_score: int      = state.get("seo_score") or 0

        # Embed resolved images
        article_with_images = _embed_images(article, image_manifest)

        # Derive meta fields
        h1s = re.findall(r"^#{1}\s+(.+)$", article_with_images, re.MULTILINE)
        h1_text = h1s[0].strip() if h1s else state.get("topic", "Pickleball Guide")

        existing_title = state.get("title_tag") or ""
        title_tag = existing_title if (existing_title and len(existing_title) <= _MAX_TITLE_LEN) \
                    else _generate_title_tag(h1_text, primary_kw)

        existing_meta = state.get("meta_description") or ""
        meta_description = existing_meta if (existing_meta and 150 <= len(existing_meta) <= _MAX_META_LEN) \
                           else _generate_meta_description(article_with_images, primary_kw)

        url_slug = state.get("url_slug") or _generate_url_slug(h1_text, primary_kw)

        # Format citations list
        citations = [f"[{i+1}] {url}" for i, url in enumerate(sources)]

        word_count = len(article_with_images.split())

        if schema_markup is None:
            schema_markup = SchemaMarkup(article="{}", faq=None, how_to=None)

        final_output = FinalOutput(
            markdown=article_with_images,
            title_tag=title_tag,
            meta_description=meta_description,
            url_slug=url_slug,
            schema_markup=schema_markup,
            image_manifest=list(image_manifest),
            citations=citations,
            word_count=word_count,
            seo_score=seo_score,
        )

        logger.info(
            "Final assembler: %d words | title='%s' | slug='%s' | images=%d | citations=%d",
            word_count, title_tag, url_slug, len(image_manifest), len(citations),
        )

        return {
            "final_output":    final_output,
            "title_tag":       title_tag,
            "meta_description": meta_description,
            "url_slug":        url_slug,
            "error":           None,
            "error_node":      None,
        }

    except Exception as exc:
        logger.exception("Final assembler failed.")
        return {"error": str(exc), "error_node": "final_assembler"}