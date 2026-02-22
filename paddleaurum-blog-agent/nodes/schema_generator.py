import json
import logging
import re
from datetime import date
from typing import List, Optional

from graph.state import AgentState, SchemaMarkup

logger = logging.getLogger(__name__)

_SITE_URL   = "https://paddleaurum.com"
_SITE_NAME  = "PaddleAurum"
_AUTHOR     = {"@type": "Person", "name": "Aurum", "jobTitle": "Certified Pickleball Coach"}


def _build_article_schema(
    title: str,
    description: str,
    url_slug: str,
    word_count: int,
) -> str:
    schema = {
        "@context":         "https://schema.org",
        "@type":            "Article",
        "headline":         title[:110],
        "description":      description[:250],
        "url":              f"{_SITE_URL}/{url_slug}",
        "datePublished":    date.today().isoformat(),
        "dateModified":     date.today().isoformat(),
        "author":           _AUTHOR,
        "publisher": {
            "@type": "Organization",
            "name":  _SITE_NAME,
            "url":   _SITE_URL,
        },
        "wordCount":        word_count,
        "inLanguage":       "en-US",
    }
    return json.dumps(schema, ensure_ascii=False)


def _build_faq_schema(faq_candidates: List[str], draft: str) -> Optional[str]:
    if not faq_candidates:
        return None

    entities: list[dict] = []
    for question in faq_candidates[:8]:
        # Attempt to extract an answer from the draft by finding content near the question keyword
        q_words = question.lower().replace("?", "").split()
        best_sentence = ""
        for sentence in re.split(r"(?<=[.!?])\s+", draft):
            if sum(1 for w in q_words if w in sentence.lower()) >= 2:
                clean = re.sub(r"\[.*?\]|[#*`>_]", "", sentence).strip()
                if len(clean) > 40:
                    best_sentence = clean[:300]
                    break
        if not best_sentence:
            best_sentence = f"See our complete guide on {question.lower().rstrip('?')} above."

        entities.append({
            "@type":           "Question",
            "name":            question,
            "acceptedAnswer": {
                "@type": "Answer",
                "text":  best_sentence,
            },
        })

    if not entities:
        return None

    schema = {
        "@context": "https://schema.org",
        "@type":    "FAQPage",
        "mainEntity": entities,
    }
    return json.dumps(schema, ensure_ascii=False)


def _build_howto_schema(draft: str, title: str) -> Optional[str]:
    # Detect numbered-step sections (HowTo indicator)
    steps = re.findall(r"^\d+\.\s+(.+)$", draft, re.MULTILINE)
    if len(steps) < 3:
        return None

    how_to_steps = [
        {"@type": "HowToStep", "text": step.strip()}
        for step in steps[:10]
    ]

    schema = {
        "@context": "https://schema.org",
        "@type":    "HowTo",
        "name":     title,
        "step":     how_to_steps,
    }
    return json.dumps(schema, ensure_ascii=False)


async def schema_generator_node(state: AgentState) -> dict:
    try:
        draft: str = state.get("formatted_article") or state.get("draft_article") or ""
        title: str = state.get("title_tag") or state.get("topic", "Pickleball Guide")
        meta_desc: str = state.get("meta_description") or ""
        url_slug: str = state.get("url_slug") or "pickleball-guide"
        faq_candidates: List[str] = state.get("faq_candidates") or []
        word_count = len(draft.split())

        article_json = _build_article_schema(title, meta_desc, url_slug, word_count)
        faq_json     = _build_faq_schema(faq_candidates, draft)
        howto_json   = _build_howto_schema(draft, title)

        schema_markup = SchemaMarkup(
            article=article_json,
            faq=faq_json,
            how_to=howto_json,
        )

        schemas_generated = ["Article"] + (["FAQ"] if faq_json else []) + (["HowTo"] if howto_json else [])
        logger.info("Schema generator: produced %s", ", ".join(schemas_generated))

        return {
            "schema_markup": schema_markup,
            "error":         None,
            "error_node":    None,
        }

    except Exception as exc:
        logger.exception("Schema generator failed.")
        return {"error": str(exc), "error_node": "schema_generator"}