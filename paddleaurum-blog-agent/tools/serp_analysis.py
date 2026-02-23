# """tools/serp_analysis.py

# Structured analysis of SERP result snippets.

# Input is a list of dicts whose keys match the ResearchSnippet TypedDict
# (graph/state.py): title, snippet, url, query, retrieved_at.  Only title,
# snippet, and url are required; the rest are ignored.

# Output is a SerpAnalysis TypedDict that the Content Strategist and Outline
# Agent use to understand what top-ranking competitor pages look like — average
# lengths, dominant heading patterns, inferred content types, common terms, and
# unique domains.
# """

# from __future__ import annotations

# import re
# from collections import Counter
# from typing import Any, Dict, List, TypedDict
# from urllib.parse import urlparse

# from tools.word_count import word_count

# # ── Output type ───────────────────────────────────────────────────────────────


# class SerpAnalysis(TypedDict):
#     total_results:         int
#     avg_title_length:      float     # chars
#     avg_snippet_length:    float     # chars
#     avg_snippet_word_count: float    # words (consistent with word_count tool)
#     common_title_terms:    List[str] # top words found across titles
#     common_snippet_terms:  List[str] # top words found across snippets
#     heading_patterns:      Dict[str, int]  # e.g. {"How to": 3, "Best": 2}
#     content_types:         Dict[str, int]  # e.g. {"guide": 4, "listicle": 2}
#     intent_signals:        Dict[str, int]  # e.g. {"informational": 5, "commercial": 1}
#     domains:               List[str]       # unique domains, up to 10


# # ── Stop-word set ─────────────────────────────────────────────────────────────
# _STOPWORDS: frozenset[str] = frozenset({
#     "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
#     "which", "this", "that", "these", "those", "then", "just", "so", "than",
#     "such", "both", "through", "about", "for", "is", "of", "while", "during",
#     "to", "from", "in", "on", "at", "by", "with", "without", "after", "before",
#     "are", "was", "were", "have", "has", "do", "does", "will", "can", "not",
#     "it", "its", "you", "your", "we", "our", "they", "their", "be", "been",
# })

# # ── Pattern classifiers ───────────────────────────────────────────────────────
# # Ordered: first match wins for content_type and intent_signal.

# _HEADING_PREFIXES: List[str] = [
#     "How to", "How To",
#     "What Is", "What is",
#     "Why", "When", "Where",
#     "Best", "Top", "Ultimate",
#     "Guide", "Tutorial",
#     "Review", "vs", "Vs",
#     "Tips", "Rules",
# ]

# # Maps title keywords → content_type label
# _CONTENT_TYPE_SIGNALS: List[tuple[str, str]] = [
#     (r"\b(?:top|best)\s+\d+\b",           "listicle"),
#     (r"\b(?:guide|complete guide|tutorial)\b", "guide"),
#     (r"\b(?:review|reviews|tested)\b",     "review"),
#     (r"\bvs\.?\b",                         "comparison"),
#     (r"\b(?:how\s+to|step[-\s]by[-\s]step)\b", "how_to"),
#     (r"\b(?:what\s+is|explained|overview)\b",  "explainer"),
#     (r"\b(?:tips|tricks|mistakes)\b",      "tips"),
#     (r"\b(?:rules|regulations|scoring)\b", "rules"),
# ]

# # Maps title/snippet keywords → intent_signal label
# _INTENT_SIGNALS: List[tuple[str, str]] = [
#     (r"\b(?:buy|shop|price|cheap|discount|deal)\b",   "commercial"),
#     (r"\b(?:best|top|review|vs|compare|versus)\b",    "commercial"),
#     (r"\b(?:how\s+to|tutorial|guide|learn|tips|rules)\b", "informational"),
#     (r"\b(?:what\s+is|explained|overview|definition)\b",  "informational"),
#     (r"\b(?:paddleaurum|site:|site:paddleaurum)\b",    "navigational"),
# ]


# def _extract_domain(url: str) -> str:
#     try:
#         return urlparse(url).netloc.removeprefix("www.")
#     except Exception:
#         return ""


# def _top_terms(texts: List[str], top_n: int) -> List[str]:
#     words: List[str] = []
#     for text in texts:
#         tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
#         words.extend(t for t in tokens if t not in _STOPWORDS and len(t) >= 3)
#     return [w for w, _ in Counter(words).most_common(top_n)]


# def _detect_heading_patterns(titles: List[str]) -> Dict[str, int]:
#     counts: Counter[str] = Counter()
#     for title in titles:
#         for prefix in _HEADING_PREFIXES:
#             if re.search(re.escape(prefix), title, re.IGNORECASE):
#                 counts[prefix] += 1
#     return dict(counts.most_common())


# def _detect_content_type(title: str) -> str:
#     lower = title.lower()
#     for pattern, label in _CONTENT_TYPE_SIGNALS:
#         if re.search(pattern, lower):
#             return label
#     return "article"


# def _detect_intent(title: str, snippet: str) -> str:
#     combined = (title + " " + snippet).lower()
#     for pattern, label in _INTENT_SIGNALS:
#         if re.search(pattern, combined):
#             return label
#     return "informational"


# def analyze_serp_patterns(
#     results: List[Dict[str, Any]],
#     top_n_terms: int = 10,
# ) -> SerpAnalysis:
#     """
#     Analyse a list of SERP result dicts and return structured competitor patterns.

#     Each dict is expected to contain at minimum:
#         "title"   — page/article title (str)
#         "snippet" — meta description or text excerpt (str)
#         "url"     — source URL (str, optional)

#     Additional keys (query, retrieved_at) are accepted but ignored.

#     Parameters
#     ----------
#     results     : List of SERP result dicts (typically from ResearchSnippet).
#     top_n_terms : Number of top terms to surface for title and snippet analysis.

#     Returns
#     -------
#     SerpAnalysis
#         Structured analysis dict.  All counts are 0 / empty on empty input.
#     """
#     _empty: SerpAnalysis = SerpAnalysis(
#         total_results=0,
#         avg_title_length=0.0,
#         avg_snippet_length=0.0,
#         avg_snippet_word_count=0.0,
#         common_title_terms=[],
#         common_snippet_terms=[],
#         heading_patterns={},
#         content_types={},
#         intent_signals={},
#         domains=[],
#     )

#     if not results:
#         return _empty

#     titles:   List[str] = []
#     snippets: List[str] = []
#     domains:  List[str] = []
#     content_type_counter: Counter[str] = Counter()
#     intent_counter:       Counter[str] = Counter()

#     for r in results:
#         title   = str(r.get("title", "") or "")
#         snippet = str(r.get("snippet", "") or "")
#         url     = str(r.get("url", "") or "")

#         titles.append(title)
#         snippets.append(snippet)

#         domain = _extract_domain(url)
#         if domain:
#             domains.append(domain)

#         content_type_counter[_detect_content_type(title)] += 1
#         intent_counter[_detect_intent(title, snippet)] += 1

#     title_lengths   = [len(t) for t in titles]
#     snippet_lengths = [len(s) for s in snippets]
#     snippet_wcs     = [word_count(s) for s in snippets]

#     n = len(results)

#     return SerpAnalysis(
#         total_results=n,
#         avg_title_length=round(sum(title_lengths) / n, 1),
#         avg_snippet_length=round(sum(snippet_lengths) / n, 1),
#         avg_snippet_word_count=round(sum(snippet_wcs) / n, 1),
#         common_title_terms=_top_terms(titles, top_n_terms),
#         common_snippet_terms=_top_terms(snippets, top_n_terms),
#         heading_patterns=_detect_heading_patterns(titles),
#         content_types=dict(content_type_counter.most_common()),
#         intent_signals=dict(intent_counter.most_common()),
#         domains=list(dict.fromkeys(domains))[:10],  # deduplicated, insertion-ordered
#     )
















# @#######################################################################################################




















# tools/serp_analysis.py
"""tools/serp_analysis.py

Structured analysis of SERP result snippets.

Input is a list of dicts whose keys match the ResearchSnippet TypedDict
(graph/state.py): title, snippet, url, query, retrieved_at.  Only title,
snippet, and url are required; the rest are ignored.

Output is a SerpAnalysis TypedDict that the Content Strategist and Outline
Agent use to understand what top-ranking competitor pages look like — average
lengths, dominant heading patterns, inferred content types, common terms, and
unique domains.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, TypedDict
from urllib.parse import urlparse

from tools.word_count import word_count

# ── Output type ───────────────────────────────────────────────────────────────


class SerpAnalysis(TypedDict):
    total_results:         int
    avg_title_length:      float     # chars
    avg_snippet_length:    float     # chars
    avg_snippet_word_count: float    # words (consistent with word_count tool)
    common_title_terms:    List[str] # top words found across titles
    common_snippet_terms:  List[str] # top words found across snippets
    heading_patterns:      Dict[str, int]  # e.g. {"How to": 3, "Best": 2}
    content_types:         Dict[str, int]  # e.g. {"guide": 4, "listicle": 2}
    intent_signals:        Dict[str, int]  # e.g. {"informational": 5, "commercial": 1}
    domains:               List[str]       # unique domains, up to 10


# ── Stop-word set ─────────────────────────────────────────────────────────────
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "of", "while", "during",
    "to", "from", "in", "on", "at", "by", "with", "without", "after", "before",
    "are", "was", "were", "have", "has", "do", "does", "will", "can", "not",
    "it", "its", "you", "your", "we", "our", "they", "their", "be", "been",
})

# ── Pattern classifiers ───────────────────────────────────────────────────────
# Ordered: first match wins for content_type and intent_signal.

_HEADING_PREFIXES: List[str] = [
    "How to", "How To",
    "What Is", "What is",
    "Why", "When", "Where",
    "Best", "Top", "Ultimate",
    "Guide", "Tutorial",
    "Review", "vs", "Vs",
    "Tips", "Rules",
]

# Maps title keywords → content_type label
_CONTENT_TYPE_SIGNALS: List[tuple[str, str]] = [
    (r"\b(?:top|best)\s+\d+\b",           "listicle"),
    (r"\b(?:guide|complete guide|tutorial)\b", "guide"),
    (r"\b(?:review|reviews|tested)\b",     "review"),
    (r"\bvs\.?\b",                         "comparison"),
    (r"\b(?:how\s+to|step[-\s]by[-\s]step)\b", "how_to"),
    (r"\b(?:what\s+is|explained|overview)\b",  "explainer"),
    (r"\b(?:tips|tricks|mistakes)\b",      "tips"),
    (r"\b(?:rules|regulations|scoring)\b", "rules"),
]

# Maps title/snippet keywords → intent_signal label
_INTENT_SIGNALS: List[tuple[str, str]] = [
    (r"\b(?:buy|shop|price|cheap|discount|deal)\b",   "commercial"),
    (r"\b(?:best|top|review|vs|compare|versus)\b",    "commercial"),
    (r"\b(?:how\s+to|tutorial|guide|learn|tips|rules)\b", "informational"),
    (r"\b(?:what\s+is|explained|overview|definition)\b",  "informational"),
    (r"\b(?:paddleaurum|site:|site:paddleaurum)\b",    "navigational"),
]


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.removeprefix("www.")
    except Exception:
        return ""


def _top_terms(texts: List[str], top_n: int) -> List[str]:
    words: List[str] = []
    for text in texts:
        tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        words.extend(t for t in tokens if t not in _STOPWORDS and len(t) >= 3)
    return [w for w, _ in Counter(words).most_common(top_n)]


def _detect_heading_patterns(titles: List[str]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for title in titles:
        for prefix in _HEADING_PREFIXES:
            if re.search(re.escape(prefix), title, re.IGNORECASE):
                counts[prefix] += 1
    return dict(counts.most_common())


def _detect_content_type(title: str) -> str:
    lower = title.lower()
    for pattern, label in _CONTENT_TYPE_SIGNALS:
        if re.search(pattern, lower):
            return label
    return "article"


def _detect_intent(title: str, snippet: str) -> str:
    combined = (title + " " + snippet).lower()
    for pattern, label in _INTENT_SIGNALS:
        if re.search(pattern, combined):
            return label
    return "informational"


def analyze_serp_patterns(
    results: List[Dict[str, Any]],
    top_n_terms: int = 10,
) -> SerpAnalysis:
    """
    Analyse a list of SERP result dicts and return structured competitor patterns.

    Each dict is expected to contain at minimum:
        "title"   — page/article title (str)
        "snippet" — meta description or text excerpt (str)
        "url"     — source URL (str, optional)

    Additional keys (query, retrieved_at) are accepted but ignored.

    Parameters
    ----------
    results     : List of SERP result dicts (typically from ResearchSnippet).
    top_n_terms : Number of top terms to surface for title and snippet analysis.

    Returns
    -------
    SerpAnalysis
        Structured analysis dict.  All counts are 0 / empty on empty input.
    """
    _empty: SerpAnalysis = SerpAnalysis(
        total_results=0,
        avg_title_length=0.0,
        avg_snippet_length=0.0,
        avg_snippet_word_count=0.0,
        common_title_terms=[],
        common_snippet_terms=[],
        heading_patterns={},
        content_types={},
        intent_signals={},
        domains=[],
    )

    if not results:
        return _empty

    titles:   List[str] = []
    snippets: List[str] = []
    domains:  List[str] = []
    content_type_counter: Counter[str] = Counter()
    intent_counter:       Counter[str] = Counter()

    for r in results:
        title   = str(r.get("title", "") or "")
        snippet = str(r.get("snippet", "") or "")
        url     = str(r.get("url", "") or "")

        titles.append(title)
        snippets.append(snippet)

        domain = _extract_domain(url)
        if domain:
            domains.append(domain)

        content_type_counter[_detect_content_type(title)] += 1
        intent_counter[_detect_intent(title, snippet)] += 1

    title_lengths   = [len(t) for t in titles]
    snippet_lengths = [len(s) for s in snippets]
    snippet_wcs     = [word_count(s) for s in snippets]

    n = len(results)

    return SerpAnalysis(
        total_results=n,
        avg_title_length=round(sum(title_lengths) / n, 1),
        avg_snippet_length=round(sum(snippet_lengths) / n, 1),
        avg_snippet_word_count=round(sum(snippet_wcs) / n, 1),
        common_title_terms=_top_terms(titles, top_n_terms),
        common_snippet_terms=_top_terms(snippets, top_n_terms),
        heading_patterns=_detect_heading_patterns(titles),
        content_types=dict(content_type_counter.most_common()),
        intent_signals=dict(intent_counter.most_common()),
        domains=list(dict.fromkeys(domains))[:10],  # deduplicated, insertion-ordered
    )