# """tools/keyword_analysis.py

# Keyword density calculation and LSI term extraction.

# Design notes
# ------------
# `calculate_keyword_density` uses phrase-level matching with re.escape so that
# multi-word primary keywords (e.g. "pickleball kitchen rules") are counted as
# full-phrase occurrences divided by total word count.  This is identical to the
# formula used in nodes/seo_auditor.py so that tool output and auditor scores
# are always consistent.

# Word count uses `re.findall(r"\\b\\w+\\b", text)` — the same tokeniser used
# in nodes/seo_auditor.py and tools/word_count.py.
# """

# from __future__ import annotations

# import re
# from collections import Counter
# from typing import List, Optional

# # ── Stop-word set ─────────────────────────────────────────────────────────────
# # frozenset for O(1) membership tests.
# _STOPWORDS: frozenset[str] = frozenset({
#     "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
#     "which", "this", "that", "these", "those", "then", "just", "so", "than",
#     "such", "both", "through", "about", "for", "is", "of", "while", "during",
#     "to", "from", "in", "on", "at", "by", "with", "without", "after", "before",
#     "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
#     "does", "did", "will", "would", "could", "should", "may", "might", "shall",
#     "not", "no", "its", "it", "we", "you", "he", "she", "they", "i", "my",
#     "your", "our", "their", "also", "can", "more", "all", "into", "up",
# })


# def _token_count(text: str) -> int:
#     """Count words using the same tokeniser as seo_auditor._count_words."""
#     return len(re.findall(r"\b\w+\b", text))


# def calculate_keyword_density(text: str, keyword: str) -> float:
#     """
#     Return keyword density as a percentage of total word count.

#     Phrase-level matching is used so that multi-word keywords (e.g.
#     "pickleball kitchen rules") are counted as complete-phrase occurrences.
#     This matches the formula in nodes/seo_auditor.py:

#         occurrences = len(re.findall(re.escape(keyword.lower()), text.lower()))
#         density = (occurrences / total_words) * 100

#     Parameters
#     ----------
#     text    : Full article text (Markdown or plain).
#     keyword : Keyword or keyword phrase (case-insensitive).

#     Returns
#     -------
#     float
#         Density as a percentage (e.g. 1.5 means 1.5%).
#         Returns 0.0 if either argument is empty.
#     """
#     if not text or not keyword:
#         return 0.0

#     total_words = _token_count(text)
#     if total_words == 0:
#         return 0.0

#     occurrences = len(re.findall(re.escape(keyword.lower()), text.lower()))
#     return round((occurrences / total_words) * 100, 2)


# def extract_lsi_terms(
#     text: str,
#     top_n: int = 10,
#     min_word_length: int = 3,
#     exclude: Optional[List[str]] = None,
# ) -> List[str]:
#     """
#     Extract the top-N content-bearing words from text as LSI term candidates.

#     Words are ranked by frequency after removing stop words, the caller-supplied
#     exclusion list, and any words shorter than `min_word_length`.  Callers should
#     pass the primary keyword and its component words via `exclude` so that the
#     keyword itself does not appear in the LSI list.

#     Parameters
#     ----------
#     text            : Article text to analyse.
#     top_n           : Maximum number of terms to return.
#     min_word_length : Minimum character length for a word to be included.
#     exclude         : Additional words/phrases to suppress (e.g. the primary
#                       keyword and its individual tokens).

#     Returns
#     -------
#     List[str]
#         Unique terms ordered by descending frequency.
#     """
#     if not text:
#         return []

#     # Build exclusion set: stop words + caller-supplied extras + their tokens
#     exclude_tokens: set[str] = set(_STOPWORDS)
#     if exclude:
#         for phrase in exclude:
#             exclude_tokens.add(phrase.lower())
#             # Also exclude individual words within multi-word phrases
#             exclude_tokens.update(re.findall(r"\b[a-zA-Z]+\b", phrase.lower()))

#     words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
#     filtered = [
#         w for w in words
#         if w not in exclude_tokens and len(w) >= min_word_length
#     ]

#     counter = Counter(filtered)
#     return [term for term, _ in counter.most_common(top_n)]




















# @#########################################################################


























# tools/keyword_analysis.py
"""tools/keyword_analysis.py

Keyword density calculation and LSI term extraction.

Design notes
------------
`calculate_keyword_density` uses phrase-level matching with re.escape so that
multi-word primary keywords (e.g. "pickleball kitchen rules") are counted as
full-phrase occurrences divided by total word count.  This is identical to the
formula used in nodes/seo_auditor.py so that tool output and auditor scores
are always consistent.

Word count uses `re.findall(r"\\b\\w+\\b", text)` — the same tokeniser used
in nodes/seo_auditor.py and tools/word_count.py.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional

# ── Stop-word set ─────────────────────────────────────────────────────────────
# frozenset for O(1) membership tests.
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "of", "while", "during",
    "to", "from", "in", "on", "at", "by", "with", "without", "after", "before",
    "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may", "might", "shall",
    "not", "no", "its", "it", "we", "you", "he", "she", "they", "i", "my",
    "your", "our", "their", "also", "can", "more", "all", "into", "up",
})


def _token_count(text: str) -> int:
    """Count words using the same tokeniser as seo_auditor._count_words."""
    return len(re.findall(r"\b\w+\b", text))


def calculate_keyword_density(text: str, keyword: str) -> float:
    """
    Return keyword density as a percentage of total word count.

    Phrase-level matching is used so that multi-word keywords (e.g.
    "pickleball kitchen rules") are counted as complete-phrase occurrences.
    This matches the formula in nodes/seo_auditor.py:

        occurrences = len(re.findall(re.escape(keyword.lower()), text.lower()))
        density = (occurrences / total_words) * 100

    Parameters
    ----------
    text    : Full article text (Markdown or plain).
    keyword : Keyword or keyword phrase (case-insensitive).

    Returns
    -------
    float
        Density as a percentage (e.g. 1.5 means 1.5%).
        Returns 0.0 if either argument is empty.
    """
    if not text or not keyword:
        return 0.0

    total_words = _token_count(text)
    if total_words == 0:
        return 0.0

    occurrences = len(re.findall(re.escape(keyword.lower()), text.lower()))
    return round((occurrences / total_words) * 100, 2)


def extract_lsi_terms(
    text: str,
    top_n: int = 10,
    min_word_length: int = 3,
    exclude: Optional[List[str]] = None,
) -> List[str]:
    """
    Extract the top-N content-bearing words from text as LSI term candidates.

    Words are ranked by frequency after removing stop words, the caller-supplied
    exclusion list, and any words shorter than `min_word_length`.  Callers should
    pass the primary keyword and its component words via `exclude` so that the
    keyword itself does not appear in the LSI list.

    Parameters
    ----------
    text            : Article text to analyse.
    top_n           : Maximum number of terms to return.
    min_word_length : Minimum character length for a word to be included.
    exclude         : Additional words/phrases to suppress (e.g. the primary
                      keyword and its individual tokens).

    Returns
    -------
    List[str]
        Unique terms ordered by descending frequency.
    """
    if not text:
        return []

    # Build exclusion set: stop words + caller-supplied extras + their tokens
    exclude_tokens: set[str] = set(_STOPWORDS)
    if exclude:
        for phrase in exclude:
            exclude_tokens.add(phrase.lower())
            # Also exclude individual words within multi-word phrases
            exclude_tokens.update(re.findall(r"\b[a-zA-Z]+\b", phrase.lower()))

    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    filtered = [
        w for w in words
        if w not in exclude_tokens and len(w) >= min_word_length
    ]

    counter = Counter(filtered)
    return [term for term, _ in counter.most_common(top_n)]