# """tools/word_count.py

# Word count utility.

# The tokeniser — `re.findall(r"\\b\\w+\\b", text)` — is identical to the one
# used in nodes/seo_auditor.py so that word counts reported by this tool always
# agree with the counts the auditor uses when computing keyword density and
# evaluating article length.
# """

# from __future__ import annotations

# import re


# def word_count(text: str) -> int:
#     """
#     Return the number of words in `text`.

#     A word is any maximal sequence of word characters (letters, digits,
#     underscore) bounded by word boundaries.  This definition strips Markdown
#     punctuation, heading markers, and bracket characters so that structural
#     markup does not inflate the count.

#     Parameters
#     ----------
#     text : Any text string, including Markdown.

#     Returns
#     -------
#     int
#         Word count.  Returns 0 for empty or whitespace-only input.
#     """
#     if not text:
#         return 0
#     return len(re.findall(r"\b\w+\b", text))


# def word_count_excluding_code(text: str) -> int:
#     """
#     Return word count after stripping fenced code blocks.

#     Useful when an article contains code examples that should not inflate
#     the prose word count reported to the SEO Auditor.

#     Parameters
#     ----------
#     text : Markdown text that may contain fenced code blocks (``` ... ```).

#     Returns
#     -------
#     int
#         Word count of non-code content only.
#     """
#     stripped = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
#     return word_count(stripped)























# 23#################################################################################################################






















# tools/word_count.py
"""tools/word_count.py

Word count utility.

The tokeniser — `re.findall(r"\\b\\w+\\b", text)` — is identical to the one
used in nodes/seo_auditor.py so that word counts reported by this tool always
agree with the counts the auditor uses when computing keyword density and
evaluating article length.
"""

from __future__ import annotations

import re


def word_count(text: str) -> int:
    """
    Return the number of words in `text`.

    A word is any maximal sequence of word characters (letters, digits,
    underscore) bounded by word boundaries.  This definition strips Markdown
    punctuation, heading markers, and bracket characters so that structural
    markup does not inflate the count.

    Parameters
    ----------
    text : Any text string, including Markdown.

    Returns
    -------
    int
        Word count.  Returns 0 for empty or whitespace-only input.
    """
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", text))


def word_count_excluding_code(text: str) -> int:
    """
    Return word count after stripping fenced code blocks.

    Useful when an article contains code examples that should not inflate
    the prose word count reported to the SEO Auditor.

    Parameters
    ----------
    text : Markdown text that may contain fenced code blocks (``` ... ```).

    Returns
    -------
    int
        Word count of non-code content only.
    """
    stripped = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    return word_count(stripped)