"""Simple word count utility."""

import re


def word_count(text: str) -> int:
    """
    Count the number of words in a text string.

    A word is defined as any sequence of characters separated by whitespace.
    """
    if not text:
        return 0
    # Split on any whitespace and filter out empty strings
    words = re.split(r"\s+", text.strip())
    return len([w for w in words if w])