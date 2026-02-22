"""Keyword density calculation and LSI term extraction."""

import re
from collections import Counter
from typing import List, Optional

# Simple stopword list (extend as needed)
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "of", "while", "during",
    "to", "from", "in", "on", "at", "by", "with", "without", "after", "before",
}


def calculate_keyword_density(text: str, keyword: str) -> float:
    """
    Calculate the percentage of keyword occurrences relative to total words.

    Args:
        text: The full article text.
        keyword: The keyword to count (case‑insensitive).

    Returns:
        Density as a float between 0 and 100.
    """
    if not text or not keyword:
        return 0.0

    # Normalize: lowercase and split on whitespace
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0

    keyword_lower = keyword.lower()
    # Count whole‑word matches only
    count = sum(1 for w in words if w == keyword_lower)
    return (count / len(words)) * 100.0


def extract_lsi_terms(
    text: str,
    top_n: int = 10,
    min_word_length: int = 3,
    exclude: Optional[List[str]] = None,
) -> List[str]:
    """
    Extract Latent Semantic Indexing (LSI) terms – frequently occurring,
    content‑bearing words from the text.

    Args:
        text: The article text.
        top_n: Number of top terms to return.
        min_word_length: Minimum length of words to consider.
        exclude: Additional words to exclude (besides STOPWORDS).

    Returns:
        List of LSI term strings, ordered by frequency descending.
    """
    if not text:
        return []

    # Tokenize and filter
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if not words:
        return []

    exclude_set = set(STOPWORDS)
    if exclude:
        exclude_set.update(w.lower() for w in exclude)

    # Count frequencies of valid words
    counter = Counter(
        w for w in words
        if w not in exclude_set and len(w) >= min_word_length
    )

    # Return top N terms
    return [word for word, _ in counter.most_common(top_n)]