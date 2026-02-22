"""Extract patterns from top search result snippets."""

from typing import List, Dict, Any, Optional
from collections import Counter
import re

# Stopwords for pattern extraction (minimal)
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "of", "while", "during",
    "to", "from", "in", "on", "at", "by", "with", "without", "after", "before",
}


def analyze_serp_patterns(
    results: List[Dict[str, str]],
    top_n_terms: int = 10,
) -> Dict[str, Any]:
    """
    Analyze a list of SERP result items to extract common patterns.

    Each result dict should contain at least:
        - "title": the page title
        - "snippet": the meta description or text snippet
        - "url": the page URL (optional)

    Returns a dictionary with:
        - "avg_title_length": average character length of titles
        - "avg_snippet_length": average character length of snippets
        - "common_title_terms": list of frequent terms in titles
        - "common_snippet_terms": list of frequent terms in snippets
        - "domains": list of unique domains (if URLs provided)
        - "total_results": number of results analyzed
    """
    if not results:
        return {
            "avg_title_length": 0,
            "avg_snippet_length": 0,
            "common_title_terms": [],
            "common_snippet_terms": [],
            "domains": [],
            "total_results": 0,
        }

    title_lengths = []
    snippet_lengths = []
    title_words = []
    snippet_words = []
    domains = []

    for r in results:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        url = r.get("url", "")

        title_lengths.append(len(title))
        snippet_lengths.append(len(snippet))

        # Extract words from title
        if title:
            words = re.findall(r"\b[a-zA-Z]+\b", title.lower())
            title_words.extend(w for w in words if w not in STOPWORDS)

        # Extract words from snippet
        if snippet:
            words = re.findall(r"\b[a-zA-Z]+\b", snippet.lower())
            snippet_words.extend(w for w in words if w not in STOPWORDS)

        # Extract domain from URL
        if url:
            import urllib.parse
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc
            if domain:
                domains.append(domain)

    # Compute averages
    avg_title = sum(title_lengths) / len(title_lengths) if title_lengths else 0
    avg_snippet = sum(snippet_lengths) / len(snippet_lengths) if snippet_lengths else 0

    # Top terms
    title_counter = Counter(title_words)
    snippet_counter = Counter(snippet_words)

    return {
        "avg_title_length": round(avg_title, 1),
        "avg_snippet_length": round(avg_snippet, 1),
        "common_title_terms": [w for w, _ in title_counter.most_common(top_n_terms)],
        "common_snippet_terms": [w for w, _ in snippet_counter.most_common(top_n_terms)],
        "domains": list(set(domains))[:10],  # deduplicate, limit
        "total_results": len(results),
    }