"""tools/ — Custom utilities for the paddleaurum.com multi-agent pipeline."""

from tools.rag_retrieval import (
    rag_retrieve,
    COLLECTION_PICKLEBALL_RULES,
    COLLECTION_COACHING_MATERIALS,
    COLLECTION_SEO_GUIDELINES,
    COLLECTION_PUBLISHED_ARTICLES,
    COLLECTION_KEYWORD_HISTORY,
)
from tools.keyword_analysis import calculate_keyword_density, extract_lsi_terms
from tools.word_count import word_count, word_count_excluding_code
from tools.url_validator import validate_url, check_url_accessibility
from tools.serp_analysis import analyze_serp_patterns, SerpAnalysis

__all__ = [
    # RAG retrieval
    "rag_retrieve",
    "COLLECTION_PICKLEBALL_RULES",
    "COLLECTION_COACHING_MATERIALS",
    "COLLECTION_SEO_GUIDELINES",
    "COLLECTION_PUBLISHED_ARTICLES",
    "COLLECTION_KEYWORD_HISTORY",
    # Keyword analysis
    "calculate_keyword_density",
    "extract_lsi_terms",
    # Word count
    "word_count",
    "word_count_excluding_code",
    # URL validation
    "validate_url",
    "check_url_accessibility",
    # SERP analysis
    "analyze_serp_patterns",
    "SerpAnalysis",
]