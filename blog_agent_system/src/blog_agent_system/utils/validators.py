"""
Input sanitization and validation utilities.
"""

import re
from urllib.parse import urlparse


def validate_url(url: str) -> bool:
    """Validate that a string is a well-formed URL with http(s) scheme."""
    try:
        result = urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except ValueError:
        return False


def sanitize_topic(topic: str) -> str:
    """Sanitize a blog topic string."""
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", topic)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to max_length, adding suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix
