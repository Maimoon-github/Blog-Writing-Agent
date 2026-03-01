"""
tools/web_fetcher.py
====================
Fetch and extract clean text content from web pages.

This module provides a resilient function to extract clean, readable text
from a given URL using httpx and BeautifulSoup. It features automatic retries
with exponential backoff, comprehensive error handling, and structured logging.

Primary function: `async_fetch_page_content` (async)
Synchronous wrapper: `fetch_page_content_sync` (calls the async version)
"""

import asyncio
import re
import random
from typing import Optional

import httpx
from bs4 import BeautifulSoup, Comment
import structlog

logger = structlog.get_logger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/115.0.0.0 Safari/537.36"
)
DEFAULT_TIMEOUT = 10.0
MAX_RETRIES = 3
BACKOFF_FACTOR = 2.0


def _clean_html(soup: BeautifulSoup) -> None:
    """Remove unwanted elements from the BeautifulSoup object in place."""
    unwanted_tags = [
        "script", "style", "nav", "footer", "header", "aside",
        "form", "button", "noscript", "svg"
    ]
    for tag in soup.find_all(unwanted_tags):
        tag.decompose()

    # Discard HTML comments as they are not visible text
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()


def _process_html(html: str, max_chars: int) -> str:
    """Parse HTML, clean it, extract semantic text, normalize, and truncate."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content tags
    _clean_html(soup)

    semantic_tags = ["main", "article", "section", "p", "h1", "h2", "h3", "h4", "h5", "h6"]
    extracted_pieces = []

    # Extract text primarily from root‑most semantic tags to prevent duplication
    for tag in soup.find_all(semantic_tags):
        # Prevent extracting from a tag if one of its parents is already in the semantic list
        is_nested = any(parent.name in semantic_tags for parent in tag.parents)
        if not is_nested:
            text = tag.get_text(separator=" ", strip=True)
            # Normalize whitespace into a single space within this block
            text = re.sub(r'\s+', ' ', text)
            if text:
                extracted_pieces.append(text)

    # Fallback if no semantic tags exist but there is body text
    if not extracted_pieces and soup.body:
        text = soup.body.get_text(separator=" ", strip=True)
        text = re.sub(r'\s+', ' ', text)
        if text:
            extracted_pieces.append(text)

    # Join extracted pieces with a newline to preserve paragraph separation
    final_text = "\n".join(extracted_pieces)

    # Truncate string length, ensuring we don't cut words in half
    if len(final_text) > max_chars:
        truncated = final_text[:max_chars]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            final_text = truncated[:last_space]
        else:
            final_text = truncated

    return final_text.strip()


async def async_fetch_page_content(url: str, max_chars: int = 3000) -> str:
    """
    Fetch web page content asynchronously and extract clean text.

    Args:
        url: The URL to fetch.
        max_chars: Maximum number of characters to return. Default is 3000.

    Returns:
        A clean, truncated string of extracted text, or an empty string on failure.
    """
    log = logger.bind(url=url, max_chars=max_chars)

    # Edge case: empty URL
    if not url or not url.strip():
        log.warning("empty_url")
        return ""

    headers = {"User-Agent": USER_AGENT}

    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(
                timeout=DEFAULT_TIMEOUT, follow_redirects=True, headers=headers
            ) as client:
                response = await client.get(url)
                response.raise_for_status()

            text_content = _process_html(response.text, max_chars)
            log.info("async_fetch_page_content_success", chars=len(text_content))
            return text_content

        except httpx.HTTPStatusError as e:
            # 5xx errors and 429 are typically transient; others are permanent
            is_transient = e.response.status_code >= 500 or e.response.status_code == 429
            if is_transient and attempt < MAX_RETRIES - 1:
                # Exponential backoff with full jitter
                delay = BACKOFF_FACTOR ** attempt
                delay = random.uniform(0, delay)
                log.warning(
                    "async_fetch_page_content_status_error_retry",
                    status_code=e.response.status_code,
                    attempt=attempt + 1,
                    next_delay=delay,
                )
                await asyncio.sleep(delay)
            else:
                log.error(
                    "async_fetch_page_content_permanent_error",
                    error=str(e),
                    status_code=e.response.status_code,
                )
                return ""
        except httpx.RequestError as e:
            if attempt < MAX_RETRIES - 1:
                delay = BACKOFF_FACTOR ** attempt
                delay = random.uniform(0, delay)
                log.warning(
                    "async_fetch_page_content_network_error_retry",
                    attempt=attempt + 1,
                    error=str(e),
                    next_delay=delay,
                )
                await asyncio.sleep(delay)
            else:
                log.error("async_fetch_page_content_network_failed", error=str(e))
                return ""
        except Exception as e:
            log.error(
                "async_fetch_page_content_unexpected_error",
                error=str(e),
                exc_info=True,
            )
            return ""

    log.error("async_fetch_page_content_failed", error="max retries reached")
    return ""


def fetch_page_content_sync(url: str, max_chars: int = 3000) -> str:
    """Synchronous wrapper around `async_fetch_page_content`."""
    return asyncio.run(async_fetch_page_content(url, max_chars))