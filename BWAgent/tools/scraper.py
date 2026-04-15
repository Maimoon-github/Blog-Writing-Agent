"""Web scraper utilities for BWAgent."""

import asyncio
import random
import re
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


def _cleanup_html(soup: BeautifulSoup) -> None:
    unwanted_tags = ["script", "style", "nav", "footer", "header", "aside", "form", "button", "noscript", "svg"]
    for tag in soup.find_all(unwanted_tags):
        tag.decompose()
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()


def _extract_text(html: str, max_chars: int) -> str:
    soup = BeautifulSoup(html, "html.parser")
    _cleanup_html(soup)
    semantic_tags = ["main", "article", "section", "p", "h1", "h2", "h3", "h4", "h5", "h6"]
    pieces = []
    for tag in soup.find_all(semantic_tags):
        nested = any(parent.name in semantic_tags for parent in tag.parents)
        if nested:
            continue
        text = tag.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        if text:
            pieces.append(text)
    if not pieces and soup.body:
        text = soup.body.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        if text:
            pieces.append(text)
    result = "\n".join(pieces)
    if len(result) > max_chars:
        truncated = result[:max_chars]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            result = truncated[:last_space]
        else:
            result = truncated
    return result.strip()


async def async_fetch_page_content(url: str, max_chars: int = 3000) -> str:
    if not url or not url.strip():
        logger.warning("scraper.empty_url")
        return ""
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, follow_redirects=True, headers=headers) as client:
                response = await client.get(url)
                response.raise_for_status()
            content = _extract_text(response.text, max_chars)
            logger.info("scraper.success", url=url, chars=len(content), attempt=attempt)
            return content
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code >= 500 or code == 429:
                delay = random.uniform(0, BACKOFF_FACTOR ** (attempt - 1))
                logger.warning("scraper.retry_status", url=url, status=code, attempt=attempt, delay=delay)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(delay)
                    continue
            logger.error("scraper.http_error", url=url, status=code, error=str(exc))
            return ""
        except httpx.RequestError as exc:
            if attempt < MAX_RETRIES:
                delay = random.uniform(0, BACKOFF_FACTOR ** (attempt - 1))
                logger.warning("scraper.retry_request", url=url, attempt=attempt, error=str(exc), delay=delay)
                await asyncio.sleep(delay)
                continue
            logger.error("scraper.request_failed", url=url, error=str(exc))
            return ""
        except Exception as exc:
            logger.error("scraper.unexpected_error", url=url, error=str(exc))
            return ""
    logger.error("scraper.failed", url=url)
    return ""
