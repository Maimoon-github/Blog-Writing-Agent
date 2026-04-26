"""
Web search tool using the Tavily API.
"""

import aiohttp
from pydantic import BaseModel, Field

from blog_agent_system.config.settings import settings
from blog_agent_system.tools.registry import tool
from blog_agent_system.utils.logging import get_logger

logger = get_logger(__name__)


class WebSearchInput(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    max_results: int = Field(default=5, ge=1, le=20)
    search_depth: str = Field(default="basic", pattern=r"^(basic|advanced)$")
    include_domains: list[str] = []


@tool(
    name="web_search",
    description="Search the web for current information on a topic",
    args_schema=WebSearchInput,
)
async def web_search(
    query: str, max_results: int, search_depth: str, include_domains: list
) -> list:
    """Execute a web search via Tavily API."""
    api_key = settings.tavily_api_key
    if not api_key:
        logger.warning("web_search.no_api_key")
        return []

    logger.info("web_search.execute", query=query, max_results=max_results)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.tavily.com/search",
            json={
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "include_domains": include_domains,
                "api_key": api_key,
            },
        ) as resp:
            if resp.status != 200:
                logger.error("web_search.api_error", status=resp.status)
                return []

            data = await resp.json()
            results = [
                {
                    "url": r["url"],
                    "title": r["title"],
                    "snippet": r.get("content", ""),
                    "credibility_score": r.get("score", 0.5),
                }
                for r in data.get("results", [])
            ]
            logger.info("web_search.complete", result_count=len(results))
            return results