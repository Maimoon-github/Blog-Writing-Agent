# src/blog_agent_system/tools/web_search.py
from pydantic import BaseModel, Field
from blog_agent_system.tools.registry import tool
import aiohttp

class WebSearchInput(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    max_results: int = Field(default=5, ge=1, le=20)
    search_depth: str = Field(default="basic", pattern="^(basic|advanced)$")
    include_domains: list[str] = []

@tool(
    name="web_search",
    description="Search the web for current information on a topic",
    args_schema=WebSearchInput
)
async def web_search(query: str, max_results: int, search_depth: str, include_domains: list) -> list:
    api_key = settings.TAVILY_API_KEY
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.tavily.com/search",
            json={
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "include_domains": include_domains,
                "api_key": api_key
            }
        ) as resp:
            data = await resp.json()
            return [
                {
                    "url": r["url"],
                    "title": r["title"],
                    "snippet": r["content"],
                    "credibility_score": r.get("score", 0.5)
                }
                for r in data.get("results", [])
            ]