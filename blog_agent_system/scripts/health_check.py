"""
Health check script — verifies all infrastructure dependencies.
"""

import asyncio
import sys


async def main():
    results = {}

    # PostgreSQL
    try:
        from blog_agent_system.persistence.database import async_engine
        from sqlalchemy import text

        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        results["PostgreSQL"] = "✅ Connected"
    except Exception as e:
        results["PostgreSQL"] = f"❌ {e}"

    # Redis
    try:
        import redis.asyncio as aioredis
        from blog_agent_system.config.settings import settings

        r = aioredis.from_url(settings.redis_url)
        await r.ping()
        await r.aclose()
        results["Redis"] = "✅ Connected"
    except Exception as e:
        results["Redis"] = f"❌ {e}"

    # ChromaDB
    try:
        import chromadb
        from blog_agent_system.config.settings import settings

        client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        client.heartbeat()
        results["ChromaDB"] = "✅ Connected"
    except Exception as e:
        results["ChromaDB"] = f"❌ {e}"

    print("\n🏥 Health Check Results:\n")
    all_healthy = True
    for service, status in results.items():
        print(f"  {service}: {status}")
        if "❌" in status:
            all_healthy = False

    print()
    sys.exit(0 if all_healthy else 1)


if __name__ == "__main__":
    asyncio.run(main())
