"""
Typer CLI for local development, debugging, and quick blog generation.
"""

import asyncio
from typing import Optional

import typer

app = typer.Typer(
    name="blog-agent",
    help="Agentic AI Blog Writing System — CLI",
    add_completion=False,
)


@app.command()
def generate(
    topic: str = typer.Argument(..., help="Blog topic to write about"),
    audience: str = typer.Option("technical professionals", "--audience", "-a"),
    tone: str = typer.Option("informative yet conversational", "--tone", "-t"),
    words: int = typer.Option(1500, "--words", "-w", min=500, max=5000),
    style: str = typer.Option("AP", "--style", "-s"),
    no_images: bool = typer.Option(False, "--no-images"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save to file"),
) -> None:
    """Generate a blog post on the given topic (uses full multi-agent pipeline)."""
    from blog_agent_system.utils.logging import setup_logging

    setup_logging()

    typer.echo(f"🚀 Starting blog generation: {topic}")
    typer.echo(f"   Audience: {audience} | Tone: {tone} | Words: {words} | Images: {not no_images}")

    async def _run():
        from blog_agent_system.core.orchestrator import BlogOrchestrator

        orchestrator = BlogOrchestrator()
        await orchestrator.initialize()
        result = await orchestrator.generate_blog(
            topic=topic,
            target_audience=audience,
            tone=tone,
            word_count_target=words,
            include_images=not no_images,
            style_guide=style,
        )
        return result

    result = asyncio.run(_run())

    if result.get("status") == "complete":
        blog_content = result.get("final_blog") or result.get("draft", "")
        score = result.get("quality_score", 0.0)
        revisions = result.get("revision_count", 0)

        typer.echo(f"\n✅ Generation complete! Quality: {score:.2f} | Revisions: {revisions}")

        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(blog_content)
            typer.echo(f"📄 Saved to: {output}")
        else:
            typer.echo("\n" + "=" * 80)
            typer.echo(blog_content)
            typer.echo("=" * 80)
    else:
        typer.echo(f"\n❌ Generation failed: {result.get('error', 'Unknown error')}", err=True)
        raise typer.Exit(code=1)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
    port: int = typer.Option(8080, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload", "-r"),
) -> None:
    """Start the FastAPI server."""
    import uvicorn

    typer.echo(f"🌐 Starting server at http://{host}:{port}")
    uvicorn.run(
        "blog_agent_system.main:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def health() -> None:
    """Run health checks against all infrastructure dependencies."""
    asyncio.run(_health_check())


async def _health_check() -> None:
    """Internal health check for Postgres, Redis, ChromaDB."""
    from blog_agent_system.utils.logging import get_logger
    from blog_agent_system.config.settings import settings

    logger = get_logger(__name__)
    checks = {}

    # PostgreSQL
    try:
        from blog_agent_system.persistence.database import async_engine
        from sqlalchemy import text

        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["postgres"] = "✅ healthy"
    except Exception as e:
        checks["postgres"] = f"❌ {type(e).__name__}"

    # Redis
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(settings.redis_url)
        await r.ping()
        await r.aclose()
        checks["redis"] = "✅ healthy"
    except Exception as e:
        checks["redis"] = f"❌ {type(e).__name__}"

    # ChromaDB
    try:
        import chromadb

        client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        client.heartbeat()
        checks["chromadb"] = "✅ healthy"
    except Exception as e:
        checks["chromadb"] = f"❌ {type(e).__name__}"

    for service, status in checks.items():
        typer.echo(f"  {service}: {status}")
        logger.info("health.check", service=service, status=status)


if __name__ == "__main__":
    app()