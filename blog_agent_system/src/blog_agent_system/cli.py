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
    audience: str = typer.Option("technical professionals", "--audience", "-a", help="Target audience"),
    tone: str = typer.Option("informative yet conversational", "--tone", "-t", help="Writing tone"),
    words: int = typer.Option(1500, "--words", "-w", help="Target word count"),
    style: str = typer.Option("AP", "--style", "-s", help="Style guide (AP, Chicago, MLA)"),
    no_images: bool = typer.Option(False, "--no-images", help="Skip image generation"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
) -> None:
    """Generate a blog post on the given topic."""
    from blog_agent_system.utils.logging import setup_logging

    setup_logging()

    typer.echo(f"🚀 Generating blog post: {topic}")
    typer.echo(f"   Audience: {audience} | Tone: {tone} | Words: {words}")

    async def _run():
        from blog_agent_system.core.orchestrator import BlogOrchestrator

        orchestrator = BlogOrchestrator()
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
        score = result.get("quality_score", 0)
        revisions = result.get("revision_count", 0)

        typer.echo(f"\n✅ Blog generated! Quality: {score:.2f} | Revisions: {revisions}")

        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(blog_content)
            typer.echo(f"📄 Saved to: {output}")
        else:
            typer.echo("\n" + "=" * 60)
            typer.echo(blog_content)
            typer.echo("=" * 60)
    else:
        typer.echo(f"\n❌ Generation failed: {result.get('error', 'Unknown error')}", err=True)
        raise typer.Exit(code=1)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8080, "--port", "-p", help="Port to listen on"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
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
    """Check if all dependencies are reachable."""
    asyncio.run(_health_check())


async def _health_check() -> None:
    """Run health checks against all infrastructure dependencies."""
    checks = {}

    # PostgreSQL
    try:
        from blog_agent_system.persistence.database import async_engine
        from sqlalchemy import text

        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["postgres"] = "✅ healthy"
    except Exception as e:
        checks["postgres"] = f"❌ {e}"

    # Redis
    try:
        import redis.asyncio as aioredis
        from blog_agent_system.config.settings import settings

        r = aioredis.from_url(settings.redis_url)
        await r.ping()
        await r.aclose()
        checks["redis"] = "✅ healthy"
    except Exception as e:
        checks["redis"] = f"❌ {e}"

    # ChromaDB
    try:
        import chromadb
        from blog_agent_system.config.settings import settings

        client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        client.heartbeat()
        checks["chromadb"] = "✅ healthy"
    except Exception as e:
        checks["chromadb"] = f"❌ {e}"

    for service, status in checks.items():
        typer.echo(f"  {service}: {status}")


if __name__ == "__main__":
    app()
