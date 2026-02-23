# #!/usr/bin/env python
# """Scheduler for automated blog generation runs (APScheduler)."""

# import asyncio
# import os
# import logging
# from datetime import datetime, timedelta

# from apscheduler.schedulers.asyncio import AsyncIOScheduler
# from apscheduler.triggers.cron import CronTrigger
# from dotenv import load_dotenv

# # Import the main pipeline runner
# from main import run_pipeline
# from state.schema import AgentState

# load_dotenv()

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
# )
# logger = logging.getLogger(__name__)

# # Example article topics and their configurations (could be stored in DB)
# # In production, these would be fetched from a queue or database.
# SCHEDULED_ARTICLES = [
#     {
#         "topic": "Pickleball kitchen rules for beginners",
#         "tone": "coach",
#         "word_count_goal": 1500,
#         "cron": "0 9 * * 1",  # Every Monday at 9:00 AM
#     },
#     {
#         "topic": "Best pickleball paddles under $100",
#         "tone": "expert",
#         "word_count_goal": 1800,
#         "cron": "0 10 * * 3",  # Every Wednesday at 10:00 AM
#     },
#     {
#         "topic": "How to improve your third-shot drop",
#         "tone": "coach",
#         "word_count_goal": 1200,
#         "cron": "0 8 * * 5",  # Every Friday at 8:00 AM
#     },
# ]


# async def scheduled_run(article_config: dict):
#     """Execute the pipeline for a given article configuration."""
#     topic = article_config["topic"]
#     logger.info(f"Starting scheduled run for topic: {topic}")

#     # Build initial state
#     initial_state: AgentState = {
#         "topic": topic,
#         "target_keyword": None,  # will be determined by planner
#         "tone": article_config.get("tone", "coach"),
#         "word_count_goal": article_config.get("word_count_goal", 1500),
#         "session_id": f"scheduled_{datetime.utcnow().isoformat()}",
#         "needs_research": False,
#         "sub_queries": [],
#         "research_snippets": [],
#         "research_sources": [],
#         "keyword_map": None,
#         "content_outline": None,
#         "faq_candidates": [],
#         "internal_link_placeholders": [],
#         "draft_article": None,
#         "revision_iteration": 0,
#         "max_iterations": 3,
#         "seo_score": None,
#         "seo_issues": [],
#         "seo_suggestions": [],
#         "image_manifest": [],
#         "formatted_article": None,
#         "schema_markup": None,
#         "title_tag": None,
#         "meta_description": None,
#         "url_slug": None,
#         "final_output": None,
#         "approved": False,
#         "human_review_requested": False,
#         "error": None,
#         "error_node": None,
#         "retry_count": 0,
#     }

#     try:
#         final_state = await run_pipeline(initial_state)
#         if final_state.get("final_output"):
#             logger.info(f"Successfully generated article: {topic}")
#             # Here you could trigger notifications, save to DB, etc.
#         else:
#             logger.warning(f"Pipeline completed but no final output for: {topic}")
#     except Exception as e:
#         logger.exception(f"Error during scheduled run for {topic}: {e}")


# def main():
#     """Set up and start the scheduler."""
#     scheduler = AsyncIOScheduler()

#     for article in SCHEDULED_ARTICLES:
#         trigger = CronTrigger.from_crontab(article["cron"])
#         scheduler.add_job(
#             scheduled_run,
#             trigger=trigger,
#             args=[article],
#             id=f"job_{article['topic'].replace(' ', '_')}",
#             replace_existing=True,
#         )
#         logger.info(f"Scheduled '{article['topic']}' with cron: {article['cron']}")

#     scheduler.start()
#     logger.info("Scheduler started. Press Ctrl+C to exit.")

#     try:
#         asyncio.get_event_loop().run_forever()
#     except KeyboardInterrupt:
#         logger.info("Shutting down scheduler...")
#         scheduler.shutdown()


# if __name__ == "__main__":
#     main()















# ````````````````````````````````````````````````````````````````````````````````````````











#!/usr/bin/env python
"""Scheduler for automated blog generation runs (APScheduler)."""

import asyncio
import os
import logging
import yaml
from datetime import datetime, timedelta
from typing import List, Dict, Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

# Import the main pipeline runner and state helpers
from main import run_pipeline
from graph.state import AgentState, make_initial_state, Tone

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default path for scheduled articles configuration
DEFAULT_CONFIG_PATH = os.getenv("SCHEDULER_CONFIG_PATH", "config/scheduled_articles.yaml")


def load_scheduled_articles(config_path: str = DEFAULT_CONFIG_PATH) -> List[Dict[str, Any]]:
    """Load article configurations from a YAML file."""
    if not os.path.exists(config_path):
        logger.warning(f"Scheduler config file not found: {config_path}. No articles scheduled.")
        return []

    try:
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        if not isinstance(data, list):
            logger.error(f"Invalid format in {config_path}: expected a list of articles.")
            return []
        logger.info(f"Loaded {len(data)} scheduled articles from {config_path}")
        return data
    except Exception as e:
        logger.exception(f"Failed to load scheduler config from {config_path}: {e}")
        return []


async def scheduled_run(article_config: dict):
    """Execute the pipeline for a given article configuration."""
    topic = article_config["topic"]
    logger.info(f"Starting scheduled run for topic: {topic}")

    # Build initial state using make_initial_state
    try:
        tone_str = article_config.get("tone", "coach")
        tone = Tone(tone_str)  # convert to enum
    except ValueError:
        logger.warning(f"Invalid tone '{tone_str}' for {topic}, defaulting to 'coach'")
        tone = Tone.COACH

    initial_state = make_initial_state(
        topic=topic,
        target_keyword=article_config.get("target_keyword"),
        tone=tone,
        word_count_goal=article_config.get("word_count_goal", 1500),
    )
    # Add session_id for scheduled run
    initial_state["session_id"] = f"scheduled_{datetime.utcnow().isoformat()}"

    try:
        final_state = await run_pipeline(initial_state)
        if final_state.get("final_output"):
            logger.info(f"Successfully generated article: {topic}")
            # Here you could trigger notifications, save to DB, etc.
        else:
            logger.warning(f"Pipeline completed but no final output for: {topic}")
    except Exception as e:
        logger.exception(f"Error during scheduled run for {topic}: {e}")


def main():
    """Set up and start the scheduler."""
    scheduler = AsyncIOScheduler()

    scheduled_articles = load_scheduled_articles()
    if not scheduled_articles:
        logger.info("No scheduled articles to run. Scheduler will idle.")
    else:
        for article in scheduled_articles:
            # Validate required fields
            if "topic" not in article or "cron" not in article:
                logger.error(f"Skipping invalid article config (missing 'topic' or 'cron'): {article}")
                continue
            try:
                trigger = CronTrigger.from_crontab(article["cron"])
            except Exception as e:
                logger.error(f"Invalid cron expression for '{article.get('topic')}': {article['cron']} - {e}")
                continue

            scheduler.add_job(
                scheduled_run,
                trigger=trigger,
                args=[article],
                id=f"job_{article['topic'].replace(' ', '_')}",
                replace_existing=True,
            )
            logger.info(f"Scheduled '{article['topic']}' with cron: {article['cron']}")

    scheduler.start()
    logger.info("Scheduler started. Press Ctrl+C to exit.")

    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down scheduler...")
        scheduler.shutdown()


if __name__ == "__main__":
    main()