# import base64
# import logging

# import aiohttp

# from graph.state import AgentState

# logger = logging.getLogger(__name__)

# _PUBLISH_TIMEOUT = aiohttp.ClientTimeout(total=30)


# def _basic_auth_header(user: str, app_password: str) -> str:
#     token = base64.b64encode(f"{user}:{app_password}".encode()).decode()
#     return f"Basic {token}"


# def _build_post_body(state: AgentState) -> dict:
#     final_output = state.get("final_output") or {}
#     schema_markup = state.get("schema_markup") or {}

#     # Embed JSON-LD schema blocks as a <script> block prepended to content
#     schema_scripts = ""
#     for key in ("article", "faq", "how_to"):
#         json_ld = schema_markup.get(key)
#         if json_ld:
#             schema_scripts += f'<script type="application/ld+json">{json_ld}</script>\n'

#     content_html = final_output.get("markdown", state.get("draft_article", ""))

#     return {
#         "title":   state.get("title_tag", state.get("topic", "")),
#         "content": schema_scripts + content_html,
#         "slug":    state.get("url_slug", ""),
#         "status":  "publish",
#         "excerpt": state.get("meta_description", ""),
#         "meta": {
#             "_yoast_wpseo_title":          state.get("title_tag", ""),
#             "_yoast_wpseo_metadesc":       state.get("meta_description", ""),
#             "_yoast_wpseo_focuskw":        (state.get("keyword_map") or {}).get("primary", ""),
#         },
#     }


# async def publish_node(
#     state: AgentState,
#     *,
#     wp_url: str,
#     wp_user: str,
#     wp_password: str,
# ) -> dict:
#     try:
#         api_endpoint = f"{wp_url.rstrip('/')}/wp-json/wp/v2/posts"
#         headers = {
#             "Authorization": _basic_auth_header(wp_user, wp_password),
#             "Content-Type":  "application/json",
#         }
#         body = _build_post_body(state)

#         async with aiohttp.ClientSession() as session:
#             async with session.post(
#                 api_endpoint,
#                 json=body,
#                 headers=headers,
#                 timeout=_PUBLISH_TIMEOUT,
#             ) as resp:
#                 response_data = await resp.json()

#                 if resp.status not in (200, 201):
#                     error_msg = response_data.get("message", f"HTTP {resp.status}")
#                     logger.error("WordPress publish failed: %s", error_msg)
#                     return {"error": f"WordPress publish failed: {error_msg}", "error_node": "publish"}

#                 post_id  = response_data.get("id")
#                 post_url = response_data.get("link", "")

#                 logger.info(
#                     "Published successfully: post_id=%s url=%s (session=%s)",
#                     post_id, post_url, state.get("session_id"),
#                 )

#                 return {
#                     "error":      None,
#                     "error_node": None,
#                 }

#     except aiohttp.ClientError as exc:
#         logger.exception("WordPress API connection failed.")
#         return {"error": f"WordPress connection error: {exc}", "error_node": "publish"}
#     except Exception as exc:
#         logger.exception("Publish node failed.")
#         return {"error": str(exc), "error_node": "publish"}

















# #@###########################################################################################


















import asyncio
import base64
import logging

import aiohttp

from graph.state import AgentState

logger = logging.getLogger(__name__)

_PUBLISH_TIMEOUT = aiohttp.ClientTimeout(total=30)
_MAX_RETRIES = 3
_RETRY_DELAYS = [1, 2, 4]  # seconds


def _basic_auth_header(user: str, app_password: str) -> str:
    token = base64.b64encode(f"{user}:{app_password}".encode()).decode()
    return f"Basic {token}"


def _build_post_body(state: AgentState) -> dict:
    final_output = state.get("final_output") or {}
    schema_markup = state.get("schema_markup") or {}

    # Embed JSON-LD schema blocks as a <script> block prepended to content
    schema_scripts = ""
    for key in ("article", "faq", "how_to"):
        json_ld = schema_markup.get(key)
        if json_ld:
            schema_scripts += f'<script type="application/ld+json">{json_ld}</script>\n'

    content_html = final_output.get("markdown", state.get("draft_article", ""))

    return {
        "title":   state.get("title_tag", state.get("topic", "")),
        "content": schema_scripts + content_html,
        "slug":    state.get("url_slug", ""),
        "status":  state.get("publish_status", "publish"),  # allow draft, future, etc.
        "excerpt": state.get("meta_description", ""),
        "meta": {
            "_yoast_wpseo_title":          state.get("title_tag", ""),
            "_yoast_wpseo_metadesc":       state.get("meta_description", ""),
            "_yoast_wpseo_focuskw":        (state.get("keyword_map") or {}).get("primary", ""),
        },
    }


async def publish_node(
    state: AgentState,
    *,
    wp_url: str,
    wp_user: str,
    wp_password: str,
) -> dict:
    try:
        api_endpoint = f"{wp_url.rstrip('/')}/wp-json/wp/v2/posts"
        headers = {
            "Authorization": _basic_auth_header(wp_user, wp_password),
            "Content-Type":  "application/json",
        }
        body = _build_post_body(state)

        last_exception = None
        for attempt in range(_MAX_RETRIES):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        api_endpoint,
                        json=body,
                        headers=headers,
                        timeout=_PUBLISH_TIMEOUT,
                    ) as resp:
                        response_data = await resp.json()

                        # Success (2xx)
                        if resp.status in (200, 201):
                            post_id  = response_data.get("id")
                            post_url = response_data.get("link", "")
                            logger.info(
                                "Published successfully: post_id=%s url=%s (session=%s, attempt=%d)",
                                post_id, post_url, state.get("session_id"), attempt + 1,
                            )
                            return {
                                "error":      None,
                                "error_node": None,
                            }

                        # Non‑retryable client error (4xx)
                        if 400 <= resp.status < 500:
                            error_msg = response_data.get("message", f"HTTP {resp.status}")
                            logger.error("WordPress publish failed (4xx): %s", error_msg)
                            return {"error": f"WordPress publish failed: {error_msg}", "error_node": "publish"}

                        # Server error or other (5xx) – retry
                        if attempt < _MAX_RETRIES - 1:
                            wait = _RETRY_DELAYS[attempt]
                            logger.warning(
                                "WordPress publish attempt %d failed (status %d). Retrying in %ds...",
                                attempt + 1, resp.status, wait,
                            )
                            await asyncio.sleep(wait)
                            continue
                        else:
                            error_msg = response_data.get("message", f"HTTP {resp.status}")
                            logger.error("WordPress publish failed after %d retries: %s", _MAX_RETRIES, error_msg)
                            return {"error": f"WordPress publish failed: {error_msg}", "error_node": "publish"}

            except aiohttp.ClientError as exc:
                last_exception = exc
                if attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_DELAYS[attempt]
                    logger.warning(
                        "WordPress connection error on attempt %d: %s. Retrying in %ds...",
                        attempt + 1, exc, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.exception("WordPress API connection failed after %d retries.", _MAX_RETRIES)
                    return {"error": f"WordPress connection error: {exc}", "error_node": "publish"}

        # Should never reach here, but just in case
        return {"error": "Unexpected error in publish retry loop", "error_node": "publish"}

    except Exception as exc:
        logger.exception("Publish node failed.")
        return {"error": str(exc), "error_node": "publish"}