# import asyncio
# import logging
# import re
# from typing import List, Optional

# import aiohttp

# from graph.state import AgentState, ImageSlot

# logger = logging.getLogger(__name__)

# _UNSPLASH_SEARCH  = "https://api.unsplash.com/search/photos"
# _PEXELS_SEARCH    = "https://api.pexels.com/v1/search"
# _REQUEST_TIMEOUT  = aiohttp.ClientTimeout(total=10)


# def _extract_image_slots(draft: str) -> List[dict]:
#     matches = re.finditer(r"\[IMAGE:\s*([^\]]+)\]", draft)
#     slots = []
#     for i, m in enumerate(matches, 1):
#         slots.append({"slot_id": f"img_{i:03d}", "alt_text_hint": m.group(1).strip()})
#     return slots


# async def _fetch_unsplash(
#     session: aiohttp.ClientSession,
#     query: str,
#     api_key: str,
# ) -> Optional[dict]:
#     try:
#         async with session.get(
#             _UNSPLASH_SEARCH,
#             headers={"Authorization": f"Client-ID {api_key}"},
#             params={"query": query, "per_page": 1, "orientation": "landscape"},
#             timeout=_REQUEST_TIMEOUT,
#         ) as resp:
#             if resp.status != 200:
#                 return None
#             data = await resp.json()
#             results = data.get("results", [])
#             if not results:
#                 return None
#             photo = results[0]
#             return {
#                 "url":    photo["urls"]["regular"],
#                 "credit": f"{photo['user']['name']} via Unsplash",
#             }
#     except Exception:
#         return None


# async def _fetch_pexels(
#     session: aiohttp.ClientSession,
#     query: str,
#     api_key: str,
# ) -> Optional[dict]:
#     try:
#         async with session.get(
#             _PEXELS_SEARCH,
#             headers={"Authorization": api_key},
#             params={"query": query, "per_page": 1, "orientation": "landscape"},
#             timeout=_REQUEST_TIMEOUT,
#         ) as resp:
#             if resp.status != 200:
#                 return None
#             data = await resp.json()
#             photos = data.get("photos", [])
#             if not photos:
#                 return None
#             photo = photos[0]
#             photographer = photo.get("photographer", "Unknown")
#             return {
#                 "url":    photo["src"]["large"],
#                 "credit": f"{photographer} via Pexels",
#             }
#     except Exception:
#         return None


# async def _resolve_slot(
#     session: aiohttp.ClientSession,
#     slot_id: str,
#     alt_text_hint: str,
#     primary_kw: str,
#     unsplash_key: str,
#     pexels_key: str,
# ) -> ImageSlot:
#     query = f"pickleball {alt_text_hint}" if "pickleball" not in alt_text_hint.lower() else alt_text_hint
#     alt_text = f"{alt_text_hint} — pickleball coaching guide | PaddleAurum"

#     photo = await _fetch_unsplash(session, query, unsplash_key)
#     if not photo:
#         photo = await _fetch_pexels(session, query, pexels_key)

#     return ImageSlot(
#         slot_id=slot_id,
#         description=alt_text_hint,
#         alt_text=alt_text,
#         url=photo["url"] if photo else None,
#         credit=photo["credit"] if photo else None,
#     )


# async def image_selector_node(
#     state: AgentState,
#     *,
#     unsplash_key: str,
#     pexels_key: str,
# ) -> dict:
#     try:
#         draft: str = state.get("draft_article") or ""
#         keyword_map = state.get("keyword_map") or {}
#         primary_kw: str = keyword_map.get("primary", "pickleball")

#         raw_slots = _extract_image_slots(draft)
#         if not raw_slots:
#             logger.warning("No [IMAGE: ...] placeholders found in draft.")
#             return {"image_manifest": [], "error": None, "error_node": None}

#         async with aiohttp.ClientSession() as session:
#             tasks = [
#                 _resolve_slot(session, s["slot_id"], s["alt_text_hint"], primary_kw, unsplash_key, pexels_key)
#                 for s in raw_slots
#             ]
#             image_manifest: List[ImageSlot] = await asyncio.gather(*tasks)

#         resolved = sum(1 for img in image_manifest if img["url"])
#         logger.info("Image selector: %d slots → %d resolved, %d placeholders",
#                     len(raw_slots), resolved, len(raw_slots) - resolved)

#         return {
#             "image_manifest": list(image_manifest),
#             "error":          None,
#             "error_node":     None,
#         }

#     except Exception as exc:
#         logger.exception("Image selector failed.")
#         return {"error": str(exc), "error_node": "image_selector"}

















# @######################################################################













import asyncio
import logging
import re
from typing import List, Optional

import aiohttp

from graph.state import AgentState, ImageSlot

logger = logging.getLogger(__name__)

_UNSPLASH_SEARCH  = "https://api.unsplash.com/search/photos"
_PEXELS_SEARCH    = "https://api.pexels.com/v1/search"
_REQUEST_TIMEOUT  = aiohttp.ClientTimeout(total=10)


def _extract_image_slots(draft: str) -> List[dict]:
    matches = re.finditer(r"\[IMAGE:\s*([^\]]+)\]", draft)
    slots = []
    for i, m in enumerate(matches, 1):
        slots.append({"slot_id": f"img_{i:03d}", "alt_text_hint": m.group(1).strip()})
    return slots


async def _fetch_unsplash(
    session: aiohttp.ClientSession,
    query: str,
    api_key: str,
) -> Optional[dict]:
    try:
        async with session.get(
            _UNSPLASH_SEARCH,
            headers={"Authorization": f"Client-ID {api_key}"},
            params={"query": query, "per_page": 1, "orientation": "landscape"},
            timeout=_REQUEST_TIMEOUT,
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            results = data.get("results", [])
            if not results:
                return None
            photo = results[0]
            return {
                "url":    photo["urls"]["regular"],
                "credit": f"{photo['user']['name']} via Unsplash",
            }
    except Exception:
        return None


async def _fetch_pexels(
    session: aiohttp.ClientSession,
    query: str,
    api_key: str,
) -> Optional[dict]:
    try:
        async with session.get(
            _PEXELS_SEARCH,
            headers={"Authorization": api_key},
            params={"query": query, "per_page": 1, "orientation": "landscape"},
            timeout=_REQUEST_TIMEOUT,
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            photos = data.get("photos", [])
            if not photos:
                return None
            photo = photos[0]
            photographer = photo.get("photographer", "Unknown")
            return {
                "url":    photo["src"]["large"],
                "credit": f"{photographer} via Pexels",
            }
    except Exception:
        return None


async def _resolve_slot(
    session: aiohttp.ClientSession,
    slot_id: str,
    alt_text_hint: str,
    primary_kw: str,
    unsplash_key: str,
    pexels_key: str,
) -> ImageSlot:
    query = f"pickleball {alt_text_hint}" if "pickleball" not in alt_text_hint.lower() else alt_text_hint
    alt_text = f"{alt_text_hint} — pickleball coaching guide | PaddleAurum"

    photo = await _fetch_unsplash(session, query, unsplash_key)
    if not photo:
        photo = await _fetch_pexels(session, query, pexels_key)

    return ImageSlot(
        slot_id=slot_id,
        description=alt_text_hint,
        alt_text=alt_text,
        url=photo["url"] if photo else None,
        credit=photo["credit"] if photo else None,
    )


async def image_selector_node(
    state: AgentState,
    *,
    unsplash_key: str,
    pexels_key: str,
) -> dict:
    try:
        draft: str = state.get("draft_article") or ""
        keyword_map = state.get("keyword_map") or {}
        primary_kw: str = keyword_map.get("primary", "pickleball")

        raw_slots = _extract_image_slots(draft)
        if not raw_slots:
            logger.warning("No [IMAGE: ...] placeholders found in draft.")
            return {"image_manifest": [], "error": None, "error_node": None}

        async with aiohttp.ClientSession() as session:
            tasks = [
                _resolve_slot(session, s["slot_id"], s["alt_text_hint"], primary_kw, unsplash_key, pexels_key)
                for s in raw_slots
            ]
            image_manifest: List[ImageSlot] = await asyncio.gather(*tasks)

        resolved = sum(1 for img in image_manifest if img["url"])
        logger.info("Image selector: %d slots → %d resolved, %d placeholders",
                    len(raw_slots), resolved, len(raw_slots) - resolved)

        return {
            "image_manifest": list(image_manifest),
            "error":          None,
            "error_node":     None,
        }

    except Exception as exc:
        logger.exception("Image selector failed.")
        return {"error": str(exc), "error_node": "image_selector"}