# import logging

# from graph.state import AgentState, ImageSlot, SchemaMarkup

# logger = logging.getLogger(__name__)

# _NON_CRITICAL_NODES = {"image_selector", "citation_formatter", "schema_generator"}
# _MAX_RETRIES = 3


# def _insert_image_placeholder(state: AgentState) -> dict:
#     return {
#         "image_manifest": [
#             ImageSlot(
#                 slot_id="img_placeholder",
#                 description="Image placeholder — upload manually before publishing.",
#                 alt_text="Pickleball player in action | PaddleAurum",
#                 url=None,
#                 credit=None,
#             )
#         ]
#     }


# def _insert_citation_placeholder(state: AgentState) -> dict:
#     return {
#         "formatted_article": state.get("draft_article") or "",
#     }


# def _insert_schema_placeholder(state: AgentState) -> dict:
#     return {
#         "schema_markup": SchemaMarkup(
#             article="{}",
#             faq=None,
#             how_to=None,
#         )
#     }


# _PLACEHOLDER_HANDLERS = {
#     "image_selector":    _insert_image_placeholder,
#     "citation_formatter": _insert_citation_placeholder,
#     "schema_generator":  _insert_schema_placeholder,
# }


# async def error_recovery_node(state: AgentState) -> dict:
#     error_node:  str = state.get("error_node") or "unknown"
#     error_msg:   str = state.get("error") or "Unknown error"
#     retry_count: int = state.get("retry_count", 0)

#     logger.warning(
#         "Error recovery: node='%s' retry=%d/%d error='%s' (session=%s)",
#         error_node, retry_count, _MAX_RETRIES, error_msg, state.get("session_id"),
#     )

#     new_retry_count = retry_count + 1

#     # Still within retry budget — clear error and let router retry the failed node
#     if new_retry_count <= _MAX_RETRIES:
#         return {
#             "retry_count": new_retry_count,
#             "error":       None,
#             "error_node":  error_node,  # kept so router knows where to send back
#         }

#     # Retries exhausted
#     if error_node in _NON_CRITICAL_NODES:
#         logger.warning(
#             "Non-critical node '%s' exhausted retries — inserting placeholder and continuing.",
#             error_node,
#         )
#         handler = _PLACEHOLDER_HANDLERS.get(error_node)
#         placeholder_update = handler(state) if handler else {}

#         return {
#             **placeholder_update,
#             "retry_count": 0,          # reset for any future errors
#             "error":       None,
#             "error_node":  error_node,  # router uses this to skip forward
#         }

#     # Critical node failure — flag for human and surface the error
#     logger.error(
#         "Critical node '%s' exhausted all retries — escalating to human review (session=%s).",
#         error_node, state.get("session_id"),
#     )

#     existing_suggestions = list(state.get("seo_suggestions") or [])
#     escalation_note = (
#         f"[PIPELINE ERROR — {error_node}]: {error_msg}. "
#         "Automatic recovery failed. Please review and fix before publishing."
#     )

#     return {
#         "seo_suggestions":       [escalation_note] + existing_suggestions,
#         "human_review_requested": True,
#         "retry_count":            0,
#         "error":                  None,
#         "error_node":             error_node,
#     }
















# @#####################################################################################














import logging

from config.settings import MAX_ITERATIONS
from graph.state import AgentState, ImageSlot, SchemaMarkup

logger = logging.getLogger(__name__)

_NON_CRITICAL_NODES = {"image_selector", "citation_formatter", "schema_generator"}


def _insert_image_placeholder(state: AgentState) -> dict:
    return {
        "image_manifest": [
            ImageSlot(
                slot_id="img_placeholder",
                description="Image placeholder — upload manually before publishing.",
                alt_text="Pickleball player in action | PaddleAurum",
                url=None,
                credit=None,
            )
        ]
    }


def _insert_citation_placeholder(state: AgentState) -> dict:
    return {
        "formatted_article": state.get("draft_article") or "",
    }


def _insert_schema_placeholder(state: AgentState) -> dict:
    return {
        "schema_markup": SchemaMarkup(
            article="{}",
            faq=None,
            how_to=None,
        )
    }


_PLACEHOLDER_HANDLERS = {
    "image_selector":    _insert_image_placeholder,
    "citation_formatter": _insert_citation_placeholder,
    "schema_generator":  _insert_schema_placeholder,
}


async def error_recovery_node(state: AgentState) -> dict:
    error_node:  str = state.get("error_node") or "unknown"
    error_msg:   str = state.get("error") or "Unknown error"
    retry_count: int = state.get("retry_count", 0)

    logger.warning(
        "Error recovery: node='%s' retry=%d/%d error='%s' (session=%s)",
        error_node, retry_count, MAX_ITERATIONS, error_msg, state.get("session_id"),
    )

    new_retry_count = retry_count + 1

    # Still within retry budget — clear error and let router retry the failed node
    if new_retry_count <= MAX_ITERATIONS:
        return {
            "retry_count": new_retry_count,
            "error":       None,
            "error_node":  error_node,  # kept so router knows where to send back
        }

    # Retries exhausted
    if error_node in _NON_CRITICAL_NODES:
        logger.warning(
            "Non-critical node '%s' exhausted retries — inserting placeholder and continuing.",
            error_node,
        )
        handler = _PLACEHOLDER_HANDLERS.get(error_node)
        placeholder_update = handler(state) if handler else {}

        return {
            **placeholder_update,
            "retry_count": 0,          # reset for any future errors
            "error":       None,
            "error_node":  error_node,  # router uses this to skip forward
        }

    # Critical node failure — flag for human and surface the error
    logger.error(
        "Critical node '%s' exhausted all retries — escalating to human review (session=%s).",
        error_node, state.get("session_id"),
    )

    existing_suggestions = list(state.get("seo_suggestions") or [])
    escalation_note = (
        f"[PIPELINE ERROR — {error_node}]: {error_msg}. "
        "Automatic recovery failed. Please review and fix before publishing."
    )

    return {
        "seo_suggestions":       [escalation_note] + existing_suggestions,
        "human_review_requested": True,
        "retry_count":            0,
        "error":                  None,
        "error_node":             error_node,
    }