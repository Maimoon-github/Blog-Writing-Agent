import re
import uuid
from graph.state import AgentState, Tone

_VALID_TONES = {t.value for t in Tone}
_MIN_WORD_COUNT = 300
_MAX_WORD_COUNT = 5000


async def input_validator_node(state: AgentState) -> dict:
    topic: str = (state.get("topic") or "").strip()
    if not topic:
        return {"error": "Topic is required and cannot be empty.", "error_node": "input_validator"}

    if len(topic) < 5:
        return {"error": f"Topic too short: '{topic}'. Provide a meaningful topic.", "error_node": "input_validator"}

    tone_raw = state.get("tone")
    if tone_raw is None:
        tone = Tone.COACH
    elif isinstance(tone_raw, Tone):
        tone = tone_raw
    elif str(tone_raw) in _VALID_TONES:
        tone = Tone(str(tone_raw))
    else:
        tone = Tone.COACH

    word_count_goal = state.get("word_count_goal") or 1500
    try:
        word_count_goal = int(word_count_goal)
    except (TypeError, ValueError):
        word_count_goal = 1500
    word_count_goal = max(_MIN_WORD_COUNT, min(_MAX_WORD_COUNT, word_count_goal))

    max_iterations = state.get("max_iterations") or 3
    try:
        max_iterations = int(max_iterations)
    except (TypeError, ValueError):
        max_iterations = 3
    max_iterations = max(1, min(5, max_iterations))

    session_id = state.get("session_id") or uuid.uuid4().hex

    target_keyword = state.get("target_keyword")
    if target_keyword:
        target_keyword = target_keyword.strip().lower()
        target_keyword = re.sub(r"\s+", " ", target_keyword) or None

    return {
        "topic":           topic,
        "tone":            tone,
        "word_count_goal": word_count_goal,
        "max_iterations":  max_iterations,
        "session_id":      session_id,
        "target_keyword":  target_keyword,
        "error":           None,
        "error_node":      None,
    }