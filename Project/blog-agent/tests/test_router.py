"""Unit tests for the router node (graph.router.router_node)."""

import pytest
from pytest_mock import MockerFixture

from graph.router import router_node


@pytest.fixture
def base_state() -> dict:
    """Return a minimal GraphState dictionary for testing."""
    return {
        "topic": "",
        "research_required": False,
        "blog_plan": None,
        "research_results": [],
        "section_drafts": [],
        "generated_images": [],
        "citation_registry": {},
        "final_blog_md": "",
        "final_blog_html": "",
        "run_id": "test",
        "error": None,
    }


@pytest.fixture(autouse=True)
def mock_prompt_file(mocker: MockerFixture) -> None:
    """
    Prevent the router from actually reading the prompt file from disk.

    Patches the read_text method of the module-level PROMPT_PATH constant
    so that it returns a dummy prompt string.
    """
    mocker.patch(
        "graph.router.PROMPT_PATH.read_text",
        return_value="You are a router. Analyze the topic and return JSON.",
    )


def setup_mock_llm(mocker: MockerFixture, response_content: str):
    """
    Helper to mock ChatOllama and set the invoke response.

    Returns the mocked ChatOllama class for optional further assertions.
    """
    mock_chat_ollama = mocker.patch("graph.router.ChatOllama")
    mock_instance = mocker.MagicMock()
    mock_instance.invoke.return_value.content = response_content
    mock_chat_ollama.return_value = mock_instance
    return mock_chat_ollama


def test_research_required_true(mocker: MockerFixture, base_state: dict) -> None:
    """Router should return research_required=True for time‑sensitive topics."""
    setup_mock_llm(
        mocker,
        '{"research_required": true, "safe": true}',
    )
    base_state["topic"] = "latest AI trends 2025"
    result = router_node(base_state)

    assert result["research_required"] is True
    assert "error" not in result


def test_research_required_false(mocker: MockerFixture, base_state: dict) -> None:
    """Router should return research_required=False for evergreen topics."""
    setup_mock_llm(
        mocker,
        '{"research_required": false, "safe": true}',
    )
    base_state["topic"] = "History of ancient Rome"
    result = router_node(base_state)

    assert result["research_required"] is False
    assert "error" not in result


def test_unsafe_topic(mocker: MockerFixture, base_state: dict) -> None:
    """Router should return an error when the topic is flagged unsafe."""
    setup_mock_llm(
        mocker,
        '{"research_required": false, "safe": false}',
    )
    base_state["topic"] = "harmful content topic"
    result = router_node(base_state)

    assert "error" in result
    assert "rejected by safety filter" in result["error"]
    # When safety rejects, research_required is forced to False
    assert result["research_required"] is False


def test_malformed_json_fallback(mocker: MockerFixture, base_state: dict) -> None:
    """Router should extract JSON from a response with extra text."""
    setup_mock_llm(
        mocker,
        'Here is the result: {"research_required": true, "safe": true} — done.',
    )
    base_state["topic"] = "any topic"
    result = router_node(base_state)

    assert result["research_required"] is True
    assert "error" not in result


def test_llm_invocation_error(mocker: MockerFixture, base_state: dict) -> None:
    """Router should handle LLM exceptions gracefully and return a default decision."""
    mock_chat_ollama = mocker.patch("graph.router.ChatOllama")
    mock_instance = mocker.MagicMock()
    mock_instance.invoke.side_effect = Exception("Connection refused")
    mock_chat_ollama.return_value = mock_instance

    base_state["topic"] = "any topic"
    result = router_node(base_state)

    # Should fall back to default research_required=True and include an error
    assert result["research_required"] is True
    assert "error" in result
    assert "LLM error" in result["error"]


def test_empty_llm_response(mocker: MockerFixture, base_state: dict) -> None:
    """Router should use default values when LLM returns an empty string."""
    setup_mock_llm(mocker, "")
    base_state["topic"] = "any topic"
    result = router_node(base_state)

    # _parse_llm_response returns defaults
    assert result["research_required"] is True
    assert "error" not in result


def test_missing_topic(base_state: dict) -> None:
    """Router should return an error if no topic is provided."""
    # base_state already has topic = ""
    result = router_node(base_state)

    assert "error" in result
    assert "No topic provided" in result["error"]
    assert result["research_required"] is False


def test_prompt_file_missing(mocker: MockerFixture, base_state: dict) -> None:
    """Router should raise FileNotFoundError if the prompt file cannot be read."""
    # Simulate a missing prompt file
    mocker.patch(
        "graph.router.PROMPT_PATH.read_text",
        side_effect=FileNotFoundError,
    )
    base_state["topic"] = "any topic"

    with pytest.raises(FileNotFoundError):
        router_node(base_state)


# If the router_node were async, you would need to:
# - Mark tests with @pytest.mark.asyncio
# - Use await router_node(base_state)
# - Use await llm.ainvoke(...) in the mock setup (or patch ainvoke instead of invoke)
# The mocking strategy remains the same.