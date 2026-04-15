from pathlib import Path

PROMPT_FILES = [
    Path(__file__).resolve().parent.parent / "prompts" / "planner_prompt.txt",
    Path(__file__).resolve().parent.parent / "prompts" / "writer_prompt.txt",
    Path(__file__).resolve().parent.parent / "prompts" / "research_prompt.txt",
    Path(__file__).resolve().parent.parent / "prompts" / "editor_prompt.txt",
]


def test_prompt_files_exist_and_contain_instructions() -> None:
    for prompt_path in PROMPT_FILES:
        content = prompt_path.read_text(encoding="utf-8").strip()
        assert content, f"Prompt file {prompt_path.name} must not be empty"
        assert "JSON" in content or "markup" in content or "Output only" in content
