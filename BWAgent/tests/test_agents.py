from agents.citation_manager import citation_manager_node
from graph.state import ResearchResult, SectionDraft


def test_citation_manager_resolves_sources() -> None:
    draft = SectionDraft(
        section_id="section_1",
        title="Introduction",
        content="This is a sample section. [SOURCE_1]",
        citation_keys=["[SOURCE_1]"],
    )
    research = ResearchResult(
        section_id="section_1",
        query="sample query",
        summary="A brief summary.",
        source_urls=["https://example.com/article"],
    )
    result = citation_manager_node({"section_drafts": [draft], "research_results": [research]})
    assert result["citation_registry"]["[SOURCE_1]"] == "https://example.com/article"
