# Architecture

This system is a LangGraph orchestrator-driven blog pipeline.

- Supervisor layer: Router, Planner, Reducer.
- Worker layer: Researcher, Writer, Editor, Image agents.
- Shared state: typed GraphState for topic, plan, research, sections, images, citations.
- Parallel execution: LangGraph Send() for worker scatter-gather.
- Outputs: Markdown blog, optional HTML, images, citation registry.
