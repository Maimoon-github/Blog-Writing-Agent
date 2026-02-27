#!/bin/bash

# Create the main project directory
mkdir -p blog-agent

# Navigate into the project directory
cd blog-agent

# Create all subdirectories
mkdir -p app
mkdir -p agents
mkdir -p graph
mkdir -p tools
mkdir -p memory
mkdir -p prompts
mkdir -p outputs/{images,blogs,logs}
mkdir -p tests/fixtures
mkdir -p docker/searxng
mkdir -p notebooks

# Create all empty files
touch app/ui.py
touch app/config.py

touch agents/router.py
touch agents/planner.py
touch agents/researcher.py
touch agents/writer.py
touch agents/image_agent.py
touch agents/citation_manager.py
touch agents/reducer.py

touch graph/state.py
touch graph/graph_builder.py
touch graph/checkpointer.py

touch tools/search.py
touch tools/web_fetcher.py
touch tools/image_gen.py

touch memory/chroma_store.py
touch memory/cache.py

touch prompts/router_prompt.txt
touch prompts/planner_prompt.txt
touch prompts/researcher_prompt.txt
touch prompts/writer_prompt.txt
touch prompts/moderator_prompt.txt

# outputs subdirectories already created with mkdir -p
touch tests/test_router.py
touch tests/test_planner.py
touch tests/test_researcher.py
touch tests/test_writer.py
touch tests/test_graph_integration.py
# tests/fixtures directory already created, keeping it empty

touch docker/docker-compose.yml
touch docker/Dockerfile
touch docker/searxng/settings.yml

touch notebooks/graph_explorer.ipynb

touch requirements.txt
touch .env.example
touch README.md

echo "Blog agent project structure created successfully!"