#!/usr/bin/env bash
set -e  # Exit on error

# Create root directory
PROJECT_ROOT="blog_agent_system"
mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Create top-level files
touch README.md
touch pyproject.toml
touch .env.example
touch docker-compose.yml
touch Makefile

# Create src/blog_agent_system and its subdirectories
mkdir -p src/blog_agent_system/{config/prompts,core,agents,tools,llm,memory,persistence/models,persistence/repositories,models,api,utils}

# Create __init__.py files in all python package directories
find src -type d -exec touch {}/__init__.py \;

# Create additional specific files under src/blog_agent_system
touch src/blog_agent_system/main.py
touch src/blog_agent_system/cli.py

# config layer
touch src/blog_agent_system/config/__init__.py
touch src/blog_agent_system/config/settings.py
touch src/blog_agent_system/config/llm_config.py
touch src/blog_agent_system/config/agents.yaml
touch src/blog_agent_system/config/prompts/{research,outline,draft,edit,seo,critique}.j2

# core layer
touch src/blog_agent_system/core/__init__.py
touch src/blog_agent_system/core/state.py
touch src/blog_agent_system/core/graph.py
touch src/blog_agent_system/core/orchestrator.py
touch src/blog_agent_system/core/checkpoint.py

# agents layer
touch src/blog_agent_system/agents/__init__.py
touch src/blog_agent_system/agents/base.py
touch src/blog_agent_system/agents/research_agent.py
touch src/blog_agent_system/agents/outline_agent.py
touch src/blog_agent_system/agents/writer_agent.py
touch src/blog_agent_system/agents/editor_agent.py
touch src/blog_agent_system/agents/seo_agent.py
touch src/blog_agent_system/agents/fact_checker_agent.py
touch src/blog_agent_system/agents/image_agent.py

# tools layer
touch src/blog_agent_system/tools/__init__.py
touch src/blog_agent_system/tools/registry.py
touch src/blog_agent_system/tools/base.py
touch src/blog_agent_system/tools/web_search.py
touch src/blog_agent_system/tools/rag_retriever.py
touch src/blog_agent_system/tools/code_executor.py
touch src/blog_agent_system/tools/file_io.py
touch src/blog_agent_system/tools/image_generator.py

# llm layer
touch src/blog_agent_system/llm/__init__.py
touch src/blog_agent_system/llm/provider.py
touch src/blog_agent_system/llm/openai_provider.py
touch src/blog_agent_system/llm/anthropic_provider.py
touch src/blog_agent_system/llm/ollama_provider.py
touch src/blog_agent_system/llm/factory.py
touch src/blog_agent_system/llm/token_manager.py
touch src/blog_agent_system/llm/structured_output.py

# memory layer
touch src/blog_agent_system/memory/__init__.py
touch src/blog_agent_system/memory/short_term.py
touch src/blog_agent_system/memory/long_term.py
touch src/blog_agent_system/memory/episodic.py
touch src/blog_agent_system/memory/shared_state.py
touch src/blog_agent_system/memory/eviction.py

# persistence layer
touch src/blog_agent_system/persistence/__init__.py
touch src/blog_agent_system/persistence/database.py
touch src/blog_agent_system/persistence/vector_store.py
touch src/blog_agent_system/persistence/models/__init__.py
touch src/blog_agent_system/persistence/models/task.py
touch src/blog_agent_system/persistence/models/agent_run.py
touch src/blog_agent_system/persistence/models/document.py
touch src/blog_agent_system/persistence/models/feedback.py
touch src/blog_agent_system/persistence/repositories/__init__.py
touch src/blog_agent_system/persistence/repositories/task_repo.py
touch src/blog_agent_system/persistence/repositories/document_repo.py

# models (pydantic schemas)
touch src/blog_agent_system/models/__init__.py
touch src/blog_agent_system/models/blog.py
touch src/blog_agent_system/models/research.py
touch src/blog_agent_system/models/workflow.py

# api layer
touch src/blog_agent_system/api/__init__.py
touch src/blog_agent_system/api/router.py
touch src/blog_agent_system/api/dependencies.py
touch src/blog_agent_system/api/schemas.py

# utils
touch src/blog_agent_system/utils/__init__.py
touch src/blog_agent_system/utils/logging.py
touch src/blog_agent_system/utils/exceptions.py
touch src/blog_agent_system/utils/validators.py

# Create tests directory
mkdir -p tests/{unit,integration,e2e}
touch tests/__init__.py
touch tests/conftest.py
touch tests/unit/__init__.py
touch tests/unit/test_agents.py
touch tests/unit/test_tools.py
touch tests/unit/test_llm_providers.py
touch tests/integration/__init__.py
touch tests/integration/test_graph_execution.py
touch tests/integration/test_memory_persistence.py
touch tests/e2e/__init__.py
touch tests/e2e/test_blog_workflow.py

# Create alembic migrations structure
mkdir -p alembic/versions
touch alembic/env.py
touch alembic/script.py.mako

# Create docs directory with ADRs
mkdir -p docs
touch docs/adr-001-orchestrator-choice.md
touch docs/adr-002-memory-strategy.md
touch docs/adr-003-llm-abstraction.md

# Create scripts directory
mkdir -p scripts
touch scripts/seed_db.py
touch scripts/health_check.py

echo "Project hierarchy created successfully under ./$PROJECT_ROOT"