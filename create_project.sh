#!/bin/bash

# Create project root
mkdir -p paddleaurum-blog-agent
cd paddleaurum-blog-agent || exit

# Create root-level files
touch .env .gitignore requirements.txt README.md

# Create main application files
touch main.py scheduler.py

# Graph directory
mkdir -p graph
touch graph/__init__.py graph/state.py graph/graph_builder.py graph/routers.py

# Nodes directory
mkdir -p nodes
touch nodes/__init__.py
touch nodes/input_validator.py
touch nodes/planner.py
touch nodes/research_worker.py
touch nodes/research_merger.py
touch nodes/keyword_mapper.py
touch nodes/outline_agent.py
touch nodes/coaching_writer.py
touch nodes/seo_auditor.py
touch nodes/reflection.py
touch nodes/image_selector.py
touch nodes/citation_formatter.py
touch nodes/schema_generator.py
touch nodes/final_assembler.py
touch nodes/human_review_gate.py
touch nodes/publish.py
touch nodes/error_recovery.py

# Agents directory
mkdir -p agents
touch agents/__init__.py
touch agents/seo_strategist.py
touch agents/content_strategist.py
touch agents/coach_writer.py

# Tools directory
mkdir -p tools
touch tools/__init__.py
touch tools/rag_retrieval.py
touch tools/keyword_analysis.py
touch tools/word_count.py
touch tools/url_validator.py
touch tools/serp_analysis.py

# RAG directory
mkdir -p rag/collections
touch rag/__init__.py
touch rag/ingest.py
touch rag/embeddings.py
touch rag/collections/pickleball_rules.py
touch rag/collections/coaching_materials.py
touch rag/collections/seo_guidelines.py
touch rag/collections/published_articles.py
touch rag/collections/keyword_history.py

# Data directory and subfolders
mkdir -p data/coaching_manuals data/published_posts
touch data/usapa_rulebook_2025.pdf
touch data/seo_checklist.md
# The subdirectories coaching_manuals/ and published_posts/ remain empty for now

# Checkpoints
mkdir -p checkpoints
touch checkpoints/checkpoints.db

# UI directory and components
mkdir -p ui/components ui/assets
touch ui/app.py
touch ui/components/article_preview.py
touch ui/components/seo_dashboard.py
touch ui/components/approval_controls.py

# Config directory and prompts
mkdir -p config/prompts
touch config/settings.py
touch config/prompts/seo_strategist.txt
touch config/prompts/content_strategist.txt
touch config/prompts/coach_writer.txt

# Tests directory
mkdir -p tests/test_nodes tests/test_tools
touch tests/test_graph_routes.py
# test_nodes and test_tools will contain files later; we only create the directories

# Logs directory
mkdir -p logs
touch logs/pipeline.log

echo "Project structure created successfully under paddleaurum-blog-agent/"