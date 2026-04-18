#!/bin/bash
# create_project.sh - Creates the Autonomous AI Blog Generation System project structure

set -e  # Exit on any error

echo "🚀 Creating Autonomous AI Blog Generation System project structure..."

# Create root files
touch .gitignore
touch README.md
touch requirements.txt
touch docker-compose.yml
touch streamlit_app.py
touch config.py
touch schemas.py
touch state.py
touch graph.py

# Create directories
mkdir -p agents
mkdir -p prompts
mkdir -p tools
mkdir -p outputs/images

# Create agent files
touch agents/router.py
touch agents/planner.py
touch agents/researcher.py
touch agents/writer.py
touch agents/editor.py
touch agents/image_agent.py
touch agents/citation_manager.py
touch agents/reducer.py

# Create prompt files
touch prompts/editor_prompt.txt
touch prompts/planner_prompt.txt
touch prompts/writer_prompt.txt
touch prompts/router_prompt.txt
touch prompts/image_prompt.txt

# Create tool file
touch tools/search.py

# Create .gitkeep for outputs/images
touch outputs/images/.gitkeep

# Make the script executable (if needed)
chmod +x "$0"

echo "✅ Project structure created successfully!"
echo ""
echo "Project layout:"
tree -L 3 || echo "Tree command not found. Structure created anyway."
echo ""
echo "Next steps:"
echo "1. Run: pip install -r requirements.txt  (after filling it)"
echo "2. Start SearxNG: docker compose up -d"
echo "3. Run: ollama pull mistral"
echo "4. Launch app: streamlit run streamlit_app.py"
