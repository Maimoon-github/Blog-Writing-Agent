#!/bin/bash
# setup_project.sh - Creates the Autonomous AI Blog Generation System project structure

echo "🚀 Creating Autonomous AI Blog Generation System project structure..."

# Create root directory
mkdir -p blog-writing-agent

# Navigate into project
cd blog-writing-agent || exit 1

# Create root files
touch .gitignore
touch README.md
touch requirements.txt
touch docker-compose.yml
touch config.py
touch schemas.py
touch state.py
touch graph.py
touch streamlit_app.py

# Create directories and files
mkdir -p agents
mkdir -p prompts
mkdir -p tools
mkdir -p outputs/blogs
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
touch prompts/router_prompt.txt
touch prompts/planner_prompt.txt
touch prompts/researcher_prompt.txt
touch prompts/writer_prompt.txt
touch prompts/editor_prompt.txt
touch prompts/image_prompt.txt

# Create tool files
touch tools/search.py
touch tools/image_gen.py

# Add basic content to .gitignore
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Project specific
outputs/
chroma_db/
*.log

# OS
.DS_Store
Thumbs.db
EOF

echo "✅ Project structure created successfully!"
echo ""
echo "Project created at: $(pwd)"
echo ""
tree -L 3 || echo "Tree command not found. Structure created anyway."
echo ""
echo "Next steps:"
echo "1. cd blog-writing-agent"
echo "2. python -m venv venv"
echo "3. source venv/bin/activate  (or venv\\Scripts\\activate on Windows)"
echo "4. pip install -r requirements.txt"
echo "5. Start building your agents!"