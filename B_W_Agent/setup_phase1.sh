#!/usr/bin/env bash
# Phase 1: Environment Setup for Autonomous AI Blog Generation System
# Run this script from the project root directory.

set -euo pipefail

# --- Color helpers ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[✓]${NC} $1"; }
warning() { echo -e "${YELLOW}[!]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# --- Configuration ---
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/bwa_env"
REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"
OLLAMA_MODEL="mistral:7b"
SEARXNG_CONTAINER_NAME="searxng-local"
SEARXNG_PORT="8080"
OLLAMA_PORT="11434"

# Global variable for detected Python command
PY_CMD=""

# --- Step 1: OS & Prerequisites Check ---
check_prerequisites() {
    info "Checking system prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH. Please install Docker Desktop or Docker Engine."
    fi
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker."
    fi
    success "Docker is running."

    # Check Ollama
    if ! command -v ollama &> /dev/null; then
        warning "Ollama is not installed. Attempting to install via official script..."
        if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
            curl -fsSL https://ollama.ai/install.sh | sh
            if ! command -v ollama &> /dev/null; then
                error "Ollama installation failed. Please install manually from https://ollama.ai"
            fi
            success "Ollama installed."
        else
            error "Automatic Ollama installation only supported on macOS/Linux. For Windows, please install manually and run this script under WSL."
        fi
    else
        success "Ollama is installed."
    fi

    # --- Python Detection (robust for Conda/Windows) ---
    info "Detecting Python interpreter..."
    if command -v python3 &> /dev/null; then
        PY_CMD="python3"
    elif command -v python &> /dev/null; then
        PY_CMD="python"
    elif command -v py &> /dev/null; then
        PY_CMD="py"
    else
        error "Python is not installed or not in PATH. Please install Python 3.11+."
    fi
    info "Using Python command: $PY_CMD"

    # Check Python version (>=3.11) without bc (portable)
    PYTHON_VERSION=$($PY_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
    if [[ "$MAJOR" -lt 3 ]] || [[ "$MAJOR" -eq 3 && "$MINOR" -lt 11 ]]; then
        error "Python 3.11+ required. Found: $PYTHON_VERSION"
    fi
    success "Python $PYTHON_VERSION detected."

    # Check if we're inside a Conda environment
    if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        info "Conda environment active: $CONDA_DEFAULT_ENV"
    else
        warning "No Conda environment active. It's recommended to use a dedicated environment (will create venv instead)."
    fi

    # Check disk space (rough estimate ~8GB free)
    # Using more portable method (works on Git Bash)
    if command -v df &> /dev/null; then
        AVAIL_GB=$(df -BG . 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//')
    else
        AVAIL_GB=""
    fi
    if [[ -n "$AVAIL_GB" && ${AVAIL_GB:-0} -lt 8 ]]; then
        warning "Less than 8GB free disk space. Model downloads may fail."
    else
        success "Disk space check OK (or skipped)."
    fi
}

# --- Step 2: Ollama Model Setup ---
setup_ollama() {
    info "Setting up Ollama with $OLLAMA_MODEL..."

    # Start Ollama server if not already running
    # pgrep not available on Windows Git Bash; use alternative
    if command -v pgrep &> /dev/null; then
        if ! pgrep -f "ollama serve" > /dev/null; then
            info "Starting Ollama server in background..."
            ollama serve &> "$PROJECT_DIR/ollama.log" &
            sleep 5
        fi
    else
        # On Windows, assume Ollama is already running as a service
        info "Skipping Ollama server start (Windows detected - please ensure Ollama is running)."
    fi

    # Verify server is up
    if ! curl -s "http://localhost:$OLLAMA_PORT/api/tags" > /dev/null; then
        error "Ollama server not responding on port $OLLAMA_PORT. Is Ollama running?"
    fi
    success "Ollama server is reachable."

    # Pull model if not present
    if ollama list | grep -q "$OLLAMA_MODEL"; then
        success "Model $OLLAMA_MODEL already pulled."
    else
        info "Pulling $OLLAMA_MODEL (this may take a while)..."
        ollama pull "$OLLAMA_MODEL"
        success "Model $OLLAMA_MODEL pulled."
    fi

    # Quick test generation
    info "Testing model with a simple prompt..."
    TEST_OUTPUT=$(ollama run "$OLLAMA_MODEL" "Say 'setup successful' in one word." 2>/dev/null | head -1)
    if [[ "$TEST_OUTPUT" == *"successful"* ]] || [[ "$TEST_OUTPUT" == *"Success"* ]]; then
        success "Model test passed."
    else
        warning "Model test returned unexpected output, but may still work."
    fi
}

# --- Step 3: SearxNG Setup ---
setup_searxng() {
    info "Setting up SearxNG on port $SEARXNG_PORT..."

    # Stop and remove existing container if present
    if docker ps -a --format '{{.Names}}' | grep -q "^${SEARXNG_CONTAINER_NAME}$"; then
        info "Removing existing SearxNG container..."
        docker rm -f "$SEARXNG_CONTAINER_NAME" > /dev/null
    fi

    # Pull and run
    info "Starting SearxNG container..."
    docker run -d \
        --name "$SEARXNG_CONTAINER_NAME" \
        -p "$SEARXNG_PORT:8080" \
        --restart unless-stopped \
        searxng/searxng:latest

    # Wait for health check
    info "Waiting for SearxNG to become healthy..."
    for i in {1..30}; do
        if curl -s "http://localhost:$SEARXNG_PORT" > /dev/null; then
            success "SearxNG is reachable at http://localhost:$SEARXNG_PORT"
            return 0
        fi
        sleep 2
    done
    error "SearxNG failed to start within 60 seconds. Check Docker logs."
}

# --- Step 4: Python Virtual Environment ---
setup_venv() {
    # Skip if Conda environment is already active
    if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        info "Using active Conda environment '$CONDA_DEFAULT_ENV' (skipping venv creation)."
        return 0
    fi

    info "Creating Python virtual environment at $VENV_DIR..."
    $PY_CMD -m venv "$VENV_DIR"
    success "Virtual environment created."

    # Activate and upgrade pip
    # Activation for subsequent steps handled by sourcing
    source "$VENV_DIR/bin/activate" || source "$VENV_DIR/Scripts/activate"
    $PY_CMD -m pip install --upgrade pip setuptools wheel
    success "pip upgraded."
}

# --- Step 5: Install Python Dependencies ---
install_dependencies() {
    info "Installing Python dependencies from $REQUIREMENTS_FILE..."
    if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
        error "requirements.txt not found in $PROJECT_DIR"
    fi

    # Ensure we're in the right environment
    if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        # Conda env active, use pip directly
        pip install -r "$REQUIREMENTS_FILE"
    else
        # Venv mode
        source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate"
        pip install -r "$REQUIREMENTS_FILE"
    fi
    success "Dependencies installed."
}

# --- Step 6: Pre-download Stable Diffusion v1.4 ---
download_sd_model() {
    info "Pre-downloading Stable Diffusion v1.4 weights..."

    # Activate appropriate environment
    if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        true  # already active
    else
        source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate"
    fi

    # Check for CUDA
    $PY_CMD -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

    # Download the pipeline (caches to ~/.cache/huggingface)
    $PY_CMD -c "
from diffusers import StableDiffusionPipeline
import torch

print('Downloading Stable Diffusion v1.4...')
pipe = StableDiffusionPipeline.from_pretrained(
    'CompVis/stable-diffusion-v1-4',
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    safety_checker=None,
    requires_safety_checker=False
)
print('Download complete. Model cached.')
"
    success "Stable Diffusion v1.4 cached."
}

# --- Step 7: Smoke Test (CrewAI + Ollama + SearxNG) ---
smoke_test() {
    info "Running full-stack smoke test (CrewAI + Ollama + SearxNG)..."

    # Activate appropriate environment
    if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        true
    else
        source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate"
    fi

    # Create a temporary Python script
    cat > "$PROJECT_DIR/smoke_test.py" << 'EOF'
import sys
from langchain_community.utilities import SearxSearchWrapper
from langchain_community.chat_models import ChatOllama
from crewai import Crew, Agent, Task

def main():
    # Set up tools and LLM
    search_tool = SearxSearchWrapper(searx_host="http://localhost:8080")
    llm = ChatOllama(model="mistral:7b", temperature=0.3)

    # Define agents
    researcher = Agent(
        role="Researcher",
        goal="Fetch background info",
        tools=[search_tool],
        llm=llm,
        verbose=True
    )
    writer = Agent(
        role="Writer",
        goal="Draft blog post",
        tools=[],
        llm=llm,
        verbose=True
    )

    # Define tasks
    research_task = Task(
        description="Research the topic 'Why local AI matters'",
        agent=researcher,
        expected_output="A summary of research findings"
    )
    write_task = Task(
        description="Write a short blog draft about 'Why local AI matters'",
        agent=writer,
        expected_output="A 2-paragraph markdown draft"
    )

    # Create crew
    blog_crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        verbose=True
    )

    # Run
    topic = "Why local AI matters"
    result = blog_crew.kickoff(inputs={"topic": topic})
    print("\n===== SMOKE TEST RESULT =====\n")
    print(result)
    print("\n===== END =====\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF

    # Run the test
    if $PY_CMD "$PROJECT_DIR/smoke_test.py"; then
        success "Smoke test passed! Blog draft generated."
        rm "$PROJECT_DIR/smoke_test.py"
    else
        error "Smoke test failed. Please check logs above."
    fi
}

# --- Step 8: Final Validation & Handover ---
final_handover() {
    echo ""
    echo "=============================================="
    echo -e "${GREEN}Phase 1 Setup Complete!${NC}"
    echo "=============================================="
    echo ""
    echo "✅ Ollama running with $OLLAMA_MODEL"
    echo "✅ SearxNG running at http://localhost:$SEARXNG_PORT"
    if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        echo "✅ Using Conda environment: $CONDA_DEFAULT_ENV"
    else
        echo "✅ Python virtual environment at $VENV_DIR"
    fi
    echo "✅ All dependencies installed"
    echo "✅ Stable Diffusion v1.4 cached"
    echo "✅ Smoke test passed"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        echo "1. You're already in Conda env '$CONDA_DEFAULT_ENV'"
    else
        echo "1. Activate the environment:"
        echo "   source $VENV_DIR/bin/activate"
    fi
    echo "2. (Optional) Create a .env file based on .env.example"
    echo "3. Proceed to Phase 2 implementation"
    echo ""
    echo "To stop SearxNG: docker stop $SEARXNG_CONTAINER_NAME"
    echo "To stop Ollama: pkill ollama   (or stop the Windows service)"
    echo ""
}

# --- Main execution ---
main() {
    echo -e "${BLUE}=============================================="
    echo "Autonomous AI Blog System - Phase 1 Setup"
    echo -e "==============================================${NC}"
    check_prerequisites
    setup_ollama
    setup_searxng
    setup_venv
    install_dependencies
    download_sd_model
    smoke_test
    final_handover
}

main "$@"