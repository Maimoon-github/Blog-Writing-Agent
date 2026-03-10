#!/usr/bin/env bash
# =============================================================================
# setup_project.sh
# Creates the full decentralized-blog-agent project file/folder hierarchy.
# Usage: bash setup_project.sh [optional-target-directory]
# =============================================================================

set -euo pipefail

ROOT="${1:-decentralized-blog-agent}"

echo "🚀 Creating project: $ROOT"

# ---------------------------------------------------------------------------
# Helper: create a file and write a one-line placeholder comment into it
# ---------------------------------------------------------------------------
mkfile() {
  local path="$ROOT/$1"
  mkdir -p "$(dirname "$path")"
  if [[ "$path" == *.py ]]; then
    echo "# $2" > "$path"
  elif [[ "$path" == *.ts || "$path" == *.tsx ]]; then
    echo "// $2" > "$path"
  elif [[ "$path" == *.sol ]]; then
    echo "// SPDX-License-Identifier: MIT" > "$path"
    echo "// $2" >> "$path"
  elif [[ "$path" == *.md ]]; then
    echo "# $2" > "$path"
  elif [[ "$path" == *.yml || "$path" == *.yaml ]]; then
    echo "# $2" > "$path"
  elif [[ "$path" == *.json ]]; then
    echo "{}" > "$path"
  elif [[ "$path" == *.txt ]]; then
    echo "# $2" > "$path"
  elif [[ "$path" == *.sh ]]; then
    echo "#!/usr/bin/env bash" > "$path"
    echo "# $2" >> "$path"
  else
    echo "# $2" > "$path"
  fi
}

# ---------------------------------------------------------------------------
# FRONTEND
# ---------------------------------------------------------------------------
mkfile "frontend/public/favicon.ico"                                          "Favicon placeholder"

mkfile "frontend/src/app/layout.tsx"                                          "Root layout with wallet provider wrappers"
mkfile "frontend/src/app/page.tsx"                                            "Landing page / topic input entry point"
mkfile "frontend/src/app/blog/[cid]/page.tsx"                                 "Dynamic route: renders blog post by IPFS CID"
mkfile "frontend/src/app/blog/page.tsx"                                       "Blog listing page showing all minted posts"
mkfile "frontend/src/app/generate/page.tsx"                                   "Generation wizard: prompt → progress → preview"
mkfile "frontend/src/app/profile/page.tsx"                                    "Author profile resolved via ENS / wallet address"

mkfile "frontend/src/components/auth/WalletConnect.tsx"                       "RainbowKit wallet connection button"
mkfile "frontend/src/components/auth/SIWEButton.tsx"                          "Sign-In with Ethereum trigger component"

mkfile "frontend/src/components/blog/BlogCard.tsx"                            "Preview card showing title, CID, author address"
mkfile "frontend/src/components/blog/BlogViewer.tsx"                          "Markdown renderer for fetched IPFS content"
mkfile "frontend/src/components/blog/ImageGallery.tsx"                        "Grid display for SD-generated section images"
mkfile "frontend/src/components/blog/CitationList.tsx"                        "Renders numbered reference list from metadata"

mkfile "frontend/src/components/generation/TopicInput.tsx"                    "Controlled input for blog topic prompt"
mkfile "frontend/src/components/generation/PlanReview.tsx"                    "Displays BlogPlan outline for user approval (HITL)"
mkfile "frontend/src/components/generation/ProgressLog.tsx"                   "Real-time SSE stream log of agent node execution"
mkfile "frontend/src/components/generation/SectionPreview.tsx"                "Live preview of individual sections as completed"

mkfile "frontend/src/components/shared/Navbar.tsx"                            "Top navigation with wallet status indicator"
mkfile "frontend/src/components/shared/Footer.tsx"                            "Site footer with IPFS/contract links"
mkfile "frontend/src/components/shared/DownloadButton.tsx"                    "Triggers .md / .html file download"

mkfile "frontend/src/hooks/useWalletAuth.ts"                                  "Wraps Wagmi hooks for SIWE auth flow"
mkfile "frontend/src/hooks/useBlogGeneration.ts"                              "Manages generation API calls and SSE streaming"
mkfile "frontend/src/hooks/useIPFS.ts"                                        "Handles IPFS upload/fetch via Web3.Storage SDK"
mkfile "frontend/src/hooks/useContract.ts"                                    "Reads/writes BlogRegistry smart contract via Wagmi"

mkfile "frontend/src/lib/wagmi.config.ts"                                     "Wagmi chain + connector configuration (Polygon, Ethereum)"
mkfile "frontend/src/lib/rainbowkit.config.ts"                                "RainbowKit theme and wallet list setup"
mkfile "frontend/src/lib/ipfs.ts"                                             "Web3.Storage / Pinata SDK wrapper for upload/fetch"
mkfile "frontend/src/lib/siwe.ts"                                             "SIWE message construction and verification helpers"
mkfile "frontend/src/lib/api.ts"                                              "Typed Axios/fetch client pointing at Django REST API"

mkfile "frontend/src/store/authStore.ts"                                      "Stores wallet address, ENS name, SIWE session"
mkfile "frontend/src/store/generationStore.ts"                                "Tracks active generation job state and progress"

mkfile "frontend/src/types/blog.ts"                                           "BlogPlan, Section, SectionDraft, ImageResult types"
mkfile "frontend/src/types/contract.ts"                                       "ABI types and on-chain BlogRecord struct"
mkfile "frontend/src/types/api.ts"                                            "Request/response types for Django REST endpoints"

mkfile "frontend/.env.local"                                                  "Frontend env vars: RPC URLs, contract address, IPFS keys"
mkfile "frontend/next.config.ts"                                              "Next.js config: rewrites, image domains, env exposure"
mkfile "frontend/tailwind.config.ts"                                          "Tailwind CSS configuration"
mkfile "frontend/tsconfig.json"                                               "TypeScript compiler options"
mkfile "frontend/package.json"                                                "Frontend dependencies (wagmi, rainbowkit, viem, etc.)"

# ---------------------------------------------------------------------------
# BACKEND — Django config
# ---------------------------------------------------------------------------
mkfile "backend/config/settings/base.py"                                      "Shared settings: installed apps, middleware, auth"
mkfile "backend/config/settings/development.py"                               "Dev overrides: DEBUG=True, local DB, CORS open"
mkfile "backend/config/settings/production.py"                                "Prod overrides: secure cookies, allowed hosts, caching"
mkfile "backend/config/urls.py"                                               "Root URL dispatcher to all app routers"
mkfile "backend/config/wsgi.py"                                               "WSGI entry point for production servers"
mkfile "backend/config/asgi.py"                                               "ASGI entry point for async/SSE support"

# ---------------------------------------------------------------------------
# BACKEND — Django apps
# ---------------------------------------------------------------------------
mkfile "backend/apps/authentication/views.py"                                 "Nonce generation and SIWE message verification endpoints"
mkfile "backend/apps/authentication/serializers.py"                           "Validates wallet address format and signed message"
mkfile "backend/apps/authentication/models.py"                                "WalletSession model storing address + nonce + expiry"
mkfile "backend/apps/authentication/urls.py"                                  "/auth/nonce/ and /auth/verify/ route definitions"
mkfile "backend/apps/authentication/middleware.py"                            "Attaches verified wallet address to request object"

mkfile "backend/apps/generation/views.py"                                     "REST endpoints: start job, stream progress, fetch result"
mkfile "backend/apps/generation/serializers.py"                               "Validates topic input; serializes BlogPlan and outputs"
mkfile "backend/apps/generation/models.py"                                    "GenerationJob model: topic, status, CID, wallet, timestamp"
mkfile "backend/apps/generation/urls.py"                                      "/generate/start/, /generate/{id}/stream/, /generate/{id}/"
mkfile "backend/apps/generation/tasks.py"                                     "Celery tasks wrapping LangGraph pipeline execution"
mkfile "backend/apps/generation/consumers.py"                                 "Django Channels WebSocket consumer for live progress"

mkfile "backend/apps/ipfs/views.py"                                           "Endpoint to upload finalized blog content to IPFS"
mkfile "backend/apps/ipfs/serializers.py"                                     "Validates upload payload and returns CID response"
mkfile "backend/apps/ipfs/services.py"                                        "web3.storage / Pinata API client abstraction"
mkfile "backend/apps/ipfs/urls.py"                                            "/ipfs/upload/ and /ipfs/fetch/{cid}/ routes"

mkfile "backend/apps/blockchain/views.py"                                     "Endpoint to trigger on-chain blog minting transaction"
mkfile "backend/apps/blockchain/serializers.py"                               "Validates CID + wallet address for minting payload"
mkfile "backend/apps/blockchain/services.py"                                  "web3.py client: connect to contract, call mintBlog()"
mkfile "backend/apps/blockchain/models.py"                                    "MintRecord model caching on-chain tx hash + CID"
mkfile "backend/apps/blockchain/urls.py"                                      "/blockchain/mint/ and /blockchain/posts/{address}/ routes"

# ---------------------------------------------------------------------------
# BACKEND — Agents
# ---------------------------------------------------------------------------
mkfile "backend/agents/router.py"                                             "Router node: classifies topic, runs safety check"
mkfile "backend/agents/planner.py"                                            "Planner node: generates structured BlogPlan via Mistral"
mkfile "backend/agents/researcher.py"                                         "Researcher worker: SearxNG search + content extraction"
mkfile "backend/agents/writer.py"                                             "Writer worker: drafts section with research context"
mkfile "backend/agents/image_agent.py"                                        "Image worker: calls Stable Diffusion v1.4 pipeline"
mkfile "backend/agents/citation_manager.py"                                   "Builds citation registry and resolves inline markers"
mkfile "backend/agents/reducer.py"                                            "Assembler: orders sections, injects images and citations"

# ---------------------------------------------------------------------------
# BACKEND — Graph
# ---------------------------------------------------------------------------
mkfile "backend/graph/state.py"                                               "GraphState TypedDict and all Pydantic models (BlogPlan etc.)"
mkfile "backend/graph/graph_builder.py"                                       "Assembles StateGraph: nodes, edges, Send() dispatch logic"
mkfile "backend/graph/checkpointer.py"                                        "SQLite checkpointer setup for crash recovery"

# ---------------------------------------------------------------------------
# BACKEND — Tools
# ---------------------------------------------------------------------------
mkfile "backend/tools/search.py"                                              "SearxNG + DuckDuckGo unified search interface"
mkfile "backend/tools/web_fetcher.py"                                         "httpx + BeautifulSoup page content extractor"
mkfile "backend/tools/image_gen.py"                                           "SD v1.4 pipeline loader and generate() wrapper"

# ---------------------------------------------------------------------------
# BACKEND — Memory
# ---------------------------------------------------------------------------
mkfile "backend/memory/chroma_store.py"                                       "ChromaDB client for long-term research vector storage"
mkfile "backend/memory/cache.py"                                              "Disk-based JSON cache for search results (24hr TTL)"

# ---------------------------------------------------------------------------
# BACKEND — Prompts
# ---------------------------------------------------------------------------
mkfile "backend/prompts/router_prompt.txt"                                    "Intent classification and content safety prompt"
mkfile "backend/prompts/planner_prompt.txt"                                   "Structured BlogPlan generation prompt"
mkfile "backend/prompts/researcher_prompt.txt"                                "Research summarization prompt"
mkfile "backend/prompts/writer_prompt.txt"                                    "Section writing with citation marker instructions"
mkfile "backend/prompts/moderator_prompt.txt"                                 "Final output hallucination and safety review prompt"

# ---------------------------------------------------------------------------
# BACKEND — Outputs (gitignored runtime directories)
# ---------------------------------------------------------------------------
mkdir -p "$ROOT/backend/outputs/images"
mkdir -p "$ROOT/backend/outputs/blogs"
mkdir -p "$ROOT/backend/outputs/logs"
touch "$ROOT/backend/outputs/images/.gitkeep"
touch "$ROOT/backend/outputs/blogs/.gitkeep"
touch "$ROOT/backend/outputs/logs/.gitkeep"

# ---------------------------------------------------------------------------
# BACKEND — Tests
# ---------------------------------------------------------------------------
mkfile "backend/tests/apps/test_authentication.py"                            "Tests SIWE nonce generation and verification"
mkfile "backend/tests/apps/test_generation.py"                                "Tests job creation, status polling, and streaming"
mkfile "backend/tests/apps/test_ipfs.py"                                      "Tests IPFS upload and CID retrieval"
mkfile "backend/tests/apps/test_blockchain.py"                                "Tests contract minting call and tx record storage"
mkfile "backend/tests/agents/test_router.py"                                  "Unit test for Router classification logic"
mkfile "backend/tests/agents/test_planner.py"                                 "Unit test for BlogPlan structured output"
mkfile "backend/tests/agents/test_researcher.py"                              "Unit test for search result summarization"
mkfile "backend/tests/agents/test_writer.py"                                  "Unit test for section drafting with mock research"
mkfile "backend/tests/agents/test_graph_integration.py"                       "End-to-end graph test on 3 standard topics"
mkdir -p "$ROOT/backend/tests/fixtures"
touch "$ROOT/backend/tests/fixtures/.gitkeep"

# ---------------------------------------------------------------------------
# BACKEND — Root files
# ---------------------------------------------------------------------------
mkfile "backend/manage.py"                                                    "Django management command entry point"
mkfile "backend/requirements.txt"                                             "All Python dependencies pinned with versions"
mkfile "backend/celery.py"                                                    "Celery app factory for background task execution"
mkfile "backend/.env.example"                                                 "Template for backend environment variables"

# ---------------------------------------------------------------------------
# CONTRACTS
# ---------------------------------------------------------------------------
mkfile "contracts/src/BlogRegistry.sol"                                       "Core contract: stores wallet → IPFS CID mappings on-chain"
mkfile "contracts/scripts/deploy.ts"                                          "Hardhat deploy script for Polygon / Ethereum testnets"
mkfile "contracts/test/BlogRegistry.test.ts"                                  "Hardhat/Chai tests for mintBlog and getBlogsByAuthor"
mkfile "contracts/abis/BlogRegistry.json"                                     "ABI generated after compilation for wagmi and web3.py"
mkfile "contracts/hardhat.config.ts"                                          "Hardhat network config (localhost, Mumbai, Mainnet)"
mkfile "contracts/package.json"                                               "Hardhat and ethers.js dev dependencies"

# ---------------------------------------------------------------------------
# INFRASTRUCTURE
# ---------------------------------------------------------------------------
mkfile "infrastructure/docker/docker-compose.yml"                             "Full local stack: app + Django + Ollama + SearxNG + Redis"
mkfile "infrastructure/docker/docker-compose.prod.yml"                        "Production overrides with resource limits and volumes"
mkfile "infrastructure/docker/Dockerfile.frontend"                            "Next.js production image build"
mkfile "infrastructure/docker/Dockerfile.backend"                             "Django + Gunicorn production image build"
mkfile "infrastructure/docker/searxng/settings.yml"                           "SearxNG engine list config (Google, DDG, Bing, Wikipedia)"
mkfile "infrastructure/nginx/nginx.conf"                                      "Reverse proxy routing /api → Django, / → Next.js"
mkfile "infrastructure/k8s/backend-deployment.yaml"                           "Django worker pod spec with HPA configuration"
mkfile "infrastructure/k8s/frontend-deployment.yaml"                          "Next.js pod spec and service definition"
mkfile "infrastructure/k8s/ollama-deployment.yaml"                            "Ollama pod with GPU node selector and volume mount"
mkfile "infrastructure/k8s/ingress.yaml"                                      "Ingress rules mapping subdomains to services"

# ---------------------------------------------------------------------------
# SHARED
# ---------------------------------------------------------------------------
mkfile "shared/types/blog-plan.schema.json"                                   "JSON Schema for BlogPlan: single source of truth for FE + BE"

# ---------------------------------------------------------------------------
# NOTEBOOKS
# ---------------------------------------------------------------------------
mkfile "notebooks/graph_explorer.ipynb"                                       "Interactive LangGraph execution and state inspection"
mkfile "notebooks/sd_prompt_tuning.ipynb"                                     "Stable Diffusion prompt quality experimentation"

# ---------------------------------------------------------------------------
# DOCS
# ---------------------------------------------------------------------------
mkfile "docs/architecture.md"                                                 "High-level architecture overview with Mermaid diagrams"
mkfile "docs/agent-design.md"                                                 "Per-agent responsibilities, inputs, outputs, and tools"
mkfile "docs/decentralization.md"                                             "Web3 layer: SIWE flow, IPFS strategy, contract design"
mkfile "docs/api-reference.md"                                                "Django REST API endpoint reference with request/response"
mkfile "docs/setup-guide.md"                                                  "Step-by-step local development setup instructions"

# ---------------------------------------------------------------------------
# ROOT FILES
# ---------------------------------------------------------------------------
mkfile ".env.example"                                                         "Root-level env template covering all services"
mkfile "Makefile"                                                             "Shortcut commands: make dev, make test, make deploy"
mkfile "README.md"                                                            "Project overview, architecture diagram, quick-start guide"

cat > "$ROOT/.gitignore" << 'EOF'
# Runtime outputs
backend/outputs/

# Environment files
.env
.env.local
**/.env

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/
.venv/
venv/

# Node
node_modules/
.next/
out/
dist/

# Jupyter
.ipynb_checkpoints/

# DB / checkpoints
*.db
*.sqlite3
checkpoints.db

# OS
.DS_Store
Thumbs.db
EOF

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "✅ Project scaffold created at: ./$ROOT"
echo ""
echo "📁 Top-level structure:"
ls "$ROOT"
echo ""
echo "Next steps:"
echo "  cd $ROOT"
echo "  # Frontend  →  cd frontend  && npm install"
echo "  # Backend   →  cd backend   && pip install -r requirements.txt"
echo "  # Contracts →  cd contracts && npm install"
