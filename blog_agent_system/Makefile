.PHONY: help install dev lint format typecheck test test-unit test-integration serve docker-up docker-down health clean pull-ollama

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Installation ───
install: ## Install production dependencies
	poetry install --no-root

dev: ## Install with development dependencies
	poetry install --with dev

# ─── Code Quality ───
lint: ## Run linter (ruff)
	poetry run ruff check src/ tests/

format: ## Auto-format code
	poetry run ruff format src/ tests/
	poetry run ruff check --fix src/ tests/

typecheck: ## Run type checker (mypy)
	poetry run mypy src/

# ─── Testing ───
test: ## Run all tests
	poetry run pytest tests/ -v

test-unit: ## Run unit tests only
	poetry run pytest tests/unit/ -v -m unit

test-integration: ## Run integration tests only
	poetry run pytest tests/integration/ -v -m integration

# ─── Application ───
serve: ## Start the FastAPI development server
	poetry run uvicorn blog_agent_system.main:app --reload --host 0.0.0.0 --port 8080

cli-generate: ## Generate a test blog (make cli-generate TOPIC="your topic")
	poetry run python -m blog_agent_system.cli generate "$(TOPIC)"

# ─── Infrastructure ───
docker-up: ## Start infrastructure (Postgres, Redis, ChromaDB, Ollama)
	docker compose up -d

docker-down: ## Stop infrastructure
	docker compose down

docker-logs: ## View logs
	docker compose logs -f

pull-ollama: ## Pull latest Ollama models (run once)
	docker compose exec ollama ollama pull mistral
	docker compose exec ollama ollama pull gemma2:9b

# ─── Database ───
migrate: ## Run migrations
	poetry run alembic upgrade head

migrate-create: ## Create new migration
	poetry run alembic revision --autogenerate -m "$(MSG)"

# ─── Utilities ───
health: ## Check infrastructure health
	poetry run python -m blog_agent_system.cli health

clean: ## Clean caches and artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage dist/ build/ *.egg-info