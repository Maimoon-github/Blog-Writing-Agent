.PHONY: help install dev lint test test-unit test-integration serve docker-up docker-down migrate health clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Installation ───

install: ## Install production dependencies
	pip install -e .

dev: ## Install with development dependencies
	pip install -e ".[dev]"

# ─── Code Quality ───

lint: ## Run linter (ruff)
	ruff check src/ tests/
	ruff format --check src/ tests/

format: ## Auto-format code
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck: ## Run type checker (mypy)
	mypy src/

# ─── Testing ───

test: ## Run all tests
	pytest tests/ -v

test-unit: ## Run unit tests only
	pytest tests/unit/ -v -m unit

test-integration: ## Run integration tests only
	pytest tests/integration/ -v -m integration

test-e2e: ## Run end-to-end tests
	pytest tests/e2e/ -v -m e2e

test-cov: ## Run tests with coverage report
	pytest tests/ --cov=blog_agent_system --cov-report=term-missing --cov-report=html

# ─── Application ───

serve: ## Start the FastAPI development server
	uvicorn blog_agent_system.main:app --reload --host 0.0.0.0 --port 8080

cli-generate: ## Generate a test blog (usage: make cli-generate TOPIC="your topic")
	python -m blog_agent_system.cli generate "$(TOPIC)"

# ─── Infrastructure ───

docker-up: ## Start infrastructure services (Postgres, Redis, ChromaDB, MinIO)
	docker compose up -d

docker-down: ## Stop infrastructure services
	docker compose down

docker-logs: ## View infrastructure logs
	docker compose logs -f

# ─── Database ───

migrate: ## Run database migrations
	alembic upgrade head

migrate-create: ## Create a new migration (usage: make migrate-create MSG="description")
	alembic revision --autogenerate -m "$(MSG)"

migrate-down: ## Rollback last migration
	alembic downgrade -1

# ─── Utilities ───

health: ## Check infrastructure health
	python -m blog_agent_system.cli health

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage dist/ build/ *.egg-info
