.PHONY: help install install-dev test lint format typecheck run backtest train dashboard docker-build docker-up docker-down clean

PYTHON ?= python
PIP ?= pip

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Installation ─────────────────────────────────────────────────

install: ## Install production dependencies
	$(PIP) install -e .

install-dev: ## Install with dev dependencies
	$(PIP) install -e ".[dev]"

# ── Quality ──────────────────────────────────────────────────────

test: ## Run test suite
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage
	$(PYTHON) -m pytest tests/ -v --cov=hedgefund --cov-report=term-missing --cov-report=html

lint: ## Run linter (ruff)
	$(PYTHON) -m ruff check src/ tests/

format: ## Auto-format code
	$(PYTHON) -m ruff format src/ tests/
	$(PYTHON) -m ruff check --fix src/ tests/

typecheck: ## Run type checker (mypy)
	$(PYTHON) -m mypy src/hedgefund/

check: lint typecheck test ## Run all checks (lint + typecheck + test)

# ── Running ──────────────────────────────────────────────────────

run: ## Start live trading
	$(PYTHON) -m hedgefund run

backtest: ## Run backtesting (use STRATEGY=, START=, END=)
	$(PYTHON) -m hedgefund backtest --strategy $(STRATEGY) --start-date $(START) --end-date $(END)

train: ## Train ML models (use MODEL=all|lstm|transformer|rl|random_forest)
	$(PYTHON) -m hedgefund train --model $(or $(MODEL),all)

dashboard: ## Start dashboard server
	$(PYTHON) -m hedgefund dashboard

# ── Docker ───────────────────────────────────────────────────────

docker-build: ## Build Docker image
	docker build -f docker/Dockerfile -t hedgefund:latest .

docker-up: ## Start full stack with docker-compose
	docker compose -f docker/docker-compose.yml up -d

docker-down: ## Stop docker-compose stack
	docker compose -f docker/docker-compose.yml down

docker-logs: ## Tail docker-compose logs
	docker compose -f docker/docker-compose.yml logs -f

# ── Utilities ────────────────────────────────────────────────────

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ .coverage
