# Developer workflow. `make setup` once, then `make test` / `make lint` / `make fmt`.
# Override PY to pick an interpreter (3.10+ for the app / MCP / browser extras).
PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-11s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install -U pip

.PHONY: setup
setup: $(BIN)/python ## Create a venv and install the core dev toolchain
	$(BIN)/python -m pip install -e ".[dev]"
	$(BIN)/pre-commit install || true

.PHONY: setup-all
setup-all: $(BIN)/python ## Install everything (app, MCP, browser) + the headless browser
	$(BIN)/python -m pip install -e ".[dev,app,mcp,browser]"
	$(BIN)/python -m playwright install chromium || true
	$(BIN)/pre-commit install || true

.PHONY: test
test: ## Run the test suite offline (mock mode)
	AGENTSYNTH_FORCE_MOCK=1 $(BIN)/python -m pytest -q

.PHONY: lint
lint: ## Run ruff + mypy exactly like CI
	$(BIN)/ruff check .
	$(BIN)/ruff format --check agentsynth app.py tests examples scripts
	$(BIN)/mypy

.PHONY: fmt
fmt: ## Auto-format and apply safe lint fixes
	$(BIN)/ruff format agentsynth app.py tests examples scripts
	$(BIN)/ruff check --fix .

.PHONY: app
app: ## Run the Gradio app locally
	$(BIN)/python app.py

.PHONY: docs
docs: ## Serve the docs site locally
	$(BIN)/mkdocs serve

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf build dist ./*.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
