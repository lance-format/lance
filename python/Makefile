.DEFAULT_GOAL := help
.PHONY: help install build test integtest doctest compattest format format-python lint lint-python lint-rust clean
PYTHON ?=
PYTEST_ARGS ?= -vvv -s
KEEP_COMPOSE ?= 0
COMPOSE_FILE ?= ../docker-compose.yml
UV_SYNC = uv sync
UV_RUN = uv run --frozen

# Set PYTHON=3.12 to use a specific Python version with uv.
ifneq ($(strip $(PYTHON)),)
  UV_PYTHON_ARG = --python $(PYTHON)
  UV_SYNC += $(UV_PYTHON_ARG)
  UV_RUN += $(UV_PYTHON_ARG)
endif

ifeq ($(CI), true)
  PYTEST_ARGS += --durations=30
endif

# Number of pytest-xdist workers, e.g. `auto` or an explicit count. Empty (the
# default) runs serially; CI sets it per runner size. Much of the suite is spent
# waiting on IO rather than saturating the Rust core's own thread pools, so
# sharding across processes wins back time a single pytest process leaves idle.
PYTEST_WORKERS ?=
ifneq ($(strip $(PYTEST_WORKERS)),)
  # loadgroup rather than the default scheduler so that tests sharing state on
  # disk can be pinned to a single worker with @pytest.mark.xdist_group.
  PYTEST_ARGS += -n $(PYTEST_WORKERS) --dist loadgroup
endif

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install: ## Sync dependencies and set up local development tools
	@if ! command -v uv > /dev/null 2>&1; then \
		echo "uv not found. Install it from https://docs.astral.sh/uv/getting-started/installation/"; \
		exit 1; \
	fi
	$(UV_SYNC)
	uv tool run pre-commit install

build: ## Build the local Rust extension with maturin
	$(UV_RUN) maturin develop --uv

test: ## Run Python tests
	pytest $(PYTEST_ARGS) python/tests

integtest: ## Start LocalStack and run integration tests
	@if [ "$(KEEP_COMPOSE)" = "1" ]; then \
		docker compose -f $(COMPOSE_FILE) up -d --wait && \
		pytest --run-integration $(PYTEST_ARGS) python/tests/test_s3_ddb.py python/tests/test_namespace_integration.py; \
	else \
		trap 'docker compose -f $(COMPOSE_FILE) down -v --remove-orphans --timeout 0 >/dev/null 2>&1 || true' EXIT; \
		docker compose -f $(COMPOSE_FILE) up -d --wait && \
		pytest --run-integration $(PYTEST_ARGS) python/tests/test_s3_ddb.py python/tests/test_namespace_integration.py; \
	fi

doctest: ## Run Python doctests
	pytest --doctest-modules $(PYTEST_ARGS) python/lance

compattest: ## Run upgrade/downgrade compatibility tests
	pytest --run-compat $(PYTEST_ARGS) python/tests/compat

format: format-python ## Auto-format Python and Rust
	cargo fmt

format-python: ## Auto-format Python (ruff)
	ruff format python
	ruff check --fix python

lint: lint-python lint-rust ## Check Python and Rust formatting/lints

lint-python: ## Check Python formatting, linting, and types
	ruff format --check --diff python
	ruff check python
	pyright

lint-rust: ## Check Rust formatting and clippy lints
	cargo fmt -- --check
	cargo clippy -- -D warnings

clean: ## Remove build artifacts
	cargo clean
	rm -rf target
