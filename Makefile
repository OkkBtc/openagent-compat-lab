.PHONY: install install-dev run format check lint collect test build

# Optional overrides: make run MODEL=qwen3 BASE_URL=http://localhost:11434/v1
MODEL ?=
BASE_URL ?=
PROFILE ?= generic
_MODEL = $(if $(MODEL),--model $(MODEL),)
_BASE = $(if $(BASE_URL),--base-url $(BASE_URL),)

# Prefer uv when available (uv-created venvs have no `pip`); fall back to pip.
PIP := $(shell command -v uv >/dev/null 2>&1 && echo "uv pip" || echo "pip")

install:        ## editable install with dev tools (ruff)
	$(PIP) install -e ".[dev]"

install-dev: install

# Run the focused agent checks (the ✔/✗ table).
run:
	agent-compat --profile $(PROFILE) $(_MODEL) $(_BASE)

# Auto-fix lint + format the tree.
format:
	ruff check --fix .
	ruff format .

# Lint + format gate (what CI runs on every PR). Modifies nothing.
check:
	ruff check .
	ruff format --check .

lint: check

# Offline: collect the inherited full suite, without API calls.
collect:
	pytest mcs/suites --collect-only -q

# Offline agent-profile tests using the local mock server.
test:
	pytest tests -q

build:
	python -m build
