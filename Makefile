.PHONY: install install-dev run format check lint collect test

# Optional overrides:  make run MODEL=Qwen/Qwen3.5-27B CAPABILITY=tools
MODEL ?=
CAPABILITY ?=
_MODEL = $(if $(MODEL),--model $(MODEL),)
_CAP = $(if $(CAPABILITY),--capability $(CAPABILITY),)

# Prefer uv when available (uv-created venvs have no `pip`); fall back to pip.
PIP := $(shell command -v uv >/dev/null 2>&1 && echo "uv pip" || echo "pip")

install:        ## editable install with dev tools (ruff)
	$(PIP) install -e ".[dev]"

install-dev: install

# Run the capability checks (the ✔/✗ table).
run:
	mcs $(_MODEL) $(_CAP)

# Auto-fix lint + format the tree.
format:
	ruff check --fix .
	ruff format .

# Lint + format gate (what CI runs on every PR). Modifies nothing.
check:
	ruff check .
	ruff format --check .

lint: check

# Offline: imports + test collection only, no API calls.
collect:
	pytest --collect-only -q

# Full suite (needs CSCS_SERVING_API / MCS_API_KEY set).
test:
	pytest
