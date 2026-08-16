#!/usr/bin/env bash
# Bootstrap + run openagent-compat-lab against an OpenAI-style API.
#
# Runs deterministic agent protocol checks and exits non-zero on failure.
#
# Remote:
#   export ACL_API_KEY=...   # bearer token
#   curl -fsSL https://raw.githubusercontent.com/OkkBtc/openagent-compat-lab/main/run.sh | \
#     bash -s -- --profile codex --base-url <url> --model <id>
#
# Inside a checkout (skip the git install, use the local tree):
#   bash run.sh --local --profile hermes --base-url <url> --model <id>
#
# Config via env: ACL_API_BASE, ACL_API_KEY, ACL_MODEL, ACL_TIMEOUT.
set -euo pipefail

REPO="${ACL_REPO:-https://github.com/OkkBtc/openagent-compat-lab}"
REF="${ACL_REF:-main}"
LOCAL=0
ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --local) LOCAL=1; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

ACL_PYTHON_BIN="${ACL_PYTHON_BIN:-python3}"
ACL_TEMP_DIR="$(mktemp -d)"
ACL_VENV="$ACL_TEMP_DIR/venv"
cleanup() { rm -rf "$ACL_TEMP_DIR"; }
trap cleanup EXIT

echo "Setting up test environment..."
"$ACL_PYTHON_BIN" -m venv "$ACL_VENV"
# shellcheck disable=SC1091
. "$ACL_VENV/bin/activate"
pip install --quiet --upgrade pip

# Install from git by default. Only use the local tree with an explicit --local
# (otherwise a stray ./pyproject.toml in the CWD -- e.g. another project -- would
# get installed instead of openagent-compat-lab).
if [ "$LOCAL" -eq 1 ]; then
  pip install --quiet -e ".[dev]"
else
  pip install --quiet "openagent-compat-lab @ git+${REPO}@${REF}"
fi

echo "Running openagent-compat-lab..."
set +e
# `${ARGS[@]+...}` guards against the macOS bash 3.2 "unbound variable" error
# when ARGS is empty under `set -u`.
agent-compat ${ARGS[@]+"${ARGS[@]}"}
status=$?
set -e
exit "$status"
