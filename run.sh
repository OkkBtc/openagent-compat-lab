#!/usr/bin/env bash
# Bootstrap + run mcs against a hosted API.
#
# Runs the deterministic suites and prints a ✔/✗ capability table (exit non-zero
# on failure). Scope with --capability TYPE; compare models with repeated --model.
#
# Remote (the common case):
#   export CSCS_SERVING_API=...   # bearer token
#   curl -fsSL https://raw.githubusercontent.com/swiss-ai/model-compatibility-suite/main/run.sh | bash
#
# Scoped:
#   curl -fsSL .../run.sh | bash -s -- --suite tools,streaming --model <id>
#
# Inside a checkout (skip the git install, use the local tree):
#   bash run.sh --local --suite core
#
# Flags: --suite a,b  --model ID  --spec openai|dev  --base-url URL  --junit PATH  --local
# Config via env: MCS_API_BASE, MCS_API_KEY|CSCS_SERVING_API,
#                 MCS_MODEL, MCS_TIMEOUT.
# See SPEC.md section 3.
set -euo pipefail

REPO="${MCS_REPO:-https://github.com/swiss-ai/model-compatibility-suite}"
REF="${MCS_REF:-main}"
LOCAL=0
ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --local) LOCAL=1; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

if [ -z "${MCS_API_KEY:-}" ] && [ -z "${CSCS_SERVING_API:-}" ]; then
  echo "error: set CSCS_SERVING_API (or MCS_API_KEY) to your bearer token" >&2
  exit 2
fi

PYTHON="${PYTHON:-python3}"
VENV="$(mktemp -d)/venv"
cleanup() { rm -rf "$(dirname "$VENV")"; }
trap cleanup EXIT

echo "Setting up test environment..."
"$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
. "$VENV/bin/activate"
pip install --quiet --upgrade pip

# Install from git by default. Only use the local tree with an explicit --local
# (otherwise a stray ./pyproject.toml in the CWD -- e.g. another project -- would
# get installed instead of mcs).
if [ "$LOCAL" -eq 1 ]; then
  pip install --quiet -e ".[dev]"
else
  pip install --quiet "mcs @ git+${REPO}@${REF}"
fi

echo "Running mcs..."
# Runs all capability checks by default. Pass --capability TYPE / --model
# through ARGS to scope or compare.
set +e
# `${ARGS[@]+...}` guards against the macOS bash 3.2 "unbound variable" error
# when ARGS is empty under `set -u`.
mcs ${ARGS[@]+"${ARGS[@]}"}
status=$?
set -e
exit "$status"
