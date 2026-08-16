# Agent Compat Lab

Find out whether an “OpenAI-compatible” LLM endpoint actually works with
**Codex**, **Hermes**, **OpenClaw**, or a generic tool-using agent.

Agent Compat Lab sends a small deterministic probe set directly from your
machine to the endpoint, then produces a pass/fail table, JSON, or Markdown.
It tests protocol behavior—not answer quality—and exits non-zero when a required
behavior is missing.

## What it checks

| Profile | API path | Checks |
|---|---|---|
| `generic` | Chat Completions | models/auth, basic response, SSE `[DONE]`, forced tools, optional argument semantics, JSON Schema, image `detail: original` |
| `hermes` | Chat Completions | the same focused checks under a Hermes-labelled report |
| `openclaw` | Chat Completions | the same focused checks under an OpenClaw-labelled report |
| `codex` | Responses API | models/auth, basic response, `response.completed`, forced tools, optional argument semantics, JSON Schema, `input_image` with `detail: original` |
| `model` | Chat Completions + optional extensions | the full upstream Model Compatibility Suite |

The agent profiles intentionally contain seven fast checks. They do not claim
that every feature of an agent is supported; they catch the protocol gaps that
most often break setup before a real task starts.

## Quick start

Requires Python 3.10 or newer.

```bash
git clone https://github.com/OkkBtc/agent-compat-lab.git
cd agent-compat-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Set credentials in the environment so they do not enter shell history:

```bash
export ACL_API_KEY="your-provider-key"

agent-compat \
  --profile hermes \
  --base-url https://provider.example/v1 \
  --model provider/model-name
```

For Codex's Responses API path:

```bash
agent-compat \
  --profile codex \
  --base-url https://provider.example/v1 \
  --model provider/model-name
```

For a local endpoint that intentionally has no authentication:

```bash
agent-compat \
  --profile generic \
  --base-url http://127.0.0.1:11434/v1 \
  --model qwen3 \
  --allow-no-auth
```

`--allow-no-auth` is explicit by design; a missing key otherwise stops before
any network request is sent.

## Reports and automation

Print machine-readable JSON:

```bash
agent-compat --profile codex --base-url "$BASE_URL" --model "$MODEL" --json
```

Write a shareable Markdown report while keeping the console table:

```bash
agent-compat \
  --profile openclaw \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --markdown compat-report.md
```

The process exits with `0` only when every check passes. This makes it suitable
for CI, provider acceptance, and regression checks.

Optional redacted wire records can help diagnose a failed provider:

```bash
agent-compat \
  --profile hermes \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --record-responses ./compat-wire
```

The destination must not already exist. Authorization headers are never
recorded, and known credentials are removed from bodies and errors.

## Configuration

Command-line values take precedence by being copied into the same environment
configuration used by the checks.

| Variable | Meaning |
|---|---|
| `ACL_API_BASE` | API root, usually ending in `/v1` |
| `ACL_API_KEY` | bearer token |
| `ACL_MODEL` | exact model id expected from `GET /models` |
| `ACL_TIMEOUT` | request timeout in seconds; default `60` |

Legacy `MCS_*` variables and `CSCS_SERVING_API` remain supported for the
inherited full model suite. The old `mcs` executable is also retained as an
alias, but new usage should call `agent-compat`.

## Full model suite

The upstream deterministic model conformance suite remains available:

```bash
agent-compat \
  --profile model \
  --base-url "$BASE_URL" \
  --model "$MODEL"

agent-compat \
  --profile model \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --capability tools,streaming
```

Use `--spec dev` only for servers that expose non-standard `/tokenize` and
`/detokenize` endpoints. See [SPEC.md](SPEC.md) for the inherited suite design.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

ruff check .
ruff format --check .
pytest tests -q
pytest mcs/suites --collect-only -q
python -m build
```

All agent-profile tests use a local standard-library HTTP mock; development and
CI do not need a real provider key.

## Security and privacy

- Requests go directly from your machine to the base URL you provide.
- The project has no telemetry and does not upload reports.
- API keys are read from environment variables, not command-line flags.
- Known keys, authorization values, URL user-info, and common secret query
  parameters are redacted from errors and reports.
- A real endpoint receives the small prompts, tool schema, and 1×1 inline test
  image described by the checks above.

Review an endpoint's privacy and billing terms before testing it.

## Origin and license

Agent Compat Lab is derived from
[`swiss-ai/model-compatibility-suite`](https://github.com/swiss-ai/model-compatibility-suite)
at commit
[`531a52813d9be66d9fdf13c6a9d30875a770df66`](https://github.com/swiss-ai/model-compatibility-suite/commit/531a52813d9be66d9fdf13c6a9d30875a770df66).
The Apache-2.0 license and upstream notices are preserved. See [NOTICE](NOTICE)
for the modification summary.
