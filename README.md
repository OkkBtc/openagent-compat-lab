# openagent-compat-lab

[English](README.md) | [简体中文](README.zh-CN.md)

**Probe the OpenAI-style protocol paths commonly used by Codex, Hermes Agent,
and OpenClaw before connecting an endpoint to an agent workflow.**

openagent-compat-lab runs a small, deterministic protocol test from your machine
and returns a test matrix, JSON, or a reviewable Markdown report. It
checks wire behavior—not benchmark quality—and exits non-zero when a required
behavior is missing.

Example output:

```text
Agent compatibility matrix for provider/model-name
  endpoint: https://provider.example/v1

  profile    passed  failed  duration       result
  ---------- ------- ------- ------------ ----------
  codex       8/8          0   1842.6 ms   passed
  hermes      8/8          0   1604.1 ms   passed
  openclaw    8/8          0   1719.8 ms   passed
```

## Why this exists

A successful `curl /chat/completions` proves only that one request returned a
response. Agents depend on more fragile protocol details:

- exact tool-call IDs and JSON arguments;
- assistant → tool → assistant role ordering;
- a second turn that consumes the tool result instead of calling the tool again;
- streamed tool-call fragments that must be merged by index;
- Responses API typed items, `call_id`, and `previous_response_id`;
- structured-output fields that differ between Chat Completions and Responses;
- correct SSE termination and non-text inputs.

openagent-compat-lab exercises those paths with fixed prompts and fake local tool
results. It never executes a shell command, reads your repository, or calls a
real order/weather service.

## What this project adds beyond upstream

This is a focused derivative, not an unchanged mirror. The upstream suite tests
broad model behavior; openagent-compat-lab adds an agent-integration acceptance
layer:

- distinct Codex, Hermes, and OpenClaw protocol profiles;
- stateful tool-result round trips instead of stopping at the first tool call;
- Responses `call_id` pairing and `previous_response_id` continuation;
- strict Chat Completions assistant/tool/assistant role ordering;
- streamed parallel tool-call reconstruction by index and ID;
- a one-command three-agent compatibility matrix;
- per-check timing plus JSON, Markdown, and JUnit reports;
- optional fail-fast runs for cost-sensitive checks and CI;
- credential redaction, explicit no-auth mode, and offline regression tests.

## Profiles

The named profiles are intentionally different—not labels over the same tests.

| Profile | API path | Protocol-specific coverage |
|---|---|---|
| `codex` | Responses API | typed output items, `response.completed`, forced function calls, `function_call_output` paired by `call_id`, and continuation through `previous_response_id` |
| `hermes` | Chat Completions | strict `assistant(tool_calls) → tool(tool_call_id) → assistant(text)` round trip and `finish_reason: tool_calls` |
| `openclaw` | Chat Completions | two parallel tool calls streamed in fragments and reconstructed by `index`, ID, name, and JSON arguments |
| `generic` | Chat Completions | fast baseline for models, text, SSE `[DONE]`, tools, optional arguments, JSON Schema, and image detail |
| `all` | Three named-agent paths | runs `codex`, `hermes`, and `openclaw`, then emits one compatibility matrix |
| `model` | Upstream full suite | retains the broader Model Compatibility Suite for model-level conformance |

Every named-agent profile also verifies `GET /models`, basic text generation,
forced tools, omission of an optional tool argument, strict JSON Schema output,
stream termination, and a 1×1 inline image with `detail: original`.

Passing means the tested paths worked at that moment. It does not certify answer
quality, every feature of the named agent, provider uptime, or production safety.
The profiles do not launch the named clients or verify their end-to-end setup,
authentication, or behavior against a real task.

## Quick start

Requires Python 3.10 or newer.

```bash
git clone https://github.com/OkkBtc/openagent-compat-lab.git
cd openagent-compat-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Keep the provider key in an environment variable so it does not appear as a CLI
argument. Load real keys with your shell's secret manager or a non-echoing
prompt if shell-history retention is a concern:

```bash
export ACL_API_KEY="your-provider-key"

agent-compat \
  --profile all \
  --base-url https://provider.example/v1 \
  --model provider/model-name
```

The base URL should be the API root to which `/models`, `/chat/completions`, and
`/responses` can be appended. For most providers it ends in `/v1`.

Run only one integration when you already know the target agent:

```bash
agent-compat --profile codex \
  --base-url "$BASE_URL" \
  --model "$MODEL"

agent-compat --profile hermes \
  --base-url "$BASE_URL" \
  --model "$MODEL"

agent-compat --profile openclaw \
  --base-url "$BASE_URL" \
  --model "$MODEL"
```

For a local endpoint that intentionally has no bearer token:

```bash
agent-compat \
  --profile generic \
  --base-url http://127.0.0.1:11434/v1 \
  --model qwen3 \
  --allow-no-auth
```

An empty key is rejected unless `--allow-no-auth` is explicit. This catches a
surprisingly common configuration mistake before any request is sent.

To stop after the first failed protocol assertion or runtime error:

```bash
agent-compat --profile all \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --fail-fast
```

With `--profile all`, fail-fast also skips the remaining profiles after one
profile stops. This can reduce paid requests and CI wait time when an early
failure already makes the run unusable. The option applies to agent profiles,
not the inherited `model` suite.

## Reports and CI

Print machine-readable JSON:

```bash
agent-compat --profile all \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --json > compat.json
```

The JSON contains a matrix summary, every check's status and failure detail, and
`duration_ms` at both check and profile level.

Write a Markdown report while keeping the console matrix:

```bash
agent-compat --profile all \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --markdown compat-report.md
```

Reports may contain provider/model identifiers and response details. Review and
redact them before sharing outside your team.

Write standard JUnit XML for GitHub Actions, GitLab, Jenkins, or another CI
test-report viewer:

```bash
agent-compat --profile all \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --junit compat-results.xml
```

The `all` profile writes one test suite per named agent. Protocol mismatches are
reported as JUnit failures, while transport or runtime problems are reported as
errors, so CI dashboards keep the two cases distinct.

Exit status is `0` only when every selected check passes. `FAIL` means the
endpoint returned a valid response that violated the asserted contract;
`BROKEN` means transport, HTTP, JSON, or another runtime error prevented the
contract from being evaluated. Either produces exit status `1`.

Optional redacted wire records make provider debugging easier:

```bash
agent-compat --profile hermes \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --record-responses ./compat-wire
```

The destination must not already exist. Files are grouped by profile and check,
and multi-request round trips are recorded in order. Known credentials,
authorization values, URL user-info, and common secret query parameters are
redacted. Treat wire records as sensitive anyway: a provider could echo data
that the redactor does not know is private.

## Configuration

Flags are copied into the same environment-backed configuration used by every
probe.

| Variable | Meaning |
|---|---|
| `ACL_API_BASE` | API root, usually ending in `/v1` |
| `ACL_API_KEY` | bearer token; no CLI key flag is provided intentionally |
| `ACL_MODEL` | exact model ID expected from `GET /models` |
| `ACL_TIMEOUT` | per-request timeout in seconds; default `60` |

Legacy `MCS_*` variables and `CSCS_SERVING_API` remain supported for the
inherited suite. The old `mcs` executable is retained as an alias, but new usage
should call `agent-compat`.

## Requests, privacy, and billing

- Requests travel directly from your computer to the base URL you provide.
- The project has no telemetry, hosted backend, account system, or report upload.
- API keys are read from the environment and Authorization headers are never
  written to wire records.
- No automatic retries are performed, so a failure is not silently hidden and a
  probe is not unexpectedly billed twice.
- A successful `all` run makes 26 HTTP requests: 23 small model generations and
  three `/models` reads. A failing round trip can end early; `--fail-fast` also
  skips later checks and, for `all`, later profiles.
- Prompts, a fake tool schema/result, and one tiny inline test image are sent to
  the provider. Review that provider's retention and privacy terms first.
- Providers may charge for every generation. Check their current pricing and
  rate limits before running a matrix repeatedly in CI.

The tool schemas use invented order and weather data. Do not replace them with
production secrets when filing an issue.

## Full model suite

The upstream deterministic model conformance suite remains available:

```bash
agent-compat --profile model \
  --base-url "$BASE_URL" \
  --model "$MODEL"

agent-compat --profile model \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --capability tools,streaming
```

Use `--spec dev` only for servers that expose non-standard `/tokenize` and
`/detokenize` endpoints. See [SPEC.md](SPEC.md) for the inherited suite design.

## Troubleshooting

| Symptom | Likely meaning |
|---|---|
| target model is not listed | the configured model ID differs from `/models`, or the endpoint filters visibility by key |
| stream ended without `[DONE]` | Chat Completions SSE framing is incomplete |
| no `response.completed` | the Responses stream did not emit the required typed completion event |
| tool result produced another tool call | the endpoint did not consume the assistant/tool history correctly |
| `call_id`/`previous_response_id` error | the endpoint implements only part of the Responses continuation contract |
| JSON decode failure in streamed tools | tool argument fragments were missing, reordered, or malformed |
| HTTP 404 on `/responses` | the provider likely supports Chat Completions only; test `hermes`, `openclaw`, or `generic` separately |

Use `--record-responses` with a new directory when the one-line detail is not
enough. Reports and errors cap long provider bodies to keep accidental leakage
small.

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

The focused agent tests use a local standard-library HTTP server and need no
network connection or real provider key. CI runs lint, all offline agent tests,
full-suite collection, and wheel/source-distribution builds.

## Scope and roadmap

The project deliberately stays small: deterministic protocol acceptance, clear
evidence, and safe diagnostics. It is not an agent runtime or a benchmark
leaderboard. Good next additions should come with a reproducible provider gap
and an offline regression test.

Issues and focused pull requests are welcome. Please include the selected
profile, redacted failure detail, provider/model identifier, and whether the
failure is reproducible.

## Origin and license

openagent-compat-lab is derived from
[`swiss-ai/model-compatibility-suite`](https://github.com/swiss-ai/model-compatibility-suite)
at commit
[`531a52813d9be66d9fdf13c6a9d30875a770df66`](https://github.com/swiss-ai/model-compatibility-suite/commit/531a52813d9be66d9fdf13c6a9d30875a770df66).

The upstream Apache-2.0 license and notices are preserved. Agent-specific
profiles, stateful tool round trips, Responses API probes, the compatibility
matrix, credential redaction, timing, reports, and offline protocol tests are
project modifications. See [NOTICE](NOTICE) for the precise attribution.

This is an independent community project. It is not affiliated with or endorsed
by OpenAI, Hermes Agent, OpenClaw, or the upstream maintainers.
