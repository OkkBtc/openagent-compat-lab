# MCS — Model Compatibility Suite

Lightweight tests that are 100% deterministic to prove the model (and system around it) will work in production. Designed to be prereq to evals and benchmarks.

Loosely based on Android's [Compatibility Test Suite (CTS)](https://source.android.com/docs/compatibility/cts): a deterministic conformance gate that certifies an implementation behaves as specified — here applied to served LLM endpoints instead of Android devices.

## Quickstart

```bash
export CSCS_SERVING_API=...   # your bearer token
curl -fsSL https://raw.githubusercontent.com/swiss-ai/model-compatibility-suite/main/run.sh | bash
```

Scope to specific areas and pick a model:

```bash
curl -fsSL https://raw.githubusercontent.com/swiss-ai/model-compatibility-suite/main/run.sh | bash -s -- --model swiss-ai/Apertus-8B-Instruct-2509
```

The default run tests the **OpenAI API surface only** (`--spec openai`):
checks that need extension endpoints the OpenAI spec does not define --
`/tokenize`, `/detokenize` (the tokenizer-roundtrip and BOS token-identity
suites) -- are excluded. Run `--spec dev` against endpoints that expose those
extensions (e.g. vLLM/SGLang directly, or a gateway that forwards them) to add
them.

Every check is **deterministic** (status codes, response schema, token counts,
substring / regex / closed-set membership) — there is intentionally no
LLM-as-judge. Semantic/quality evaluation belongs in LLM evals, not in a
functional pass/fail gate.

## Usage

One command. It runs the deterministic suites against a model and prints a
**✔/✗ capability table**, exiting non-zero on any failure.

```bash
mcs                              # OpenAI-spec checks, default model
mcs --model swiss-ai/Apertus-1.5-8B-Instruct-sft-dpo-tools
mcs --capability tools           # just one capability
mcs --spec dev                   # + checks needing /tokenize etc.
mcs --model A --model B          # compare models (table)
mcs --model A --model B --detail # + per-model failure reasons
mcs --json                       # machine-readable
```

Status per check: `✔` pass · `✗` an assertion failed (a real gap) · `⚠` the
check errored (e.g. the server returned an HTTP error) · `–` skipped. The
capabilities are: `core` · `streaming` · `tools` · `multimodal` · `multiturn` ·
`reasoning` · `robustness` (· `perf`, opt-in).

Local development:

```bash
uv venv && source .venv/bin/activate   # or python -m venv .venv && source .venv/bin/activate
make install                      # editable install (auto-detects `uv pip` / `pip`)
make run MODEL=Qwen/Qwen3.5-27B   # run the checks
make format                       # ruff auto-fix + format
make check                        # ruff lint + format check (the CI PR gate)
```

Every PR runs `ruff check` + `ruff format --check` (see `.github/workflows/ci.yml`);
keep the tree clean with `make format` before pushing.

## Example results

Default OpenAI-spec run (`mcs --model swiss-ai/Apertus-v1.5-70B --model swiss-ai/Apertus-v1.5-8B`)
against `https://api.swissai.svc.cscs.ch/v1`, 2026-08-06:

| Check                                | M1 | M2 |
|--------------------------------------|----|----|
| core_health                          | ✔  | ✔  |
| core_system                          | ✔  | ✔  |
| core_maxtokens                       | ✔  | ✔  |
| core_stop                            | ✔  | ✔  |
| core_usage                           | ✔  | ✔  |
| core_template_no_leak                | ✔  | ✔  |
| core_no_degeneration                 | ✔  | ✔  |
| core_no_degeneration_hard            | ✔  | ✔  |
| core_multi_system                    | ⚠  | ⚠  |
| core_assistant_prefill               | ✔  | ✔  |
| core_determinism                     | ✔  | ✔  |
| mm_image_small                       | ✔  | ✔  |
| mm_image_large                       | ✔  | ✔  |
| mm_image_multi                       | ✔  | ✔  |
| mm_audio_small                       | ✔  | ✔  |
| mm_audio_large                       | ✔  | ✔  |
| mm_interleaved                       | ✔  | ✔  |
| mm_no_degeneration_hard              | ✔  | ✔  |
| mm_no_degeneration_hard_audio        | ✔  | ✔  |
| mt_context                           | ✔  | ✔  |
| mt_roles                             | ✔  | ✔  |
| reason_parser_wired                  | –  | –  |
| reason_produced                      | –  | –  |
| reason_separation                    | ✔  | ✔  |
| reason_clean_channel                 | –  | –  |
| reason_answer                        | ✔  | ✔  |
| reason_stream                        | –  | –  |
| reason_tools                         | –  | –  |
| reason_nothink_no_inner_leak         | ✔  | ✔  |
| reason_nothink_no_inner_leak_sampled | ✔  | ✔  |
| reason_disabled                      | –  | –  |
| robust_specialtokens                 | ✔  | ✔  |
| robust_consecutive_role              | ✔  | ✔  |
| robust_unicode                       | ✔  | ✔  |
| robust_empty                         | ✔  | ✔  |
| robust_errors                        | ✔  | ✔  |
| eos                                  | ✔  | ✔  |
| stream_basic                         | ✔  | ✔  |
| stream_finish                        | ✔  | ✔  |
| stream_stop                          | ✔  | ✔  |
| stream_equiv                         | ✔  | ✔  |
| tools_single                         | ✔  | ✔  |
| tools_choice_required                | ✗  | ✔  |
| tools_choice_named                   | ✔  | ✔  |
| tools_stream                         | ✔  | ✔  |
| tools_none                           | ✔  | ✔  |
| tools_parallel                       | ✗  | ✗  |
| tools_multiturn                      | ✔  | ✔  |
| tools_followup                       | ✔  | ✔  |
| tools_no_content_leak                | ✗  | ✗  |
| tools_arg_schema                     | ✔  | ✔  |
| tools_empty_args                     | ✔  | ✔  |
| tools_phantom                        | ✔  | ✔  |
| **passed**                           | **43** | **44** |
| **failed/broken**                    | **4**  | **3**  |
| **skipped**                          | **6**  | **6**  |

Legend: ✔ pass · ✗ fail · ⚠ broken · – skip.
M1 = `swiss-ai/Apertus-v1.5-70B` · M2 = `swiss-ai/Apertus-v1.5-8B`.
The `reason_*` skips are expected — these are the non-thinking builds. The
`core_multi_system` ⚠ is the gateway rejecting a second `system` message
(HTTP 400), not a model gap.

## Configuration

| Env var | Default |
|---------|---------|
| `MCS_API_BASE` | `https://api.swissai.svc.cscs.ch/v1` |
| `MCS_API_KEY` (or `CSCS_SERVING_API`) | — (required) |
| `MCS_MODEL` | `swiss-ai/Apertus-8B-Instruct-2509` |

## Suites

`core` · `streaming` · `tools` · `multimodal` · `multiturn` · `reasoning` ·
`robustness` · `perf` (opt-in).

Only `core` is implemented today; the rest are stubs. **See
[`SPEC.md`](./SPEC.md)** for the full specification, test catalog, open
questions to probe against the live API, and implementation milestones — it is
written so another engineer can take over the build from it directly.
