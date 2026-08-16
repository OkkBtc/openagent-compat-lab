# MCS (Model Compatibility Suite) — Upstream Specification

> This document describes the full model suite inherited from
> `swiss-ai/model-compatibility-suite` at commit
> `531a52813d9be66d9fdf13c6a9d30875a770df66`. openagent-compat-lab's focused
> agent profiles and current usage are documented in `README.md`.

> Handoff spec. This document is authoritative: another engineer (human or
> Claude) should be able to implement the full suite from this file alone.
> When in doubt, follow this spec; if reality (the live API) contradicts it,
> update this spec in the same PR so it stays the source of truth.

## 1. Purpose

The sibling repo `apertus-omni-tokenizer` validates tokenizer **artifacts at
rest** (md5 of `tokenizer.json`, chat template, etc. via `validate_model.sh`).

This repo validates the **served model's runtime behavior** — that a hosted
Apertus endpoint actually exercises every functional path the chat template
defines: streaming, tool calling, multimodal content, multi-turn state,
reasoning blocks, and robustness against chat-template injection.

The two are complementary: `validate_model.sh` proves the files are correct;
this proves the running model behaves correctly.

## 2. Locked design decisions

These were decided up front. Do not re-litigate without a reason.

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Repo | Standalone sibling repo `../mcs` | Separates runtime-behavior tests from tokenizer-definition artifacts. |
| HTTP client | **`requests` only** (no `openai` SDK) | Tests the raw OpenAI-compatible wire format (SSE framing, `tool_calls` JSON) with no SDK abstraction hiding bugs. Keeps `curl \| bash` bootstrap light. |
| Assertion depth | **Deterministic structural checks only — no LLM judge** | Every check must be 100% reproducible (status, schema, token counts, substring/regex/closed-set membership, SSE framing). Semantic quality ("is the answer good") is explicitly OUT OF SCOPE and belongs in LLM evals — mixing it in makes the gate flaky and its pass/fail meaningless. |
| Bootstrap | **`run.sh` → temp venv → pip install from git → pytest** | Isolated, no system pollution, full pytest reporting, single `curl \| bash` entrypoint. |
| Test framework | **pytest** | Parametrization, markers for suite selection, JUnit XML out of the box. |

## 3. Invocation contract

Primary (remote, mirrors `validate_model.sh` in the tokenizer repo):

```bash
export CSCS_SERVING_API=...   # bearer token (also accepted: MCS_API_KEY)
curl -fsSL https://raw.githubusercontent.com/swiss-ai/model-compatibility-suite/main/run.sh | bash
```

Scoped run (args after `--` pass through to `run.sh`):

```bash
curl -fsSL .../run.sh | bash -s -- \
  --suite tools,streaming \
  --model swiss-ai/Apertus-8B-Instruct-2509
```

Local checkout:

```bash
git clone https://github.com/swiss-ai/model-compatibility-suite && cd model-compatibility-suite
pip install -e ".[dev]"
pytest                       # or: mcs --suite tools
```

### Configuration (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `MCS_API_BASE` | `https://api.swissai.svc.cscs.ch/v1` | OpenAI-compatible base URL. |
| `MCS_API_KEY` | falls back to `CSCS_SERVING_API` | Bearer token. |
| `MCS_MODEL` | `swiss-ai/Apertus-8B-Instruct-2509` | Model id sent in requests. |
| `MCS_TIMEOUT` | `120` | Per-request timeout (seconds). |

### `run.sh` responsibilities

1. Resolve config from env + flags (`--suite`, `--model`,
   `--base-url`, `--junit <path>`, `--local` to skip the git install).
2. Create a temp venv (`python3 -m venv`), `pip install` the package from the
   git repo (`pip install "git+https://github.com/swiss-ai/model-compatibility-suite@main"`),
   or `pip install -e .` when run inside a checkout.
3. Map `--suite a,b` → `pytest -m "a or b"`; default runs all non-perf suites.
4. Run pytest, print a `✔/✗` per-test summary and a final line, exit non-zero
   if any test failed (so CI / shell callers can gate on it).
5. Never print the bearer token.

## 4. Repo layout

```
mcs/
├── run.sh                      # curl|bash entrypoint (section 3)
├── SPEC.md                     # this file
├── README.md                   # quickstart, points to SPEC
├── pyproject.toml              # package "mcs", deps: requests; dev: pytest
├── conftest.py                 # fixtures: client, config
├── pytest.ini / [tool.pytest]  # markers: core, special_tokens, streaming, tools,
│                               #          multimodal, multiturn, reasoning,
│                               #          robustness, perf
├── mcs/
│   ├── __init__.py
│   ├── config.py               # Config dataclass from env/flags
│   ├── client.py               # ChatClient: chat(), stream(), raw POST helpers
│   ├── cli.py                  # mcs entrypoint (flags -> env -> pytest)
│   ├── assets/                 # tiny + large fixture image/audio files
│   └── suites/
│       ├── core.py
│       ├── special_tokens.py   # BOS/EOS ownership (apertus-program #420)
│       ├── streaming.py
│       ├── tools.py
│       ├── multimodal.py
│       ├── multiturn.py
│       ├── reasoning.py
│       └── robustness.py
└── .github/workflows/ci.yml    # lint + run suite against a test endpoint (secret)
```

## 5. Client design (`client.py`)

A thin wrapper over `requests`. No ret/ries-by-default magic; tests should see
raw behavior. Minimum surface:

```python
class ChatClient:
    def __init__(self, config: Config): ...
    def chat(
        self,
        messages,
        *,
        tools=None,
        tool_choice=None,
        max_tokens=None,
        stop=None,
        temperature=0.0,
        response_format=None,
        extra=None,
    ) -> dict:
        """POST /chat/completions, stream=False. Returns parsed JSON.
        Raises ApiError(status, body) on non-2xx."""

    def stream(self, messages, **kw) -> Iterator[dict]:
        """stream=True. Yields parsed SSE delta chunks; stops on [DONE]."""

    def raw(self, payload: dict) -> requests.Response:
        """Escape hatch for malformed-request / error-path tests."""
```

SSE parsing: split on `\n\n`, strip `data: `, ignore `[DONE]`, `json.loads`
each chunk. Keep it explicit — that framing is itself under test.

## 6. Assertion model

**Every assertion is deterministic. There is no LLM-as-judge.** A test must give
the same verdict on every run against a healthy endpoint, or it does not belong
here. Semantic quality ("is this a good answer") is OUT OF SCOPE — that is what
LLM evals are for. This suite answers "does the functional path work", not "is
the model smart".

Allowed deterministic checks:
- HTTP status and error bodies
- Response schema: `choices[0]`, `message`, `finish_reason`, `usage` fields and
  arithmetic consistency
- `tool_calls` array shape + JSON-parseable `arguments` (optionally validated
  against the declared JSON schema)
- SSE chunk framing and the `[DONE]` sentinel
- Token-count bounds (e.g. `max_tokens` honored)
- Substring / regex / **closed-set membership** in the output (constrain the
  prompt so the correct answer is a small known set — e.g. "reply with one of
  red/blue/yellow", then assert membership)
- Absence of raw special tokens (`<|...|>`, `<think>`) leaking into
  user-visible `content`

To check "the model used information X" deterministically, make X a unique
sentinel you control (a tool returns the value `4827`; assert `"4827"` appears),
rather than asking a judge whether the answer is correct.

Each test must be **independent and idempotent** (no shared server state) and
use `temperature=0` for reproducibility.

## 7. Test catalog

Every test is deterministic (see section 6). Each suite is a pytest module with
the matching marker. IDs are stable handles. Where a test needs to verify the
model "used" something, the prompt is constrained so the correct output is a
known sentinel / closed set — phrased in the "Pass criteria" column.

### 7.1 `core` — API contract
| ID | Test | Send | Pass criteria |
|----|------|------|---------------|
| core-health | Basic completion | single user msg | 200; non-empty `choices[0].message.content`; `usage` present |
| core-system | System prompt adherence | system constrains output to one of red/blue/yellow | output (lowercased, stripped) ∈ {red, blue, yellow} |
| core-maxtokens | `max_tokens` honored | `max_tokens=16` | `completion_tokens` ≤ limit (+1); `finish_reason` ∈ {length, stop} |
| core-stop | `stop` honored | `stop=["three"]` | output contains no stop string |
| core-usage | Usage accounting | any | `total == prompt + completion`, all > 0 |
| core-determinism | temp=0 stability | same req ×2 | byte-identical outputs (relax only if §9.5 proves the endpoint is nondeterministic) |
| core-no-degeneration | Coherent output on an easy prompt | "write two sentences" | no word repeats ≥6× consecutively; no word is >50% of the output |
| core-no-degeneration-hard | Hard prompt completes and stops | clinical vignette, bounded answer, `max_tokens=4096` | `finish_reason="stop"` (not `"length"`) and not degenerate — the end-to-end signature of the double-BOS runaway (apertus-program #420) |

### 7.2 `special_tokens` — BOS/EOS ownership

Mostly **token-level**: read back what the server tokenized (via `/tokenize`,
`/detokenize`, or `/completions` `prompt_logprobs`) and assert an invariant about
the special tokens at the prompt's edges. **No request in this suite sends
`add_special_tokens`** — every check reads the server's DEFAULT tokenization, since
the default is what a client that doesn't override the flag actually gets, and it is
the only behavior the server is answerable for. The per-endpoint defaults are
therefore what these checks really probe:

| | chat form | completion form |
|---|---|---|
| `/tokenize` | `False` | `True` |
| generation | `/chat/completions` → `False` | `/completions` → `True` |

All checks are **model-agnostic**: the BOS is discovered from a rendered chat prompt
(the template emits it first), and a model with no BOS (e.g. Qwen) skips.

> Background (apertus-program #420, raised by the SML eval team on vLLM 0.19):
> a chat template that hardcodes `{{ bos_token }}` plus a server that also
> auto-prepends BOS yields `<s><s>…` → text degeneration. The fix makes exactly
> one layer the BOS owner *per path*. The over-correction matters as much as the
> original bug: stripping the tokenizer's post-processor BOS fixes chat but
> leaves `/completions` and lm-eval loglikelihood paths with no BOS at all →
> train/inference mismatch. So these checks assert **exactly one**, never "at
> most one", wherever a BOS is expected.

> **Read before "fixing" a red check.** Apertus 1.5 has **two BOS owners**: the
> template emits `{{ bos_token }}`, and the tokenizer's `TemplateProcessing`
> post-processor prepends `<s>` on `add_special_tokens=True`. That dual ownership *is*
> #420. vLLM's `add_special_tokens` default, before per-model overrides, is `True` on
> **every** path except text-only chat:
>
> | model type | completion path | chat path |
> |---|---|---|
> | text-only | `True` | `False` ← the only carve-out |
> | multimodal | `True` | `True` ← mm chat loses the carve-out |
>
> Two exposed paths, each with a reproduction here: **multimodal chat** (the *server*
> renders and re-encodes) → `mm-no-degeneration-hard` in §7.5; and the **completion
> path** (the *caller* renders — OpenWebUI, lm-eval) → `bos-no-double-in-rendered-
> completion` and `bos-rendered-prompt-stops`.
>
> Which checks are red depends on **which fix ships**, and two of them encode a
> contested position. `bos-single-in-raw-tokenize` / `bos-single-in-completions` demand
> exactly 1 BOS on the raw default path — i.e. the tokenizer keeps owning it there
> (`/completions` and lm-eval loglikelihood never invoke the template and would
> otherwise lose the attention-sink token). But apertus-omni-tokenizer#18
> (template-owns) strips the post-processor's BOS, making the raw default prepend
> **zero** — those two go red *by design* if it lands, while the two reproductions go
> green. This suite does not describe settled behavior on the raw path; it makes the
> ownership visible. Update it when the ownership question is decided.

| ID | Test | Pass criteria |
|----|------|---------------|
| bos-single-in-chat | chat path: template owns the BOS. Covers a single-turn AND a multi-turn prompt (a per-turn `{{ bos_token }}` only doubles on the 2nd turn) | rendered chat prompt starts with exactly 1 BOS — 2 is the original bug, 0 means the template stopped emitting it. Subsumes the former `no-double-bos-chat` (`≤1`) |
| bos-rendered-prompt-stops | **#420 reproduction, behavioral.** The symptom the issue reported, text-only. Sends `core._HARD_PROMPT` rendered through the server's own chat template to `/completions` under the DEFAULT tokenization (which prepends a 2nd BOS). The control is `core-no-degeneration-hard`: same prompt, text-only chat, one BOS, green | `finish_reason="stop"` and not degenerate. Green here on the RED image (2026-07-10): the doubled BOS did not make this checkpoint run away, so the token-level probes — not this — are what catch #420 |
| bos-single-in-raw-tokenize | raw path, via `/tokenize`: tokenizer owns the BOS | raw DEFAULT tokenization starts with exactly 1 BOS — 0 is the over-correction, 2 the original bug |
| bos-single-in-completions | same invariant on the `/completions` DEFAULT (the endpoint that generates; a gateway may default differently than `/tokenize`). `prompt_logprobs` hides position 0, so anchor on the content: take the DEFAULT `/tokenize` of the same prompt and drop its leading BOS to get the body, then see where `body[0]` lands | `ids[1] == body[0]` → exactly 1 BOS. `ids[1] == bos_id` → doubled. `ids[1] == body[1]` → no BOS (lm-eval loglikelihood loses the attention-sink token) |
| bos-generation-matches-tokenize | **closes the proxy gap.** Every other probe reads `/tokenize`, which need not traverse the generation path — the doubling lives in the *renderer* (`default_tok_params`). `usage.prompt_tokens` from a real chat call is the only observable count of what the model consumed | `usage.prompt_tokens == len(tokenize_chat(m))`. Off-by-one = double-BOS present in **generation** but not in `/tokenize` (or vice versa). Text-only: mm `prompt_tokens` includes vision-placeholder expansion |
| bos-single-token | the BOS *string* encodes to one BOS token. Also the only check that feeds the default an already-rendered (BOS-prefixed) prompt, so it pins the server's behavior on the #420 path: the default adds its own BOS on top of the caller's, deterministically and by design | `tokenize(bos + text) == [bos_id] + tokenize(text)`; a split means template/tokenizer disagree |
| bos-consistent-identity | chat and raw paths agree | same BOS id from both. Skips when the raw path prepends no BOS (`bos-single-in-raw-tokenize` flags that) |
| bos-single-in-mm-chat | image+text chat prompt — the exact mm path #420 was reported on (vLLM restores `add_special_tokens=True` via `default_tok_params`) | exactly 1 leading BOS |
| bos-single-in-mm-chat-audio | audio+text chat prompt (same `default_tok_params` path) | exactly 1 leading BOS |
| eos | model stops on its own EOS | short bounded question → `finish_reason="stop"`, no raw control/EOS token in `content` |
| eos-not-appended-to-prompt | EOS-side mirror of the BOS checks | raw default tokenization does not append a control/EOS token (`add_eos_token=True` would put a premature stop mid-context) |

The **end-to-end** counterpart — degeneration and runaway generation, the symptom
a double-BOS actually produces — lives in `core`
(`core-no-degeneration{,-hard}`), since degeneration has causes beyond a doubled
BOS.

> Note: these checks carry the `special_tokens` marker, so `--capability core`
> and `--capability multimodal` no longer run them. The default run (all
> capabilities) is unchanged; CI gates on `--capability core,special_tokens`.

### 7.3 `streaming`
> Note (2026-06): the swissai endpoint does **not** emit a `data: [DONE]`
> sentinel — it terminates the stream with a usage chunk (`choices: []`,
> `usage: {...}`). stream-basic accepts either terminal convention.

| ID | Test | Pass criteria |
|----|------|---------------|
| stream-basic | `stream=True` yields ≥2 chunks; concatenated content non-empty; stream terminates cleanly (`[DONE]` **or** a final usage chunk) |
| stream-finish | a chunk carries a `finish_reason` |
| stream-stop | streaming respects `stop` / `max_tokens` (token bound / no stop string) |
| stream-equiv | concatenated stream == non-stream `content` for same temp=0 prompt. **NOT implemented yet** — depends on temp=0 determinism (open question 5), unconfirmed for this endpoint. |

### 7.4 `tools` — function calling (OAI tools schema; see tokenizer repo PR #3)
> Target a tool-capable model, e.g. `swiss-ai/Apertus-1.5-8B-Instruct-sft-dpo-tools`.
> The suite probes once (`tools_supported` fixture) and **hard-fails** when the
> configured model does not emit `tool_calls` when forced (no silent skips; the
> non-`-tools` builds reject `tools` with a 400 because they were launched
> without `--tool-call-parser`). Probed wire shape: forced call →
> `finish_reason="tool_calls"`, `message.content=null`,
> `tool_calls[0].function.{name, arguments(JSON string)}`.
>
> Per "fail loudly, no silent skips": broken paths are red failures (non-zero
> exit), not skips or xfails — so a CI/`run.sh` caller gates on them. Against the
> `-tools` build (2026-06-30): tools-single/choice/stream/none/multiturn/followup
> pass and tools-parallel/leak fail; against `Qwen/Qwen3.5-27B` all pass.
> (tools-multiturn now passes — the `str + dict` chat-template bug is fixed in the
> served template; see tokenizer repo PR #9.)

| ID | Test | Pass criteria |
|----|------|---------------|
| tools-single | one tool offered, prompt forces use → `tool_calls[0].function.name` == expected; `arguments` is JSON-parseable and matches the declared schema (required keys present) |
| tools-multiturn | multi-turn tool round-trip: call → append `tool` result message carrying sentinel `4827` → final `content` contains `"4827"`. **Now passes on the `-tools` build** (2026-06-30): the `"can only concatenate str (not dict) to str"` chat-template bug — which used to 400 the moment an assistant `tool_calls` turn was echoed back — is fixed in the served template (tokenizer repo PR #9). Passes on Qwen. |
| tools-followup | a SECOND call after a completed round-trip: replay assistant `tool_calls` (Zurich) → `tool` result → assistant answer, then a new user turn asks for a different city → a FRESH `get_weather` call for that city (closed-set sentinel: London). Distinct from tools-multiturn (which stops at the first answer); doubles as a regression for the dict-args replay 400. **Passes on the `-tools` build** (2026-06). |
| tools-parallel | prompt needing 2 calls → ≥2 entries in `tool_calls`. **Fails on the `-tools` build** (a 2-target prompt yields a single call; parallel unsupported). Passes on Qwen. |
| tools-leak | agentic system prompt + a `bash` tool, action request → a structured `tool_calls` entry with **no tool scaffolding leaking into `content`** (no bare tool name, no `<info>`/`<bash>`/`<\|...\|>`/`<think>` markup — SPEC line 148-149). **Fails on the `-tools` build** — it returns `content="bash"` beside the call (and under opencode's protocol leaks `<info>…</info>` with empty `tool_calls`, so agents execute nothing). Passes on Qwen (`content=null`). |
| tools-choice | `tool_choice="required"` forces a call; a specific `{"function":{"name":...}}` forces that function (both confirmed) |
| tools-stream | streamed tool-call arg deltas accumulate to JSON-parseable `arguments` (the final SSE chunk carries `usage` with an empty `choices` list — guard it) |
| tools-none | tools offered but prompt irrelevant → normal content, `tool_calls` absent/empty |

### 7.5 `multimodal`
> Input format RESOLVED (2026-06, §9.1). The endpoint takes OpenAI-style content
> parts:
> - image: `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`
> - audio: `{"type":"audio_url","audio_url":{"url":"data:audio/wav;base64,..."}}`
>   — note `audio_url` (a swissai extension), **not** OpenAI's `input_audio`.
>
> Target a multimodal Apertus, e.g.
> `swiss-ai/Apertus-1.5-8B-SFT-RL-DPO-SDPO-Mix-Less-Refuse-Feedback` (the whole
> Apertus-1.5 "omni" family reads image+audio). The suite **hard-fails** (via the
> `mm_supported` probe) when the model can't read a sentinel image — no silent
> skips. Checks are functional (modality read, well-formed, no token leak), NOT
> "is the description good".
>
> Determinism: fixtures in `mcs/assets/` embed sentinels the model
> can't guess — images render a numeric code (`4827` / `1593`), audio says a
> fixed pangram — and the tests assert the sentinel/keyword appears in `content`.
> A non-guessable numeric sentinel matters: a common word like `BANANA` can be
> hallucinated, so a pass would not prove the image was actually read.

| ID | Test | Pass criteria |
|----|------|---------------|
| mm-image-small | small image (with embedded sentinel text) → 200, well-formed, no token leak; sentinel substring present |
| mm-image-large | large image accepted (limits/resize handled) → 200, well-formed |
| mm-image-multi | 2 images, each with a distinct sentinel → both sentinels appear |
| mm-audio-small | short audio clip (saying a sentinel) → 200; sentinel substring present |
| mm-audio-large | large audio accepted → 200, well-formed |
| mm-interleaved | text+image(+audio) in one message → 200, well-formed, no token leak |
| mm-no-degeneration-hard | **the end-to-end reproduction of apertus-program #420.** Sends `core._HARD_PROMPT` — the exact prompt `core-no-degeneration-hard` sends text-only and passes — with an image attached, so generation goes through the mm path that doubles the BOS. Same prompt, same 4096 budget; the modality is the only variable | `finish_reason="stop"` (not `"length"`) and not degenerate. Red here + green on `core-no-degeneration-hard` pins the fault to the multimodal **generation** path, which the `/tokenize` probes (`bos-single-in-mm-chat`) can only proxy |
| mm-no-degeneration-hard-audio | same, with audio (`bos-single-in-mm-chat-audio` shows audio doubles too) | as above |

### 7.6 `multiturn`
| ID | Test | Pass criteria |
|----|------|---------------|
| mt-context | turn 1 states sentinel `"my code is 4827"`; turn 3 asks for it → `"4827"` in final content |
| mt-tools | tool call mid-conversation returns sentinel; a later turn's content contains it |
| mt-roles | alternating user/assistant history → 200, well-formed, no role bleed / special-token leak |

### 7.7 `reasoning` — `<think>` / `<|inner_prefix|>` path (reasoning-parser proof)
> Apertus 1.5 wraps chain-of-thought between `<|inner_prefix|>` … `<|inner_suffix|>`
> (ids 32/33; `<think>`/`</think>` alias). The server-side **reasoning parser**
> (vLLM `--reasoning-parser qwen3`, the SGLang equivalent) splits that raw stream
> into two OpenAI-compatible channels: `message.reasoning_content` (the scratch
> work) and `message.content` (the user-facing answer). These tests prove the
> parser performs that split correctly — non-streaming, streaming, and in
> cooperation with the tool parser. Surfacing RESOLVED (§9 q2): a separate
> `reasoning_content` field, NOT inline `<think>` tags in `content`.
>
> **Launch requirement.** Two independent launch flags, one per side of the
> round-trip:
> - `--default-chat-template-kwargs.enable_thinking true` — sets the default for
>   **Apertus 1.5's own `enable_thinking` chat-template kwarg** (the template
>   branches on it to emit "Deliberation: enabled"). This is the Apertus-specific
>   switch that actually makes the model deliberate; off → no thinking at all.
> - `--reasoning-parser qwen3` — selects vLLM's reasoning-parser *implementation*
>   that splits the generated stream into `reasoning_content` / `content`.
>   "qwen3" is just vLLM's name for that boundary format; it is NOT
>   Apertus-specific and has nothing to do with the `enable_thinking` kwarg.
>
> A reasoning-capable model served without the `enable_thinking` default emits no
> `reasoning_content` — so a skip here can mean a missing launch flag, not a
> model gap.
>
> The suite probes once (`reasoning_supported`). On an endpoint that surfaces no
> `reasoning_content` channel (a plain instruct model, a missing launch flag, or
> a gateway that drops the field) the parser-specific rows **skip with a clear
> reason** — so a skip, not a red fail (§8). `reason-separation` and
> `reason-answer` hold regardless of how thinking is surfaced and always run.
> `reason-disabled` sends a per-request `chat_template_kwargs={"enable_thinking":
> false}` to **override** that server default, exercising request-over-launch
> precedence (and skips if the override is not honored).

| ID | Test | Pass criteria |
|----|------|---------------|
| reason-produced | non-stream: parser emits both channels | `message.reasoning_content` present and non-empty AND `content` non-empty — the parser split both ways and did not swallow the answer into the reasoning channel |
| reason-separation | answer channel is clean | final `content` contains NO raw `<think>` / `<|inner_*|>` tokens |
| reason-clean-channel | reasoning channel is clean | `reasoning_content` itself contains NO raw `<|inner_prefix|>` / `<|inner_suffix|>` / `<think>` delimiters — the parser CONSUMED the boundary tokens, not merely relocated them |
| reason-answer | answer survives the split | closed-set prompt ("what is 6×7? reply with only the number") → exact `42` in `content` |
| reason-stream | streaming boundary is correct | for `stream=True`: every `delta.reasoning_content` chunk precedes the first `delta.content` chunk; the reasoning→answer transition happens exactly once (no flip-back); reassembled `content` carries the answer sentinel; no raw tokens leak in either streamed field |
| reason-stream-equiv | stream split == non-stream split | streamed (`reasoning_content`, `content`) concatenations equal the non-stream `message.{reasoning_content, content}` for the same temp=0 prompt. **Deferred** — depends on temp=0 determinism (§9 q5); tracks the companion non-streaming reasoning-split issue |
| reason-tools | reasoning- + tool-parser cooperation | reasoning prompt + one tool offered (forced) → the call lands in `tool_calls` with JSON-parseable `arguments`; neither `content` nor `reasoning_content` contains raw tool JSON / `<|tools_prefix|>` scaffolding. Skipped unless BOTH reasoning and tools are supported |
| reason-tools-stream | the same handoff, STREAMING | `stream=True` with a tool offered → `tool_calls` arrive as deltas whose accumulated `arguments` parse as JSON, and no scaffolding leaks into either streamed channel. A distinct code path from reason-tools: the server decides per-delta when the deliberation ends and the tool parser takes over |
| reason-tools-stream-nothink | streamed call with NO deliberation block | `enable_thinking=false` + `skip_special_tokens=false`, streamed, tool offered → real `tool_calls` deltas all the same. Targets a parser that only ends the reasoning phase on the END delimiter: when the model skips deliberating and commits straight to a call that delimiter never arrives, the stream never leaves the reasoning phase, and `<|tools_prefix|>[...]` reaches the user as `content`. Non-streaming is unaffected, so reason-tools can pass while this fails. NOT gated on the `reasoning_supported` probe — it skips exactly the non-think endpoints this targets |
| reason-tools-stream-resumed | streamed call on a RESUMED turn | replay a finished round-trip (assistant `tool_calls` → `tool` result → answer), then stream a new user turn needing a fresh call → the call arrives for the new target (closed-set: London) with no `<|inner_suffix|>` leak. Apertus 1.5 holds the deliberation block open across tool calls, so a resumed turn starts in a different parser state than a fresh one |
| reason-disabled | parser respects the think toggle | `chat_template_kwargs={"enable_thinking": false}` → `reasoning_content` absent/empty AND answer still correct. Skipped if the endpoint ignores the kwarg (keeps thinking) |

### 7.8 `robustness` — chat-template injection surface
| ID | Test | Pass criteria |
|----|------|---------------|
| robust-specialtokens | user content containing `<|assistant_end|>`, `<think>`, `<|inner_prefix|>` → 200, no rendering break, no role escape, no token leak in output |
<!-- | robust-roleinject | user text impersonating a system/assistant turn with a sentinel instruction (e.g. "say HACKED") → output does NOT contain the sentinel `HACKED` | -->
| robust-unicode | multilingual/emoji/RTL input → 200; echoes a provided Unicode sentinel intact |
| robust-empty | empty / whitespace-only content → clean handling (defined status), not a 500/hang |
| robust-errors | malformed request (bad role, missing field) → 4xx with error body, not 5xx/hang |

## 8. Reporting & exit codes

- Console: one `✔ <id>` / `✗ <id> — <reason>` line per test, grouped by suite,
  then a summary `N passed, M failed, K skipped`.
- `--junit <path>` emits JUnit XML (pytest `--junitxml`).
- `--record-responses <dir>` records every request + response to disk, one folder
  per test: `<dir>/<test-name>/<model>_input.txt` and `<model>_output.txt`
  (streamed bodies captured too; multiple calls per test appended). The model id
  is in the filename, so a multi-model comparison writes each model's I/O side by
  side per test — for eyeballing what a given model actually received/returned
  (reasoning split, tool-call leak, double-BOS) instead of re-deriving it from
  pass/fail. Passive: it never changes what is sent or asserted.
- Exit `0` iff zero failures (skips are OK). Non-zero otherwise.
- A capability/format not supported by the target model `pytest.skip(...)`s with
  a clear reason (e.g. `"model has no audio capability"`) so the omission is
  visible, not silent.

## 9. Open questions — PROBE THE LIVE API FIRST

Resolve these empirically before writing the dependent suites, then update
sections 7.5 / 7.7 with the real formats:

1. **Multimodal input format.** RESOLVED (2026-06): yes, OpenAI-style
   `content: [{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]`
   works; audio uses `{"type":"audio_url","audio_url":{"url":"data:audio/wav;base64,..."}}`
   (a swissai extension, not OpenAI's `input_audio`). Confirmed against
   `swiss-ai/Apertus-1.5-8B-SFT-RL-DPO-SDPO-Mix-Less-Refuse-Feedback`: reads a
   numeric sentinel from an image and transcribes a wav clip. See §7.5.
2. **Reasoning surfacing.** RESOLVED (2026-06): thinking is returned in a
   separate `reasoning_content` field (vLLM `--reasoning-parser qwen3`), NOT
   inline `<think>` tags in `content`. **Launch config (two distinct flags):**
   `--default-chat-template-kwargs.enable_thinking true` sets the default for
   *Apertus 1.5's own* `enable_thinking` chat-template kwarg (the Apertus switch
   that makes the model deliberate); `--reasoning-parser qwen3` selects vLLM's
   stream-splitting parser (a generic implementation name, NOT Apertus- or
   qwen-model-specific). Without the `enable_thinking` default the template emits
   no thinking, so no reasoning is produced regardless of the parser.
   Caveat: the swissai gateway was *dropping* that field (Pydantic
   `extra="ignore"`); fixed in serving-api. Now that the
   field is surfaced, §7.7 asserts the **reasoning parser's split directly**:
   both channels populated (reason-produced), each channel free of raw boundary
   tokens (reason-separation / reason-clean-channel), the streaming reasoning→
   answer boundary monotonic (reason-stream), and reasoning- + tool-parser
   cooperation (reason-tools) — all gated behind the `reasoning_supported` probe
   so they skip (not fail) on endpoints that surface no `reasoning_content`.
   reason-separation and reason-answer hold regardless of surfacing.
3. **Tool schema specifics.** RESOLVED (2026-06, `-sft-dpo-tools` build):
   OAI `tools`/`tool_calls` shape confirmed; `tool_choice: "required"` and named
   `{"function":{"name":...}}` both honored; streaming emits arg deltas (final
   chunk has empty `choices` + `usage`). **Open/broken:** parallel calls
   unsupported (2-target prompt → 1 call), and the multi-turn round-trip 400s
   when an assistant `tool_calls` turn is echoed back ("can only concatenate str
   (not dict) to str" — server chat-template bug). See §7.4; `tools-multiturn` is
   `xfail` and `tools-parallel` skips until these are fixed server-side.
4. **Capability matrix per model.** ADDRESSED: capabilities and tests are one
   thing. `mcs` runs the suites and renders a `✔/✗/⚠` table
   (`capabilities.py`), exiting non-zero on failure. Scope with `--capability
   TYPE`, compare with repeated `--model`, machine-read with `--json`. Status is
   derived from the pytest outcome: pass / fail (assertion) / broken (errored) /
   skip.
   Findings (2026-06): the `-tools` build does chat/streaming/tools/named-choice/
   tool-streaming but lacks parallel calls and breaks the multi-turn loop; the
   non-`-tools` build returns 400 `requires --tool-call-parser to be set` for any
   `tools` request (a serving-config gap, not a model gap).
5. **Determinism guarantees.** Does the endpoint produce byte-identical output
   at `temperature=0` (and honor `seed`)? If not, `core-determinism` and
   `stream-equiv` must relax to a defined tolerance — document the exact
   tolerance here once measured. Prefer constrained-output tests (closed set /
   sentinel) over free-form equality wherever possible.

## 10. Implementation milestones (suggested order)

1. **M0 skeleton** ✅ — `pyproject`, `client.py`, `config.py`, `conftest.py`,
   `run.sh`, and `core` suite. `curl | bash` runs end-to-end and exits correctly.
2. **M1 streaming + tools** ✅ — implemented (stream-equiv deferred per §9.5).
3. **M2 multiturn + reasoning** ✅ — implemented. Reasoning expanded to prove the
   parser split (reason-produced / -clean-channel / -stream / -tools / -disabled),
   gated by the `reasoning_supported` probe; reason-stream-equiv deferred (mt-tools
   covered by tools-multiturn). See suite docstrings.
4. **M3 multimodal** ✅ — implemented with fixture assets in `assets/`.
5. **M4 robustness** ✅ — injection/error paths implemented.
6. **M5 perf + CI** — CI (lint + collect + live gate) ✅; the optional `perf`
   suite is still a stub.

Every suite is now implemented (no stubs). Deferred items, each noted in its
suite docstring: `stream-equiv` and `reason-stream-equiv` (need determinism,
§9.5), `core-determinism`, `mt-tools` (covered by `tools-multiturn`), and the
optional `perf` suite. The reasoning parser-split rows (`reason-produced`,
`reason-clean-channel`, `reason-stream`, `reason-tools`, `reason-disabled`) are
implemented and gated behind the `reasoning_supported` probe.

## 11. Conventions for adding a test

- One pytest function per ID, named `test_<id_with_underscores>`, decorated
  with its suite marker.
- Pull the client/config from fixtures; never construct config inline.
- **Deterministic assertions only.** Constrain the prompt so the correct output
  is a known sentinel / closed set, then assert membership or substring. If you
  cannot make a check deterministic, it does not belong here — it is an eval.
- Keep prompts short, `temperature=0`, and self-contained (no shared state).
- Skip (don't fail) when the target model lacks a capability, with a clear
  reason.
- If a test reveals the live API differs from this spec, fix the test AND
  update the relevant spec section in the same change.

## 12. Future extension — local model launch (vLLM)

A future mode: instead of pointing at a remote API, point the suite at a model
directory in CWD and have it **launch the model locally in vLLM**, wait for
readiness, then run the exact same suite against `http://localhost:<port>/v1`.

Sketch (do not block the core suite on this):

```bash
# in a dir containing a model:
curl -fsSL .../run.sh | bash -s -- --serve .            # launch vLLM on ./ then test
curl -fsSL .../run.sh | bash -s -- --serve ./my-model --port 8000
```

Design notes for the implementer:
- `--serve <path>` makes `run.sh` start `vllm serve <path>` (or
  `python -m vllm.entrypoints.openai.api_server`) in the background, poll
  `/health` until ready (timeout), set `MCS_API_BASE=http://localhost:PORT/v1`
  and `MCS_API_KEY` to a dummy, run the suite, then tear vLLM down on exit
  (trap). Surface vLLM logs on failure.
- Because vLLM exposes the same OpenAI-compatible API, **the suites are
  unchanged** — only the bootstrap differs. This is purely a `run.sh` concern.
- This pairs naturally with the tokenizer repo's `validate_model.sh`: validate
  the files at rest, then `--serve` the same dir to validate behavior. A future
  combined entrypoint could do both.
- Open considerations: GPU availability/detection, vLLM install (heavy, make it
  an opt-in extra `pip install ".[serve]"`), port selection, multi-GPU flags,
  and how to pass a chat template / tokenizer to vLLM if not bundled.
- **Launch flags to match production.** For the `tools` and `reasoning` suites to
  exercise their paths (not just skip), the local `vllm serve` must mirror the
  deployment flags: `--tool-call-parser <name>` for tools; and for reasoning BOTH
  `--default-chat-template-kwargs.enable_thinking true` (turns on Apertus 1.5's
  own `enable_thinking` chat-template kwarg — the Apertus-specific switch) AND
  `--reasoning-parser qwen3` (vLLM's generic stream-splitter, not Apertus-specific).
  Miss the `enable_thinking` default and the template emits no thinking, so the
  whole `reasoning` suite skips.
