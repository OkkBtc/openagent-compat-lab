"""reasoning suite -- proves the server-side REASONING PARSER works. See SPEC.md 7.7.

Apertus 1.5 wraps chain-of-thought between `<|inner_prefix|>` ... `<|inner_suffix|>`.
The reasoning parser (vLLM `--reasoning-parser qwen3`, SGLang equivalent) splits
that raw stream into two OpenAI channels: `message.reasoning_content` (scratch
work) and `message.content` (the user-facing answer). These checks prove the
split is correct -- non-streaming, streaming, and alongside the tool parser.

No quality judging (SPEC.md 6): every assertion is structural -- a field is
present/absent, a channel is free of boundary tokens, the streamed boundary is
monotonic, a closed-set answer matches.

Launch requirement -- two distinct flags, easy to conflate:
  * `--default-chat-template-kwargs.enable_thinking true` -- sets the default for
    APERTUS 1.5's OWN `enable_thinking` chat-template kwarg (the template branches
    on it to deliberate). This is the Apertus-specific switch; off => no thinking.
  * `--reasoning-parser qwen3` -- vLLM's stream-splitter implementation; "qwen3"
    is just the boundary-format name, NOT Apertus- or qwen-model-specific, and
    unrelated to the enable_thinking kwarg.
Miss the enable_thinking default and no `reasoning_content` is produced at all.

Gating: surfacing is model- AND launch-dependent (SPEC.md 7.7 / open question 2).
The `reasoning_supported` probe runs once; on an endpoint that exposes no
`reasoning_content` channel (a plain instruct model, a missing launch flag, or a
gateway dropping the field) the parser-specific checks SKIP with a clear reason
rather than failing red. `reason-separation` and `reason-answer` hold regardless
of surfacing and always run.

Budgets are generous: a reasoning model may spend hundreds of tokens thinking
before it emits the answer, so a tight max_tokens would truncate it.
"""

from __future__ import annotations

import json
import re
from typing import NamedTuple

import pytest

from mcs.client import ApiError, ChatClient

pytestmark = pytest.mark.reasoning

# Budget for reasoning prompts: thinking + answer can run long; don't truncate.
REASON_MAX_TOKENS = 4096

# A reasoning-eliciting prompt whose answer is a known sentinel (8*9 = 72). Used
# by the probe and the streaming check so the answer is deterministically
# assertable in `content`.
REASONING_PROMPT = "Think step by step, then give the answer: if a train travels 60 km in 1.5 hours, what is its average speed in km/h?"
ANSWER_SENTINEL = "40"

# Raw boundary / special tokens that must never appear in EITHER channel: the
# parser is expected to consume the delimiters, not relocate them. `<|...|>`
# also catches tool scaffolding (`<|tools_prefix|>` etc.) for reason-tools.
THINK_TOKEN_RE = re.compile(
    r"<\|inner_prefix\|>|<\|inner_suffix\|>|</?think\b|<\|[^>]*\|>", re.IGNORECASE
)

# The literal boundary tokens Apertus 1.5 emits around its chain-of-thought. If
# these survive in `content` (visible only with skip_special_tokens=false) while
# `reasoning_content` stays empty, the model IS thinking but the server's
# reasoning parser isn't splitting on them. See test_reason_parser_wired.
_REASON_DELIMS = ("<|inner_prefix|>", "<|inner_suffix|>")

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}


@pytest.fixture(scope="session")
def reasoning_supported(client):
    """Probe once: does the endpoint surface a separate `reasoning_content`
    channel? Returns the probe response (reused by reason-produced to avoid a
    second call). SKIP (not fail) the parser-specific checks when no channel is
    surfaced -- a plain model or a gateway that drops the field is a visible
    skip, not a red failure (SPEC.md 7.7 / section 8)."""
    resp = client.chat(
        [{"role": "user", "content": REASONING_PROMPT}], max_tokens=REASON_MAX_TOKENS
    )
    rc = ChatClient.reasoning_content(resp)
    if not (rc and rc.strip()):
        pytest.skip(
            "endpoint surfaces no reasoning_content channel "
            "(plain model, or the gateway drops the field)"
        )
    return resp


def test_reason_parser_wired(client):
    """reason-parser-wired: if the model emits reasoning delimiters, the server's
    reasoning parser must actually split on them.

    Independent of the `reasoning_supported` probe on purpose -- it targets the
    exact failure that probe hides: the model DOES think (emits
    `<|inner_prefix|>...<|inner_suffix|>`) but the parser doesn't extract it, so
    `reasoning_content` is empty and the raw monologue is left in `content` (with
    the delimiters stripped as special tokens on the default decode).

    Requests with `skip_special_tokens=false` so the boundary tokens are visible.
    - reasoning_content populated            -> parser is wired  (PASS)
    - delimiters in content, reasoning empty -> parser NOT wired (FAIL)
    - no delimiters at all                   -> not a thinking build (SKIP)
    """
    try:
        resp = client.chat(
            [{"role": "user", "content": REASONING_PROMPT}],
            max_tokens=REASON_MAX_TOKENS,
            extra={"skip_special_tokens": False},
        )
    except ApiError as exc:
        pytest.skip(f"endpoint rejected skip_special_tokens override: {exc}")
    content = ChatClient.content(resp) or ""
    reasoning = ChatClient.reasoning_content(resp) or ""
    if reasoning.strip():
        return  # parser populated the reasoning channel -> wired
    if any(tok in content for tok in _REASON_DELIMS):
        pytest.fail(
            "reasoning parser not wired to the emitted delimiters: the model "
            f"emitted {_REASON_DELIMS} but reasoning_content is empty, so the "
            f"chain-of-thought leaks into content. content[:200]={content[:200]!r}"
        )
    pytest.skip("model emitted no reasoning delimiters (not a thinking build)")


def test_reason_produced(reasoning_supported):
    """reason-produced: non-stream parser populates BOTH channels.

    reasoning_content non-empty proves the parser ran; content non-empty proves
    it did not swallow the answer into the reasoning channel."""
    resp = reasoning_supported
    reasoning = ChatClient.reasoning_content(resp) or ""
    content = ChatClient.content(resp) or ""
    assert reasoning.strip(), "reasoning_content empty -- parser produced no thinking"
    assert content.strip(), (
        "content empty -- parser swallowed the answer into reasoning_content"
    )


def test_reason_separation(client):
    """reason-separation: final content carries NO raw <think> / <|inner_*|> tokens."""
    content = (
        ChatClient.content(
            client.chat(
                [
                    {
                        "role": "user",
                        "content": "Think step by step, then give the answer: "
                        "if a train travels 60 km in 1.5 hours, what is "
                        "its average speed in km/h?",
                    }
                ],
                max_tokens=REASON_MAX_TOKENS,
            )
        )
        or ""
    )
    assert content.strip(), "empty content"
    leak = THINK_TOKEN_RE.search(content)
    assert not leak, (
        f"raw reasoning token {leak.group(0)!r} leaked into content: {content!r}"
    )


def test_reason_clean_channel(reasoning_supported):
    """reason-clean-channel: the reasoning_content channel itself is free of raw
    boundary tokens -- the parser CONSUMED the delimiters, not just relocated
    them into the other channel."""
    reasoning = ChatClient.reasoning_content(reasoning_supported) or ""
    leak = THINK_TOKEN_RE.search(reasoning)
    assert not leak, (
        f"raw boundary token {leak.group(0)!r} leaked into "
        f"reasoning_content: {reasoning!r}"
    )


def test_reason_answer(client):
    """reason-answer: a constrained arithmetic prompt yields the exact answer."""
    content = (
        ChatClient.content(
            client.chat(
                [
                    {
                        "role": "user",
                        "content": "What is 6 times 7? Reply with only the number.",
                    }
                ],
                max_tokens=REASON_MAX_TOKENS,
            )
        )
        or ""
    )
    assert "42" in content, f"expected 42 in the answer, got: {content!r}"


def test_reason_stream(client, reasoning_supported):
    """reason-stream: the streaming reasoning->answer boundary is monotonic.

    Every `delta.reasoning_content` must arrive before the first
    `delta.content`; the transition happens exactly once (no flip back to
    reasoning after content begins); the reassembled content carries the answer
    sentinel; and no raw boundary token leaks into either streamed field."""
    reasoning_parts, content_parts = [], []
    seen_content = False
    reasoning_after_content = False
    for ch in client.stream(
        [{"role": "user", "content": REASONING_PROMPT}], max_tokens=REASON_MAX_TOKENS
    ):
        # The terminal chunk carries usage with an empty `choices` list.
        for choice in ch.get("choices") or []:
            delta = choice.get("delta", {})
            rc = ChatClient.reasoning_delta(delta)
            c = delta.get("content")
            if rc:
                reasoning_parts.append(rc)
                if seen_content:
                    reasoning_after_content = True
            if c:
                content_parts.append(c)
                seen_content = True
    reasoning = "".join(reasoning_parts)
    content = "".join(content_parts)

    assert reasoning.strip(), "no streamed reasoning_content deltas"
    assert content.strip(), "no streamed content deltas"
    assert not reasoning_after_content, (
        "reasoning_content resumed after content began -- boundary is not monotonic"
    )
    assert ANSWER_SENTINEL in content, (
        f"expected {ANSWER_SENTINEL!r} in streamed content, got: {content!r}"
    )
    for name, chan in (("reasoning_content", reasoning), ("content", content)):
        leak = THINK_TOKEN_RE.search(chan)
        assert not leak, f"raw token {leak.group(0)!r} leaked into streamed {name}"


def test_reason_tools(client, reasoning_supported):
    """reason-tools: reasoning- and tool-parser cooperate.

    With a tool offered and forced, the call must land in `tool_calls` with
    JSON-parseable arguments, and NO raw tool scaffolding (`<|tools_prefix|>`
    etc.) may leak into either the content or reasoning_content channel. Skipped
    when the endpoint does not support tool calling (that gap is the `tools`
    suite's to report, not this one's)."""
    try:
        resp = client.chat(
            [
                {
                    "role": "user",
                    "content": "Think about which tool to use, then get the "
                    "weather in Zurich.",
                }
            ],
            tools=[WEATHER_TOOL],
            tool_choice="required",
            max_tokens=REASON_MAX_TOKENS,
        )
    except ApiError as exc:
        pytest.skip(f"tool calling not supported by endpoint: {exc}")

    tool_calls = resp["choices"][0]["message"].get("tool_calls") or []
    if not tool_calls:
        pytest.skip("model emitted no tool_call under force; tools unsupported")

    fn = tool_calls[0]["function"]
    assert fn["name"] == "get_weather", f"unexpected tool: {fn['name']!r}"
    json.loads(fn["arguments"])  # arguments must be JSON-parseable

    for name in ("content", "reasoning_content", "reasoning"):
        chan = resp["choices"][0]["message"].get(name) or ""
        leak = THINK_TOKEN_RE.search(chan)
        assert not leak, (
            f"tool/boundary scaffolding {leak.group(0)!r} leaked into {name}"
        )


class _StreamedTurn(NamedTuple):
    """What one streamed turn produced, per channel."""

    reasoning: str
    content: str
    tool_name: str | None
    tool_args: str
    tool_deltas: int


def _stream_turn(client, messages, **kw) -> _StreamedTurn:
    """Drain a streamed turn, accumulating every channel it emits."""
    reasoning, content, args = [], [], []
    name, tool_deltas = None, 0
    for ch in client.stream(messages, **kw):
        # The terminal chunk carries usage with an empty `choices` list.
        for choice in ch.get("choices") or []:
            delta = choice.get("delta", {})
            rc = ChatClient.reasoning_delta(delta)
            if rc:
                reasoning.append(rc)
            if delta.get("content"):
                content.append(delta["content"])
            for tc in delta.get("tool_calls") or []:
                tool_deltas += 1
                fn = tc.get("function") or {}
                if fn.get("name"):
                    name = fn["name"]
                if fn.get("arguments"):
                    args.append(fn["arguments"])
    return _StreamedTurn(
        "".join(reasoning), "".join(content), name, "".join(args), tool_deltas
    )


def _assert_streamed_weather_call(turn: _StreamedTurn, context: str) -> None:
    """The streamed turn produced a clean get_weather call, not raw markup.

    The two assertions are deliberately paired: whether the raw tool block shows
    up verbatim in `content` depends on `skip_special_tokens`, but a stream that
    never leaves the reasoning phase always fails the FIRST one -- no tool_call
    deltas at all -- however the server decodes the delimiters.
    """
    assert turn.tool_deltas, (
        f"{context}: no streamed tool_call deltas -- the tool parser never ran, so "
        f"an agent would execute nothing. content={turn.content[:200]!r}"
    )
    assert turn.tool_name == "get_weather", (
        f"{context}: unexpected streamed tool {turn.tool_name!r}"
    )
    try:
        json.loads(turn.tool_args)
    except json.JSONDecodeError as exc:
        pytest.fail(f"{context}: streamed arguments are not valid JSON ({exc})")
    for label, chan in (
        ("content", turn.content),
        ("reasoning_content", turn.reasoning),
    ):
        leak = THINK_TOKEN_RE.search(chan)
        assert not leak, (
            f"{context}: raw scaffolding {leak.group(0)!r} leaked into streamed "
            f"{label}: {chan[:200]!r}"
        )


def test_reason_tools_stream(client, reasoning_supported):
    """reason-tools-stream: reasoning- and tool-parser cooperate WHILE STREAMING.

    The streaming counterpart of reason-tools, and a different code path: the
    server decides per-delta when the deliberation ends and the tool parser takes
    over. Get that handoff wrong and the call never reaches `tool_calls` -- the
    raw `<|tools_prefix|>[...]` block streams to the user as `content` instead,
    so an MCP/agent client renders JSON and executes nothing.

    Skipped when the endpoint does not support tool calling (that gap is the
    `tools` suite's to report, not this one's).
    """
    try:
        turn = _stream_turn(
            client,
            [{"role": "user", "content": "Get the weather in Zurich. Use the tool."}],
            tools=[WEATHER_TOOL],
            max_tokens=REASON_MAX_TOKENS,
        )
    except ApiError as exc:
        pytest.skip(f"tool calling not supported by endpoint: {exc}")
    _assert_streamed_weather_call(turn, "streamed tool call")


def test_reason_tools_stream_nothink(client):
    """reason-tools-stream-nothink: a streamed tool call with NO deliberation block.

    The failure mode this targets: a reasoning parser splits on an end delimiter,
    so when the model skips the deliberation entirely and commits straight to a
    tool call, that delimiter never arrives and the stream never leaves the
    reasoning phase. Non-streaming is unaffected (the parser sees the whole
    output at once), which is why reason-tools can pass while this fails.

    Forces `enable_thinking=false` to make the no-inner-block path deterministic
    rather than hoping the model skips deliberating on its own, and
    `skip_special_tokens=false` so leaked delimiters are visible instead of
    silently stripped. Deliberately NOT gated on `reasoning_supported`: that probe
    skips exactly the non-think endpoints where this bites hardest. On an endpoint
    with no reasoning parser at all it degenerates to a plain streamed tool-call
    check, which is still a valid assertion.
    """
    try:
        turn = _stream_turn(
            client,
            [{"role": "user", "content": "Get the weather in Zurich. Use the tool."}],
            tools=[WEATHER_TOOL],
            max_tokens=REASON_MAX_TOKENS,
            extra={
                "chat_template_kwargs": {"enable_thinking": False},
                "skip_special_tokens": False,
            },
        )
    except ApiError as exc:
        pytest.skip(
            "endpoint rejected tool calling or the enable_thinking / "
            f"skip_special_tokens overrides: {exc}"
        )
    _assert_streamed_weather_call(turn, "streamed tool call with thinking disabled")


def test_reason_tools_stream_resumed(client, reasoning_supported):
    """reason-tools-stream-resumed: a streamed SECOND call after a round-trip.

    Apertus 1.5 holds the deliberation block open ACROSS tool calls, so a turn
    resumed after a tool result can start mid-deliberation -- a different initial
    parser state from the fresh turn reason-tools-stream covers. Getting it wrong
    ends the reasoning phase immediately and hands the still-running deliberation
    to the tool parser, which leaks `<|inner_suffix|>` into user-visible content.

    Replays a completed round-trip (assistant `tool_calls` -> `tool` result ->
    answer), then streams a new user turn that needs a fresh call. Skipped when
    the endpoint rejects the replayed history -- that round-trip gap belongs to
    the `tools` suite.
    """
    messages = [
        {"role": "user", "content": "What's the weather in Zurich? Use the tool."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Zurich"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "14C light rain"},
        {"role": "assistant", "content": "Zurich is 14C with light rain."},
        {"role": "user", "content": "Now check London. Use the tool."},
    ]
    try:
        turn = _stream_turn(
            client, messages, tools=[WEATHER_TOOL], max_tokens=REASON_MAX_TOKENS
        )
    except ApiError as exc:
        pytest.skip(f"endpoint rejected tool calling or the replayed round-trip: {exc}")
    _assert_streamed_weather_call(turn, "streamed follow-up call")
    args = json.loads(turn.tool_args)
    assert "london" in str(args.get("city", "")).lower(), (
        f"follow-up call did not target London: {args!r}"
    )


# An agentic system prompt plus an offered tool is the shape that provoked the
# non-think leak in the wild: a large IronClaw-style system prompt with tool
# definitions drove the non-think 8B model to emit a full
# `<|inner_prefix|>...<|inner_suffix|>` deliberation block into `content`, even
# though the deployment renders "Deliberation: disabled" (Apertus 1.5
# chat_template.jinja lines 181-186 -- non-think is signalled by that developer
# line, NOT by a prefilled inner block; the generation prompt is a bare
# `<|assistant_start|>`). The benign prompts in reason-separation /
# core-template-no-leak don't reproduce it -- the tools + system-prompt shape does.
_AGENTIC_SYSTEM = (
    "You are a secure autonomous assistant. Be concise and direct. Call tools "
    "when they would help accomplish the task. Respond directly with your final "
    "answer -- do not wrap it in any special tags or narrate your reasoning."
)


def test_reason_nothink_no_inner_leak(client):
    """reason-nothink-no-inner-leak: with thinking disabled, no deliberation
    delimiters may reach `content` -- even under an agentic system prompt with
    tools offered.

    Regression guard for the observed non-think leak: a deployment launched
    non-think (`enable_thinking=false` -> template renders "Deliberation:
    disabled") still emitted a `<|inner_prefix|>...<|inner_suffix|>` block into
    `content` when driven with a large agentic system prompt + tool definitions.
    A non-think turn must go straight to the answer with no inner block.

    Deliberately NOT gated on the `reasoning_supported` probe: that fixture skips
    exactly the non-think endpoints this check targets. Forces
    `enable_thinking=false` and `skip_special_tokens=false` so the delimiters, if
    emitted, are visible rather than silently stripped; skips if the endpoint
    rejects either override. The assertion is mode-agnostic and stays valid even
    if the endpoint ignores the toggle and keeps thinking: a wired reasoning
    parser routes the block to `reasoning_content`, leaving `content` clean --
    only a genuine leak into `content` fails it.
    """
    try:
        resp = client.chat(
            [
                {"role": "system", "content": _AGENTIC_SYSTEM},
                {"role": "user", "content": REASONING_PROMPT},
            ],
            tools=[WEATHER_TOOL],
            max_tokens=REASON_MAX_TOKENS,
            extra={
                "chat_template_kwargs": {"enable_thinking": False},
                "skip_special_tokens": False,
            },
        )
    except ApiError as exc:
        pytest.skip(f"endpoint rejected enable_thinking / skip_special_tokens: {exc}")

    content = ChatClient.content(resp) or ""
    leak = next((d for d in _REASON_DELIMS if d in content), None)
    assert not leak, (
        f"non-think deliberation leaked into content: the model emitted {leak!r} "
        f"with enable_thinking=false, so an inner-reasoning block reached `content` "
        f"instead of the turn going straight to the answer. This is the non-think "
        f"deployment emitting a deliberation block it should have skipped "
        f"(chat_template 'Deliberation: disabled'). content[:200]={content[:200]!r}"
    )


def test_reason_nothink_no_inner_leak_sampled(client):
    """reason-nothink-no-inner-leak-sampled: the temp>0 version of the guard above.

    The greedy (temp=0) check passes even on a leaky non-think endpoint, because
    the leak is a SAMPLING event: at temperature>0 the model occasionally draws
    `<|inner_prefix|>` despite "Deliberation: disabled", and with no reasoning
    parser on the non-think endpoint that block leaks raw into `content`. This is
    exactly the shape observed in the wild (temperature 0.7, agentic system
    prompt, tools offered).

    Probabilistic by nature: draws `_SAMPLES` completions at temperature 0.7 and
    fails if ANY carries an inner delimiter in `content`. A pass is not a proof of
    absence (the leak may be rarer than the sample budget), but a fail is a solid
    positive. Skips if the endpoint rejects the overrides.
    """
    _SAMPLES = 8
    for i in range(_SAMPLES):
        try:
            resp = client.chat(
                [
                    {"role": "system", "content": _AGENTIC_SYSTEM},
                    {"role": "user", "content": REASONING_PROMPT},
                ],
                tools=[WEATHER_TOOL],
                max_tokens=REASON_MAX_TOKENS,
                temperature=0.7,
                extra={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "skip_special_tokens": False,
                },
            )
        except ApiError as exc:
            pytest.skip(
                f"endpoint rejected enable_thinking / skip_special_tokens: {exc}"
            )
        content = ChatClient.content(resp) or ""
        leak = next((d for d in _REASON_DELIMS if d in content), None)
        assert not leak, (
            f"non-think deliberation leaked into content on sample {i + 1}/{_SAMPLES} "
            f"(temperature=0.7): the model emitted {leak!r} with enable_thinking=false, "
            f"so an inner-reasoning block reached `content`. content[:200]="
            f"{content[:200]!r}"
        )


def test_reason_disabled(client, reasoning_supported):
    """reason-disabled: the parser respects the think toggle.

    A per-request `chat_template_kwargs={"enable_thinking": false}` must OVERRIDE
    the server's launch default (`--default-chat-template-kwargs.enable_thinking
    true`): no reasoning channel, yet still a correct answer. Skipped when the
    endpoint rejects or ignores the kwarg (keeps thinking)."""
    try:
        resp = client.chat(
            [
                {
                    "role": "user",
                    "content": "What is 6 times 7? Reply with only the number.",
                }
            ],
            max_tokens=REASON_MAX_TOKENS,
            extra={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except ApiError as exc:
        pytest.skip(f"endpoint rejected enable_thinking kwarg: {exc}")

    reasoning = ChatClient.reasoning_content(resp)
    if reasoning and reasoning.strip():
        pytest.skip("endpoint ignores enable_thinking=false (still surfaces reasoning)")

    content = ChatClient.content(resp) or ""
    assert "42" in content, f"expected 42 with thinking disabled, got: {content!r}"
