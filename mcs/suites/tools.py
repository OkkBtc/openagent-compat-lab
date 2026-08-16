"""tools suite -- OpenAI-compatible function / tool calling. See SPEC.md 7.4.

Deterministic structural checks only (see SPEC.md section 6): tool-call shape,
JSON-parseable `arguments`, required-key presence, closed-set sentinels. No
LLM-as-judge.

Target model: a tool-capable Apertus, e.g.
    --model swiss-ai/Apertus-1.5-8B-Instruct-sft-dpo-tools
The whole suite HARD-FAILS (via the `tools_supported` probe) when the configured
model does not emit tool_calls -- no silent skips.

Probed wire behavior (2026-06, sft-dpo-tools model):
  - forced call -> finish_reason="tool_calls", message.content=null,
    tool_calls[0].function.{name, arguments(JSON string)}
  - tool_choice "required" and {"function":{"name":...}} both honored
  - streamed tool calls arrive as deltas; the final chunk carries usage with an
    EMPTY choices list -- callers must guard `choices`
  - PARALLEL calls: FAIL -- a 2-target prompt yields a single call
  - MULTI-TURN loop: FAIL -- echoing an assistant tool_calls message back 400s
    with "can only concatenate str (not dict) to str" (server chat-template bug)
  - TOOL-MARKUP LEAK: FAIL -- with an agentic system prompt the model emits its
    tool intent as `<info>`/`<bash>` text in `content` with EMPTY tool_calls,
    instead of a structured call (breaks opencode etc.). See SPEC.md 7.4 / 7.8.
"""

import json
import re

import pytest

from mcs.client import ApiError, ChatClient

pytestmark = pytest.mark.tools

# Tool-orchestration / special-token markup that must never appear in
# user-visible content -- the model's tool intent belongs in `tool_calls`, not
# as literal tags. Catches the Apertus-1.5 `<info>...</info>` / `<bash>...</bash>`
# leak and any `<|...|>` / <think> special-token leak (SPEC.md line 148-149).
TOOL_MARKUP_RE = re.compile(
    r"<\|[^>]*\|>"
    r"|</?(?:info|bash|tool_call|tool_calls|function_call|think"
    r"|inner_prefix|inner_suffix)\b",
    re.IGNORECASE,
)


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_code",
        "description": "Look up a secret numeric code by name.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
}

# A tool with NO parameters -- exercises the empty-arguments path of the parser.
PING_TOOL = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "Ping the server. Takes no arguments.",
        "parameters": {"type": "object", "properties": {}},
    },
}


def _tool_calls(resp: dict) -> list:
    """tool_calls for choice 0, normalized to a list (never None)."""
    return resp["choices"][0]["message"].get("tool_calls") or []


@pytest.fixture(scope="session")
def tools_supported(client):
    """Probe once: does the model emit a tool_call for an obvious NATURAL prompt?

    Probes with the DEFAULT tool_choice (auto), not `required`, on purpose: a
    weather question with a weather tool offered should elicit a call from any
    tool-capable model. Forcing (`tool_choice="required"`) is itself a separate,
    breakable code path -- e.g. the Apertus `-THINKING` endpoint returns
    finish_reason="tool_calls" but an EMPTY call list under force while auto works
    fine -- so gating on the forced path would mis-report the whole suite as
    broken when natural calling is healthy. The forced path is exercised (and
    allowed to fail on its own) by tools-choice-required / tools-choice-named.

    HARD FAIL (not skip) when even natural calling produces nothing -- pointing
    this suite at a non-tools model is an error so the gate goes red, never
    silently green. Run it against a tool-capable build (see module doc).
    """
    try:
        resp = client.chat(
            [
                {
                    "role": "user",
                    "content": "What is the weather in Paris? Use the tool.",
                }
            ],
            # generous budget: a reasoning model may think before it tool-calls
            tools=[WEATHER_TOOL],
            max_tokens=256,
        )
    except ApiError as e:
        pytest.fail(
            f"model {client.config.model!r} rejected tool calling "
            f"({e.status}): {e.body[:200]}"
        )
    assert _tool_calls(resp), (
        f"model {client.config.model!r} produced no tool_calls for a natural "
        f"tool prompt (auto tool_choice)"
    )
    return True


def test_tools_single(client, tools_supported):
    """tools-single: forced use -> expected name + schema-valid arguments."""
    resp = client.chat(
        [{"role": "user", "content": "What's the weather in Paris? Use the tool."}],
        tools=[WEATHER_TOOL],
        max_tokens=256,
    )
    calls = _tool_calls(resp)
    assert calls, "expected a tool call, got none"
    fn = calls[0]["function"]
    assert fn["name"] == "get_weather", f"unexpected tool: {fn['name']!r}"
    args = json.loads(fn["arguments"])  # must be JSON-parseable
    assert "city" in args, f"required key 'city' missing: {args!r}"


def test_tools_choice_required(client, tools_supported):
    """tools-choice (required): tool_choice='required' forces a call even when
    the prompt does not obviously call for one."""
    resp = client.chat(
        [{"role": "user", "content": "Just say hello."}],
        tools=[WEATHER_TOOL],
        tool_choice="required",
        max_tokens=128,
    )
    assert _tool_calls(resp), "tool_choice='required' did not force a call"
    assert resp["choices"][0]["finish_reason"] == "tool_calls"


def test_tools_choice_named(client, tools_supported):
    """tools-choice (named): a specific function name forces THAT function."""
    resp = client.chat(
        [{"role": "user", "content": "Hello there."}],
        tools=[WEATHER_TOOL, LOOKUP_TOOL],
        tool_choice={"type": "function", "function": {"name": "lookup_code"}},
        max_tokens=128,
    )
    calls = _tool_calls(resp)
    assert calls, "named tool_choice did not produce a call"
    assert calls[0]["function"]["name"] == "lookup_code", (
        f"forced wrong function: {calls[0]['function']['name']!r}"
    )


def test_tools_stream(client, tools_supported):
    """tools-stream: streamed tool-call arg deltas accumulate to parseable JSON.

    The final chunk carries usage with an empty `choices` list -- guarded here.
    """
    name, args = None, ""
    saw_delta = False
    for chunk in client.stream(
        [{"role": "user", "content": "Weather in Paris? Use the tool."}],
        tools=[WEATHER_TOOL],
        max_tokens=256,
    ):
        for choice in chunk.get("choices") or []:
            for tc in choice.get("delta", {}).get("tool_calls") or []:
                saw_delta = True
                fn = tc.get("function", {})
                if fn.get("name"):
                    name = fn["name"]
                if fn.get("arguments"):
                    args += fn["arguments"]
    assert saw_delta, "no streamed tool_call deltas"
    assert name == "get_weather", f"unexpected streamed tool: {name!r}"
    parsed = json.loads(args)  # accumulated deltas must form valid JSON
    assert "city" in parsed, f"required key 'city' missing: {parsed!r}"


def test_tools_none(client, tools_supported):
    """tools-none: tools offered but prompt irrelevant -> plain content, no call."""
    resp = client.chat(
        [{"role": "user", "content": "Reply with exactly the word: hello"}],
        tools=[WEATHER_TOOL],
        max_tokens=64,
    )
    assert not _tool_calls(resp), "model called a tool for an irrelevant prompt"
    content = ChatClient.content(resp)
    assert content and content.strip(), "expected normal content, got none"


def test_tools_parallel(client, tools_supported):
    """tools-parallel: a 2-target prompt must yield >=2 calls.

    HARD FAIL when the model emits a single call. Parallel calling is currently
    unsupported on the -tools build (2-target prompt -> 1 call), so this fails
    today by design -- the gap is a red failure, not a hidden skip. See SPEC.md
    7.4 / open question 3.
    """
    resp = client.chat(
        [
            {
                "role": "user",
                "content": "Get the weather in Paris AND in Tokyo. "
                "Call the tool separately for each city.",
            }
        ],
        tools=[WEATHER_TOOL],
        max_tokens=256,
    )
    calls = _tool_calls(resp)
    assert len(calls) >= 2, (
        f"expected >=2 parallel tool calls, got {len(calls)} (parallel unsupported)"
    )
    names = [c["function"]["name"] for c in calls]
    assert all(n == "get_weather" for n in names), names


def test_tools_multiturn(client, tools_supported):
    """tools-multiturn: a multi-turn tool round-trip -- call -> append the tool
    result with sentinel 4827 -> final content contains '4827'.

    HARD FAIL today: the round-trip 400s the moment the assistant tool-call turn
    is re-sent (server chat-template bug, "can only concatenate str (not dict) to
    str"), so the model never sees the sentinel. This is a real, loud failure
    until the served template is fixed -- not a skip or xfail. See SPEC.md 7.4.
    """
    messages = [
        {"role": "user", "content": "Look up the code for 'alpha'. Use the tool."}
    ]
    resp = client.chat(messages, tools=[LOOKUP_TOOL], max_tokens=256)
    call = _tool_calls(resp)[0]
    msg = resp["choices"][0]["message"]
    messages.append(
        {
            "role": "assistant",
            "content": msg.get("content"),
            "tool_calls": msg["tool_calls"],
        }
    )
    messages.append({"role": "tool", "tool_call_id": call["id"], "content": "4827"})
    final = client.chat(messages, tools=[LOOKUP_TOOL], max_tokens=256)
    content = ChatClient.content(final) or ""
    assert "4827" in content, f"sentinel from tool result missing: {content!r}"


def test_tools_followup(client, tools_supported):
    """tools-followup: a SECOND tool call after a completed round-trip.

    Replays a finished round-trip in the history -- assistant `tool_calls`
    (Zurich) -> `tool` result -> assistant answer -- then a new user turn asks
    for a different city. The model must emit a FRESH get_weather call for that
    city (closed-set sentinel: London).

    Doubles as a regression for the dict-args replay 400 ("can only concatenate
    str (not dict) to str"): the prior assistant `tool_calls` turn is echoed back
    here, which is exactly what tripped the server chat-template bug. Distinct
    from tools-multiturn, which stops at the first round-trip's final answer.
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
        {
            "role": "assistant",
            "content": "The weather in Zurich is 14C with light rain.",
        },
        {"role": "user", "content": "Now what about London?"},
    ]
    resp = client.chat(messages, tools=[WEATHER_TOOL], max_tokens=256)
    calls = _tool_calls(resp)
    assert calls, (
        "expected a follow-up tool call for London, got none; "
        f"content={ChatClient.content(resp)!r}"
    )
    assert calls[0]["function"]["name"] == "get_weather", calls[0]["function"]["name"]
    args = json.loads(calls[0]["function"]["arguments"])
    assert "london" in str(args.get("city", "")).lower(), (
        f"follow-up call did not target London: {args!r}"
    )


# Agentic system prompt that triggers the Apertus-1.5 `-tools` leak: instead of a
# structured tool call, the model emits `<info>...</info>` / `<bash>...</bash>`
# tool intent as plain content with EMPTY tool_calls -- so agents (opencode etc.)
# never execute anything. See module docstring and SPEC.md 7.4 / 7.8.
_AGENT_SYS = (
    "You are an autonomous coding agent operating in a terminal. You have a "
    "`bash` tool. When the user asks you to inspect or change the filesystem, "
    "CALL the bash tool with the command. Do not describe the command in prose."
)
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command in the shell and return its output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run"}
            },
            "required": ["command"],
        },
    },
}


def test_tools_no_content_leak(client, tools_supported):
    """tools-leak: a tool call must not leak scaffolding into user content.

    When the model makes a tool call, user-visible `content` must be empty/None
    or genuine prose -- never the bare tool name or tool-orchestration markup
    (`<info>`/`<bash>`/`<|...|>`/`<think>`). The Apertus-1.5 `-tools` build
    deterministically returns content='bash' (the tool name) beside the call,
    and under opencode's protocol leaks `<info>...</info>` markup -- i.e. the
    served tool-call parser is not cleanly separating tool scaffolding from
    content, so agents render junk / execute nothing. Qwen3.5 returns
    content=None here. SPEC.md line 148-149 / 7.8.
    """
    resp = client.chat(
        [
            {"role": "system", "content": _AGENT_SYS},
            {"role": "user", "content": "Create a directory called rob-test here."},
        ],
        tools=[BASH_TOOL],
        max_tokens=256,
    )
    calls = _tool_calls(resp)
    assert calls, (
        "agentic action produced no structured tool_call; an agent (opencode) "
        f"would execute nothing. content={ChatClient.content(resp)!r}"
    )
    content = (ChatClient.content(resp) or "").strip()
    leak = TOOL_MARKUP_RE.search(content)
    assert not leak, (
        f"tool-orchestration markup {leak.group(0)!r} leaked into content: {content!r}"
    )
    leaked_names = {c["function"]["name"].lower() for c in calls}
    assert content.lower() not in leaked_names, (
        f"bare tool name leaked into content beside the call: {content!r}"
    )


# A tool whose schema exercises the parser: a required enum and a required nested
# object with its own required field -- so "the call parsed" is not enough; the
# arguments must actually conform.
_SCHEMA_TOOL = {
    "type": "function",
    "function": {
        "name": "book_flight",
        "description": "Book a flight for a passenger.",
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {"type": "string", "description": "Destination city"},
                "cabin": {
                    "type": "string",
                    "enum": ["economy", "business", "first"],
                    "description": "Cabin class",
                },
                "passenger": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            "required": ["destination", "cabin", "passenger"],
        },
    },
}
_CABINS = {"economy", "business", "first"}


def test_tools_arg_schema(client, tools_supported):
    """tools-arg-schema: a forced tool call emits arguments that CONFORM to the
    declared schema.

    Beyond "a call happened" (which tools-single covers): the tool parser must
    produce JSON that parses AND satisfies the schema -- required keys present,
    the enum value legal, the nested object shaped right.
    """
    resp = client.chat(
        [
            {
                "role": "user",
                "content": "Book a business-class flight to Tokyo for Alice.",
            }
        ],
        tools=[_SCHEMA_TOOL],
        tool_choice="required",
        max_tokens=512,
    )
    calls = _tool_calls(resp)
    assert calls, (
        "tool_choice='required' returned no tool_calls (forced-choice path broken); "
        f"finish_reason={resp['choices'][0]['finish_reason']!r}"
    )
    fn = calls[0]["function"]
    assert fn["name"] == "book_flight", f"unexpected tool: {fn['name']!r}"
    args = json.loads(fn["arguments"])
    for key in ("destination", "cabin", "passenger"):
        assert key in args, f"missing required arg {key!r}: {args!r}"
    assert args["cabin"] in _CABINS, (
        f"cabin {args['cabin']!r} not in enum {sorted(_CABINS)}"
    )
    passenger = args["passenger"]
    assert isinstance(passenger, dict) and "name" in passenger, (
        f"passenger missing required nested 'name': {passenger!r}"
    )


def test_tools_empty_args(client, tools_supported):
    """tools-empty-args: a call to a no-parameter tool still emits a JSON object.

    The empty-arguments case has its own failure mode: some parsers emit `""` or
    `null` instead of `"{}"`, which breaks strict OpenAI clients that
    `json.loads(arguments)`. Distinct from tools-arg-schema (which needs a
    non-trivial object). Force a call to a parameterless tool and require the
    arguments to parse to a dict.
    """
    resp = client.chat(
        [{"role": "user", "content": "Call the ping tool."}],
        tools=[PING_TOOL],
        tool_choice="required",
        max_tokens=128,
    )
    calls = _tool_calls(resp)
    assert calls, (
        "tool_choice='required' returned no tool_calls (forced-choice path broken); "
        f"finish_reason={resp['choices'][0]['finish_reason']!r}"
    )
    fn = calls[0]["function"]
    assert fn["name"] == "ping", f"unexpected tool: {fn['name']!r}"
    raw = fn["arguments"]
    try:
        args = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        pytest.fail(
            f"empty-arg call arguments not JSON-parseable ({exc}); expected '{{}}', "
            f"got {raw!r} (breaks strict OpenAI clients)"
        )
    assert isinstance(args, dict), (
        f"expected a JSON object for empty args, got: {args!r}"
    )


def test_tools_phantom(client, tools_supported):
    """tools-phantom: the model must not fabricate a call to an un-offered tool.

    Only `get_weather` is offered, but the prompt asks for a flight booking (no
    such tool exists). The model may answer in prose or decline, but it must NOT
    invent a call to a function that was never provided -- a hallucinated tool name
    is a real correctness gap for any agent that dispatches on it.
    """
    resp = client.chat(
        [{"role": "user", "content": "Book me a flight to Tokyo."}],
        tools=[WEATHER_TOOL],
        max_tokens=256,
    )
    names = [c["function"]["name"] for c in _tool_calls(resp)]
    assert all(n == "get_weather" for n in names), (
        f"model fabricated a call to an un-offered tool: {names}"
    )
