"""Fast protocol checks for common coding-agent API paths."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path

from . import recording
from .capabilities import BROKEN, FAIL, PASS, Result
from .client import ChatClient
from .config import Config
from .redaction import redact

CHAT_PROFILES = {"generic", "hermes", "openclaw"}
AGENT_PROFILES = CHAT_PROFILES | {"codex"}
MATRIX_PROFILES = ("codex", "hermes", "openclaw")

_IMAGE_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "YAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)
_ORDER_PARAMETERS = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string"},
        "include_history": {"type": "boolean"},
    },
    "required": ["order_id"],
    "additionalProperties": False,
}
_CHAT_ORDER_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_order",
        "description": "Look up an order by id.",
        "parameters": _ORDER_PARAMETERS,
        "strict": True,
    },
}
_CHAT_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}
_RESPONSE_ORDER_TOOL = {
    "type": "function",
    "name": "lookup_order",
    "description": "Look up an order by id.",
    "parameters": _ORDER_PARAMETERS,
    "strict": True,
}
_RESULT_SCHEMA = {
    "type": "object",
    "properties": {"compatible": {"type": "boolean"}},
    "required": ["compatible"],
    "additionalProperties": False,
}


def _message(response: dict) -> dict:
    choices = response.get("choices")
    assert isinstance(choices, list) and choices, "missing non-empty choices array"
    message = choices[0].get("message")
    assert isinstance(message, dict), "missing choices[0].message object"
    return message


def _response_output(response: dict) -> list:
    output = response.get("output")
    assert isinstance(output, list) and output, "missing non-empty output array"
    return output


def _response_text(response: dict) -> str:
    pieces = []
    for item in _response_output(response):
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "output_text":
                pieces.append(content.get("text") or "")
    text = "".join(pieces)
    assert text, "missing Responses output_text"
    return text


def _chat_tool_call(client: ChatClient) -> dict:
    response = client.chat(
        [{"role": "user", "content": "Look up order A-100."}],
        tools=[_CHAT_ORDER_TOOL],
        tool_choice={"type": "function", "function": {"name": "lookup_order"}},
    )
    calls = _message(response).get("tool_calls")
    assert isinstance(calls, list) and calls, "forced tool produced no tool_calls"
    function = calls[0].get("function")
    assert isinstance(function, dict), "tool call is missing function"
    return function


def _responses_tool_call(client: ChatClient) -> dict:
    response = client.response(
        "Look up order A-100.",
        tools=[_RESPONSE_ORDER_TOOL],
        tool_choice={"type": "function", "name": "lookup_order"},
    )
    calls = [
        item
        for item in _response_output(response)
        if item.get("type") == "function_call"
    ]
    assert calls, "forced tool produced no function_call output"
    return calls[0]


def _check_models(client: ChatClient) -> None:
    response = client.models()
    models = response.get("data")
    assert isinstance(models, list), "GET /models response is missing data array"
    ids = {item.get("id") for item in models if isinstance(item, dict)}
    assert client.config.model in ids, (
        f"target model {client.config.model!r} is not listed"
    )


def _check_chat_basic(client: ChatClient) -> None:
    message = _message(
        client.chat([{"role": "user", "content": "Reply with: agent-compat-ok"}])
    )
    assert isinstance(message.get("content"), str), "message.content is not a string"


def _check_chat_stream(client: ChatClient) -> None:
    chunks, done = client.chat_stream_events(
        [{"role": "user", "content": "Reply with: agent-compat-ok"}]
    )
    assert chunks, "stream returned no data chunks"
    assert done, "stream ended without the [DONE] sentinel"
    assert any(chunk.get("choices") is not None for chunk in chunks), (
        "stream chunks are missing choices"
    )


def _check_chat_tool(client: ChatClient) -> None:
    function = _chat_tool_call(client)
    assert function.get("name") == "lookup_order", "forced tool name was not honored"
    arguments = json.loads(function.get("arguments") or "")
    assert arguments.get("order_id") == "A-100", "required order_id was not extracted"


def _check_hermes_tool_roundtrip(client: ChatClient) -> None:
    messages = [{"role": "user", "content": "Look up order A-100."}]
    first = client.chat(
        messages,
        tools=[_CHAT_ORDER_TOOL],
        tool_choice={"type": "function", "function": {"name": "lookup_order"}},
    )
    choice = first["choices"][0]
    assistant = _message(first)
    calls = assistant.get("tool_calls")
    assert isinstance(calls, list) and len(calls) == 1, (
        "initial assistant message must contain exactly one tool call"
    )
    call = calls[0]
    assert choice.get("finish_reason") == "tool_calls", (
        "tool request did not finish with tool_calls"
    )
    assert call.get("id"), "tool call is missing an id"
    assert assistant.get("role") == "assistant", (
        "tool call message role is not assistant"
    )

    final = client.chat(
        [
            *messages,
            assistant,
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps({"order_id": "A-100", "status": "ready"}),
            },
        ],
        tools=[_CHAT_ORDER_TOOL],
    )
    final_message = _message(final)
    assert isinstance(final_message.get("content"), str), (
        "assistant did not produce final text after the tool result"
    )
    assert not final_message.get("tool_calls"), (
        "assistant requested another tool instead of consuming the tool result"
    )


def _check_openclaw_streamed_parallel_tools(client: ChatClient) -> None:
    chunks, done = client.chat_stream_events(
        [
            {
                "role": "user",
                "content": "Look up order A-100 and the weather in Paris.",
            }
        ],
        tools=[_CHAT_ORDER_TOOL, _CHAT_WEATHER_TOOL],
        tool_choice="required",
    )
    assert done, "parallel tool stream ended without the [DONE] sentinel"

    calls = {}
    for chunk in chunks:
        choices = chunk.get("choices") or []
        if not choices:
            continue
        for part in choices[0].get("delta", {}).get("tool_calls") or []:
            index = part.get("index")
            assert isinstance(index, int), "streamed tool delta is missing its index"
            call = calls.setdefault(
                index, {"id": "", "type": "", "name": "", "arguments": ""}
            )
            if part.get("id"):
                call["id"] = part["id"]
            if part.get("type"):
                call["type"] = part["type"]
            function = part.get("function") or {}
            call["name"] += function.get("name") or ""
            call["arguments"] += function.get("arguments") or ""

    assert len(calls) == 2, "expected two indexed tool calls in the stream"
    by_name = {call["name"]: call for call in calls.values()}
    assert set(by_name) == {"lookup_order", "get_weather"}, (
        "streamed tool names could not be reconstructed"
    )
    assert all(call["id"] for call in calls.values()), (
        "a streamed tool call is missing its id"
    )
    order_args = json.loads(by_name["lookup_order"]["arguments"])
    weather_args = json.loads(by_name["get_weather"]["arguments"])
    assert order_args.get("order_id") == "A-100", "streamed order id is incorrect"
    assert weather_args.get("location") == "Paris", (
        "streamed weather location is incorrect"
    )


def _check_chat_optional_args(client: ChatClient) -> None:
    arguments = json.loads(_chat_tool_call(client).get("arguments") or "")
    assert "include_history" not in arguments, (
        "optional include_history was invented even though the user omitted it"
    )


def _check_chat_json_schema(client: ChatClient) -> None:
    response = client.chat(
        [{"role": "user", "content": "Return whether this endpoint is compatible."}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "compatibility_result",
                "strict": True,
                "schema": _RESULT_SCHEMA,
            },
        },
    )
    parsed = json.loads(_message(response).get("content") or "")
    assert isinstance(parsed.get("compatible"), bool), (
        "structured output is missing boolean compatible"
    )


def _check_chat_image_original(client: ChatClient) -> None:
    response = client.chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Acknowledge this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": _IMAGE_DATA_URL, "detail": "original"},
                    },
                ],
            }
        ]
    )
    _message(response)


def _check_responses_basic(client: ChatClient) -> None:
    _response_text(client.response("Reply with: agent-compat-ok"))


def _check_responses_stream(client: ChatClient) -> None:
    events = client.response_stream("Reply with: agent-compat-ok")
    assert events, "stream returned no SSE events"
    event_types = {event or data.get("type") for event, data in events}
    assert "response.completed" in event_types, (
        "Responses stream ended without response.completed"
    )


def _check_responses_tool(client: ChatClient) -> None:
    call = _responses_tool_call(client)
    assert call.get("name") == "lookup_order", "forced tool name was not honored"
    arguments = json.loads(call.get("arguments") or "")
    assert arguments.get("order_id") == "A-100", "required order_id was not extracted"


def _check_responses_tool_roundtrip(client: ChatClient) -> None:
    first = client.response(
        "Look up order A-100.",
        tools=[_RESPONSE_ORDER_TOOL],
        tool_choice={"type": "function", "name": "lookup_order"},
    )
    response_id = first.get("id")
    assert isinstance(response_id, str) and response_id, (
        "initial Responses result is missing its response id"
    )
    calls = [
        item for item in _response_output(first) if item.get("type") == "function_call"
    ]
    assert len(calls) == 1, "initial response must contain exactly one function_call"
    call_id = calls[0].get("call_id")
    assert isinstance(call_id, str) and call_id, "function_call is missing call_id"

    final = client.response(
        [
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps({"order_id": "A-100", "status": "ready"}),
            }
        ],
        previous_response_id=response_id,
        tools=[_RESPONSE_ORDER_TOOL],
    )
    _response_text(final)


def _check_responses_optional_args(client: ChatClient) -> None:
    arguments = json.loads(_responses_tool_call(client).get("arguments") or "")
    assert "include_history" not in arguments, (
        "optional include_history was invented even though the user omitted it"
    )


def _check_responses_json_schema(client: ChatClient) -> None:
    response = client.response(
        "Return whether this endpoint is compatible.",
        text={
            "format": {
                "type": "json_schema",
                "name": "compatibility_result",
                "strict": True,
                "schema": _RESULT_SCHEMA,
            }
        },
    )
    parsed = json.loads(_response_text(response))
    assert isinstance(parsed.get("compatible"), bool), (
        "structured output is missing boolean compatible"
    )


def _check_responses_image_original(client: ChatClient) -> None:
    response = client.response(
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Acknowledge this image."},
                    {
                        "type": "input_image",
                        "image_url": _IMAGE_DATA_URL,
                        "detail": "original",
                    },
                ],
            }
        ]
    )
    _response_output(response)


_CHAT_CHECKS = [
    ("models_auth_and_target", _check_models),
    ("chat_basic", _check_chat_basic),
    ("chat_stream_done", _check_chat_stream),
    ("chat_forced_tool", _check_chat_tool),
    ("chat_optional_args", _check_chat_optional_args),
    ("chat_json_schema", _check_chat_json_schema),
    ("chat_image_detail_original", _check_chat_image_original),
]
_CODEX_CHECKS = [
    ("models_auth_and_target", _check_models),
    ("responses_basic", _check_responses_basic),
    ("responses_stream_completed", _check_responses_stream),
    ("responses_forced_tool", _check_responses_tool),
    ("responses_optional_args", _check_responses_optional_args),
    ("responses_json_schema", _check_responses_json_schema),
    ("responses_image_detail_original", _check_responses_image_original),
]
_PROFILE_CHECKS = {
    "generic": _CHAT_CHECKS,
    "hermes": [
        *_CHAT_CHECKS,
        ("hermes_tool_result_roundtrip", _check_hermes_tool_roundtrip),
    ],
    "openclaw": [
        *_CHAT_CHECKS,
        ("openclaw_streamed_parallel_tools", _check_openclaw_streamed_parallel_tools),
    ],
    "codex": [
        *_CODEX_CHECKS,
        ("responses_tool_result_roundtrip", _check_responses_tool_roundtrip),
    ],
}


def _run_one(
    profile: str,
    name: str,
    check: Callable[[ChatClient], None],
    client: ChatClient,
) -> Result:
    recording.configure(
        os.environ.get("ACL_RECORD_DIR") or os.environ.get("MCS_RECORD_DIR"),
        f"{profile}_{name}",
        redact(client.config.model, client.config.api_key),
    )
    started = time.perf_counter()
    try:
        check(client)
    except AssertionError as exc:
        return Result(
            name,
            FAIL,
            redact(exc, client.config.api_key)[:200],
            round((time.perf_counter() - started) * 1000, 1),
        )
    except Exception as exc:  # noqa: BLE001 - each probe must produce a result
        return Result(
            name,
            BROKEN,
            redact(exc, client.config.api_key)[:200],
            round((time.perf_counter() - started) * 1000, 1),
        )
    finally:
        recording.reset()
    return Result(name, PASS, "", round((time.perf_counter() - started) * 1000, 1))


def run_agent_checks(config: Config, profile: str) -> list[Result]:
    """Run deterministic protocol checks for one agent profile."""
    if profile not in AGENT_PROFILES:
        raise ValueError(f"unknown agent profile: {profile}")
    client = ChatClient(config)
    return [
        _run_one(profile, name, check, client)
        for name, check in _PROFILE_CHECKS[profile]
    ]


def run_agent_matrix(config: Config) -> dict[str, list[Result]]:
    """Run the three named-agent profiles in stable display order."""
    return {profile: run_agent_checks(config, profile) for profile in MATRIX_PROFILES}


def _safe_endpoint(config: Config) -> str:
    return redact(config.api_base, config.api_key)


def _safe_model(config: Config) -> str:
    return redact(config.model, config.api_key)


def _summary(results: list[Result]) -> dict:
    passed = sum(result.status == PASS for result in results)
    failed = len(results) - passed
    return {
        "passed": passed,
        "failed_or_broken": failed,
        "total": len(results),
        "duration_ms": round(sum(result.duration_ms for result in results), 1),
        "compatible": failed == 0,
    }


def _markdown(config: Config, profile: str, results: list[Result]) -> str:
    summary = _summary(results)
    lines = [
        "# Agent Compat Lab report",
        "",
        f"- Profile: `{profile}`",
        f"- Model: `{_safe_model(config)}`",
        f"- Endpoint: `{_safe_endpoint(config)}`",
        (
            f"- Result: **{summary['passed']} passed, "
            f"{summary['failed_or_broken']} failed/broken**"
        ),
        f"- Probe time: **{summary['duration_ms']:.1f} ms**",
        "",
        "| Check | Status | Time | Detail |",
        "|---|---|---:|---|",
    ]
    icons = {PASS: "PASS", FAIL: "FAIL", BROKEN: "BROKEN"}
    for result in results:
        detail = result.detail.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| `{result.name}` | {icons[result.status]} | "
            f"{result.duration_ms:.1f} ms | {detail} |"
        )
    lines.append("")
    return "\n".join(lines)


def _matrix_markdown(config: Config, runs: dict[str, list[Result]]) -> str:
    paths = {
        "codex": "Responses API",
        "hermes": "Chat Completions",
        "openclaw": "Chat Completions stream",
    }
    lines = [
        "# Agent Compat Lab matrix",
        "",
        f"- Model: `{_safe_model(config)}`",
        f"- Endpoint: `{_safe_endpoint(config)}`",
        "",
        "| Profile | API path | Passed | Failed/broken | Time | Verdict |",
        "|---|---|---:|---:|---:|---|",
    ]
    for profile, results in runs.items():
        summary = _summary(results)
        verdict = "COMPATIBLE" if summary["compatible"] else "INCOMPATIBLE"
        lines.append(
            f"| `{profile}` | {paths[profile]} | {summary['passed']}/"
            f"{summary['total']} | {summary['failed_or_broken']} | "
            f"{summary['duration_ms']:.1f} ms | **{verdict}** |"
        )
    for profile, results in runs.items():
        lines.extend(
            [
                "",
                f"## {profile}",
                "",
                "| Check | Status | Time | Detail |",
                "|---|---|---:|---|",
            ]
        )
        for result in results:
            detail = result.detail.replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| `{result.name}` | {result.status.upper()} | "
                f"{result.duration_ms:.1f} ms | {detail} |"
            )
    lines.append("")
    return "\n".join(lines)


def report_agent(
    config: Config,
    profile: str,
    *,
    as_json: bool = False,
    markdown_path: str | None = None,
) -> int:
    results = run_agent_checks(config, profile)
    payload = {
        "profile": profile,
        "model": _safe_model(config),
        "api_base": _safe_endpoint(config),
        "summary": _summary(results),
        "checks": [result.__dict__ for result in results],
    }
    if markdown_path:
        path = Path(markdown_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_markdown(config, profile, results), encoding="utf-8")

    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"Agent compatibility checks for {_safe_model(config)}")
        print(f"  profile:  {profile}")
        print(f"  endpoint: {_safe_endpoint(config)}\n")
        width = max(len(result.name) for result in results)
        icons = {PASS: "✔", FAIL: "✗", BROKEN: "⚠"}
        for result in results:
            print(
                f"  {icons[result.status]} {result.name:<{width}}  "
                f"{result.duration_ms:>7.1f} ms  {result.detail}"
            )
        summary = _summary(results)
        print(
            f"\n  {summary['total']} checks · {summary['passed']} passed · "
            f"{summary['failed_or_broken']} failed/broken · "
            f"{summary['duration_ms']:.1f} ms"
        )
        if markdown_path:
            print(f"  markdown: {markdown_path}")
    return 1 if any(result.status in {FAIL, BROKEN} for result in results) else 0


def report_agent_matrix(
    config: Config,
    *,
    as_json: bool = False,
    markdown_path: str | None = None,
) -> int:
    runs = run_agent_matrix(config)
    rows = []
    for profile, results in runs.items():
        rows.append({"profile": profile, **_summary(results)})
    payload = {
        "profile": "all",
        "model": _safe_model(config),
        "api_base": _safe_endpoint(config),
        "matrix": rows,
        "profiles": {
            profile: {
                "summary": _summary(results),
                "checks": [result.__dict__ for result in results],
            }
            for profile, results in runs.items()
        },
    }
    if markdown_path:
        path = Path(markdown_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_matrix_markdown(config, runs), encoding="utf-8")

    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"Agent compatibility matrix for {_safe_model(config)}")
        print(f"  endpoint: {_safe_endpoint(config)}\n")
        print("  profile    passed  failed  duration     verdict")
        print("  ---------- ------- ------- ------------ ----------")
        for row in rows:
            verdict = "compatible" if row["compatible"] else "incompatible"
            print(
                f"  {row['profile']:<10} {row['passed']:>3}/{row['total']:<3} "
                f"{row['failed_or_broken']:>7} {row['duration_ms']:>8.1f} ms "
                f"{verdict:>12}"
            )
        if markdown_path:
            print(f"\n  markdown: {markdown_path}")
    return 1 if any(not row["compatible"] for row in rows) else 0
