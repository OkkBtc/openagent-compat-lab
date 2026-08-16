"""Fast protocol checks for common coding-agent API paths."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

from . import recording
from .capabilities import BROKEN, FAIL, PASS, Result
from .client import ChatClient
from .config import Config
from .redaction import redact

CHAT_PROFILES = {"generic", "hermes", "openclaw"}
AGENT_PROFILES = CHAT_PROFILES | {"codex"}

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


def _run_one(
    name: str, check: Callable[[ChatClient], None], client: ChatClient
) -> Result:
    recording.configure(
        os.environ.get("ACL_RECORD_DIR") or os.environ.get("MCS_RECORD_DIR"),
        name,
        redact(client.config.model, client.config.api_key),
    )
    try:
        check(client)
    except AssertionError as exc:
        return Result(name, FAIL, redact(exc, client.config.api_key)[:200])
    except Exception as exc:  # noqa: BLE001 - each probe must produce a result
        return Result(name, BROKEN, redact(exc, client.config.api_key)[:200])
    finally:
        recording.reset()
    return Result(name, PASS)


def run_agent_checks(config: Config, profile: str) -> list[Result]:
    """Run the seven deterministic checks for one agent profile."""
    if profile not in AGENT_PROFILES:
        raise ValueError(f"unknown agent profile: {profile}")
    checks = _CODEX_CHECKS if profile == "codex" else _CHAT_CHECKS
    client = ChatClient(config)
    return [_run_one(name, check, client) for name, check in checks]


def _safe_endpoint(config: Config) -> str:
    return redact(config.api_base, config.api_key)


def _safe_model(config: Config) -> str:
    return redact(config.model, config.api_key)


def _markdown(config: Config, profile: str, results: list[Result]) -> str:
    passed = sum(result.status == PASS for result in results)
    failed = len(results) - passed
    lines = [
        "# Agent Compat Lab report",
        "",
        f"- Profile: `{profile}`",
        f"- Model: `{_safe_model(config)}`",
        f"- Endpoint: `{_safe_endpoint(config)}`",
        f"- Result: **{passed} passed, {failed} failed/broken**",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    icons = {PASS: "PASS", FAIL: "FAIL", BROKEN: "BROKEN"}
    for result in results:
        detail = result.detail.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{result.name}` | {icons[result.status]} | {detail} |")
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
            print(f"  {icons[result.status]} {result.name:<{width}}  {result.detail}")
        passed = sum(result.status == PASS for result in results)
        failed = len(results) - passed
        print(f"\n  {len(results)} checks · {passed} passed · {failed} failed/broken")
        if markdown_path:
            print(f"  markdown: {markdown_path}")
    return 1 if any(result.status in {FAIL, BROKEN} for result in results) else 0
