"""streaming suite -- SSE streaming behavior. See SPEC.md 7.2.

Deterministic structural checks only (SPEC.md 6): chunk count, stream
termination, finish_reason presence, stop-sequence honoring. The SSE framing is
itself under test, so stream-basic parses the raw wire bytes.

Note (2026-06): the swissai endpoint does NOT emit a `data: [DONE]` sentinel --
it terminates with a usage chunk (`choices: []`, `usage: {...}`). stream-basic
accepts either terminal convention. See SPEC.md 7.2.

stream-equiv (concatenated stream == non-stream content) is intentionally NOT
implemented yet -- it depends on temperature=0 determinism (open question 5),
which is unconfirmed for this endpoint.
"""

import json

import pytest

from mcs.client import ChatClient

pytestmark = pytest.mark.streaming

# Thinking-safe budget: a reasoning model streams a stripped `<think>` block
# before any `content` delta, so a tight max_tokens yields an empty concatenated
# stream. See core.py / SPEC.md 7.7.
_THINKING_MAX_TOKENS = 1024


def test_stream_basic(client):
    """stream-basic: >=2 SSE chunks, non-empty content, clean termination.

    Termination is either a `[DONE]` sentinel OR a final usage chunk (empty
    `choices` + `usage`) -- this endpoint uses the latter.
    """
    resp = client.raw(
        client._payload(
            [{"role": "user", "content": "Count from one to five."}],
            stream=True,
            max_tokens=_THINKING_MAX_TOKENS,
        ),
        stream=True,
    )
    assert resp.ok, f"stream request failed: {resp.status_code}"
    n_data, saw_terminal, content = 0, False, []
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            saw_terminal = True
            continue
        n_data += 1
        chunk = json.loads(payload)
        choices = chunk.get("choices") or []
        if not choices and chunk.get("usage"):
            saw_terminal = True  # this endpoint's terminal marker
        for choice in choices:
            piece = choice.get("delta", {}).get("content")
            if piece:
                content.append(piece)
    assert n_data >= 2, f"expected >=2 SSE chunks, got {n_data}"
    assert saw_terminal, "stream did not terminate with [DONE] or a usage chunk"
    assert "".join(content).strip(), "concatenated stream content was empty"


def test_stream_finish(client):
    """stream-finish: some chunk carries a finish_reason."""
    chunks = list(
        client.stream([{"role": "user", "content": "Say hello."}], max_tokens=32)
    )
    reasons = [
        choice.get("finish_reason")
        for ch in chunks
        for choice in (ch.get("choices") or [])
    ]
    assert any(r for r in reasons), f"no finish_reason in any chunk: {reasons}"


def test_stream_stop(client):
    """stream-stop: a streamed response honors the stop sequence, across BOTH
    output channels.

    Mirrors core-stop for the streaming path: a reasoning model can hit the stop
    string while still inside its `<think>` block, so the partial output arrives
    as `reasoning_content` deltas (or, on an endpoint that drops that channel, not
    at all) rather than `content`. Assert the stop string leaked into NEITHER
    streamed channel, and that the stop took effect -- some streamed output, or a
    `finish_reason` of 'stop'.
    """
    content_parts, reasoning_parts, finish = [], [], None
    for ch in client.stream(
        [{"role": "user", "content": "Count: one two three four five"}],
        stop=["three"],
        max_tokens=_THINKING_MAX_TOKENS,
    ):
        # The terminal chunk carries usage with an empty `choices` list.
        for choice in ch.get("choices") or []:
            delta = choice.get("delta", {})
            if delta.get("content"):
                content_parts.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    assert "three" not in content, (
        f"stop string leaked into streamed content: {content!r}"
    )
    assert "three" not in reasoning, (
        f"stop string leaked into streamed reasoning_content: {reasoning!r}"
    )
    assert content.strip() or reasoning.strip() or finish == "stop", (
        f"no streamed output in either channel and "
        f"finish_reason={finish!r} (expected 'stop')"
    )


def test_stream_equiv(client):
    """stream-equiv: streamed content == non-streamed content at temperature=0.

    `chat()`/`stream()` both send temperature=0, so the two transport paths must
    return the same answer text. A mismatch means the streaming and non-streaming
    code paths disagree (different templating/decoding), or the endpoint is not
    deterministic at temp=0 (SPEC.md open question 5) -- either is a real finding.
    Compares the user-facing `content` channel; skips if neither path produced any.
    """
    prompt = [{"role": "user", "content": "List the days of the week, one per line."}]
    # Thinking-safe budget: a reasoning model streams a stripped `<think>` block
    # before any content delta, so a tight budget yields empty content on both
    # paths and nothing to compare.
    non_stream = (
        ChatClient.content(client.chat(prompt, max_tokens=_THINKING_MAX_TOKENS)) or ""
    ).strip()
    streamed = ChatClient.stream_text(
        client.stream(prompt, max_tokens=_THINKING_MAX_TOKENS)
    ).strip()
    if not non_stream and not streamed:
        pytest.skip("no content in either path to compare")
    assert streamed == non_stream, (
        f"stream vs non-stream content differ at temp=0: stream={streamed!r}    non-stream={non_stream!r}    ."
    )
