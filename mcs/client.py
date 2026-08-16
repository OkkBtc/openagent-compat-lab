"""Thin requests-based client for the OpenAI-compatible chat completions API.

Intentionally minimal: no automatic retries or response massaging, because the
raw wire behavior (SSE framing, tool_calls JSON, error bodies) is itself under
test. See SPEC.md section 5.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import requests

from . import recording
from .config import Config


def _record_iter_lines(resp: requests.Response) -> None:
    """Wrap ``resp.iter_lines`` so the streamed SSE body is recorded as the
    consumer drains it. Records on normal exhaustion or early close (finally),
    while the test context is still active. No-op when recording is off."""
    original = resp.iter_lines

    def teed(*args, **kwargs):
        buf = []
        try:
            for line in original(*args, **kwargs):
                if line:
                    buf.append(
                        line
                        if isinstance(line, str)
                        else line.decode("utf-8", "replace")
                    )
                yield line
        finally:
            recording.record("output", "\n".join(buf))

    resp.iter_lines = teed


class ApiError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body


class ChatClient:
    def __init__(self, config: Config):
        self.config = config

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.config.api_key:
            h["Authorization"] = f"Bearer {self.config.api_key}"
        return h

    def _payload(
        self,
        messages,
        *,
        stream,
        tools=None,
        tool_choice=None,
        max_tokens=None,
        stop=None,
        temperature=0.0,
        response_format=None,
        extra=None,
    ) -> dict:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop is not None:
            payload["stop"] = stop
        if response_format is not None:
            payload["response_format"] = response_format
        if extra:
            payload.update(extra)
        return payload

    def _post(
        self, path: str, body: dict, *, stream: bool = False
    ) -> requests.Response:
        """POST ``body`` to ``path``, recording the request and (non-stream)
        response so every endpoint the suite touches -- /tokenize, /detokenize,
        /completions -- shows up under ``--record-responses``, not just
        /chat/completions."""
        recording.record("input", json.dumps(body, indent=2, ensure_ascii=False))
        resp = requests.post(
            f"{self.config.api_base}{path}",
            headers=self._headers(),
            json=body,
            stream=stream,
            timeout=self.config.timeout,
        )
        if not stream:
            # Non-stream body is safe to read here; requests caches it so the
            # caller's .json()/.text still works.
            recording.record("output", resp.text)
        else:
            # Tee iter_lines so the SSE body is recorded as whoever consumes it
            # (client.stream() OR a test iterating raw() directly) drains it.
            _record_iter_lines(resp)
        return resp

    def raw(self, payload: dict, *, stream: bool = False) -> requests.Response:
        """Escape hatch for error-path / malformed-request tests."""
        return self._post("/chat/completions", payload, stream=stream)

    def chat(self, messages, **kw) -> dict:
        resp = self.raw(self._payload(messages, stream=False, **kw))
        if not resp.ok:
            raise ApiError(resp.status_code, resp.text)
        return resp.json()

    def stream(self, messages, **kw) -> Iterator[dict]:
        """Yield parsed SSE delta chunks until the [DONE] sentinel."""
        resp = self.raw(self._payload(messages, stream=True, **kw), stream=True)
        if not resp.ok:
            raise ApiError(resp.status_code, resp.text)
        for line in resp.iter_lines(decode_unicode=True):  # teed by raw() to record
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                return
            yield json.loads(data)

    def complete(
        self, prompt: str, *, max_tokens: int, temperature: float = 0.0
    ) -> dict:
        """Generate from a raw ``/v1/completions`` prompt (no chat template applied).

        The path OpenWebUI and lm-eval hit: the CALLER renders the chat template and
        posts the resulting string. ``add_special_tokens`` is deliberately never sent,
        so the result reflects the server's own default -- what the BOS checks are
        meant to probe (same rationale as ``tokenize_chat``)."""
        body = {
            "model": self.config.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = self._post("/completions", body)
        if not resp.ok:
            raise ApiError(resp.status_code, resp.text)
        return resp.json()

    @staticmethod
    def completion_text(response: dict) -> str:
        """The generated text from a ``/completions`` response."""
        return response["choices"][0].get("text") or ""

    def prompt_token_ids(self, prompt: str, n: int = 6) -> list:
        """Return the first ``n`` prompt token ids the server actually tokenized
        a raw ``/v1/completions`` prompt into, via ``prompt_logprobs`` (works
        through gateways that don't expose ``/tokenize``).

        The first ``prompt_logprobs`` entry is ``null`` (no logprob for the very
        first token), so position 0 is returned as ``None``; positions 1+ carry
        real ids. Raises ApiError if the endpoint doesn't return
        ``prompt_logprobs``.

        ``add_special_tokens`` is deliberately never sent, so the result reflects
        the server's own default -- what the BOS checks are meant to probe (same
        rationale as ``tokenize_chat``)."""
        body = {
            "model": self.config.model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "prompt_logprobs": 0,
        }
        resp = self._post("/completions", body)
        if not resp.ok:
            raise ApiError(resp.status_code, resp.text)
        pl = resp.json()["choices"][0].get("prompt_logprobs")
        if not pl:
            raise ApiError(resp.status_code, "endpoint returned no prompt_logprobs")
        return [int(next(iter(e))) if e else None for e in pl[:n]]

    def tokenize(self, prompt: str, add_special_tokens: bool | None = None) -> list:
        """Token ids for ``prompt`` via the ``/tokenize`` endpoint.

        By default ``add_special_tokens`` is NOT sent, so the result reflects the
        server's own default -- the behavior every client that doesn't override
        the flag actually gets, and what the BOS checks are meant to probe (same
        rationale as ``tokenize_chat``). Pass True/False explicitly only where a
        test is about the flag itself (e.g. the no-specials round-trip checks)."""
        body = {"model": self.config.model, "prompt": prompt}
        if add_special_tokens is not None:
            body["add_special_tokens"] = add_special_tokens
        resp = self._post("/tokenize", body)
        if not resp.ok:
            raise ApiError(resp.status_code, resp.text)
        return resp.json()["tokens"]

    def tokenize_chat(self, messages: list, add_generation_prompt: bool = True) -> list:
        """Token ids the server produces for ``messages`` via the ``/tokenize``
        endpoint's chat form -- the server applies its OWN chat template and then
        tokenizes the rendered string. This is the path a chat client hits, and the
        one the double-BOS bug lives on: the template emits the BOS, and if the
        server also tokenizes with ``add_special_tokens=True`` the prompt starts
        ``<s><s>...``. ``add_special_tokens`` is deliberately NOT sent, so the
        result reflects the server's own default for chat tokenization."""
        resp = self._post(
            "/tokenize",
            {
                "model": self.config.model,
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
            },
        )
        if not resp.ok:
            raise ApiError(resp.status_code, resp.text)
        return resp.json()["tokens"]

    def detokenize(self, tokens: list) -> str:
        """Text for ``tokens`` via the ``/detokenize`` endpoint."""
        resp = self._post("/detokenize", {"model": self.config.model, "tokens": tokens})
        if not resp.ok:
            raise ApiError(resp.status_code, resp.text)
        return resp.json()["prompt"]

    # -- convenience helpers used by suites ---------------------------------

    @staticmethod
    def content(response: dict) -> str | None:
        return response["choices"][0]["message"].get("content")

    # Field names different stacks use for the separate reasoning channel:
    # vLLM/SGLang use `reasoning_content`; the swissai gateway uses `reasoning`
    # (the DeepSeek-style name). Accept either, in priority order.
    _REASONING_KEYS = ("reasoning_content", "reasoning")

    @staticmethod
    def reasoning_content(response: dict) -> str | None:
        """The separate reasoning channel a reasoning-parser populates, or None.

        Surfaced under different field names by different stacks (see
        `_REASONING_KEYS`). None means no reasoning channel was surfaced at all
        (a plain model, or a gateway that drops it)."""
        msg = response["choices"][0]["message"]
        return next((msg[k] for k in ChatClient._REASONING_KEYS if msg.get(k)), None)

    @staticmethod
    def reasoning_delta(delta: dict) -> str | None:
        """The reasoning piece from a streaming `delta`, under either field name."""
        return next(
            (delta[k] for k in ChatClient._REASONING_KEYS if delta.get(k)), None
        )

    @staticmethod
    def stream_text(chunks) -> str:
        out = []
        for ch in chunks:
            # The final chunk often carries usage with an empty `choices` list.
            for choice in ch.get("choices") or []:
                delta = choice.get("delta", {})
                if delta.get("content"):
                    out.append(delta["content"])
        return "".join(out)
