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
from .redaction import redact


def _record_iter_lines(resp: requests.Response, *secrets: str) -> None:
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
            recording.record("output", redact("\n".join(buf), *secrets))

    resp.iter_lines = teed


class ApiError(Exception):
    def __init__(self, status: int, body: str, *secrets: str):
        safe_body = redact(body, *secrets)
        super().__init__(f"HTTP {status}: {safe_body[:500]}")
        self.status = status
        self.body = safe_body


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

    def _request(
        self, method: str, path: str, body: dict | None = None, *, stream: bool = False
    ) -> requests.Response:
        """Send one raw request without retries or response normalization."""
        if body is not None:
            request_text = json.dumps(body, indent=2, ensure_ascii=False)
            recording.record("input", redact(request_text, self.config.api_key))
        try:
            resp = requests.request(
                method,
                f"{self.config.api_base}{path}",
                headers=self._headers(),
                json=body,
                stream=stream,
                timeout=self.config.timeout,
            )
        except requests.RequestException as exc:
            raise ApiError(0, str(exc), self.config.api_key) from exc
        if not stream:
            recording.record("output", redact(resp.text, self.config.api_key))
        else:
            _record_iter_lines(resp, self.config.api_key)
        return resp

    def _post(
        self, path: str, body: dict, *, stream: bool = False
    ) -> requests.Response:
        """POST ``body`` to ``path`` and optionally stream the response."""
        return self._request("POST", path, body, stream=stream)

    def _json_or_error(self, resp: requests.Response) -> dict:
        if not resp.ok:
            raise ApiError(resp.status_code, resp.text, self.config.api_key)
        try:
            return resp.json()
        except requests.JSONDecodeError as exc:
            raise ApiError(
                resp.status_code,
                f"invalid JSON response: {resp.text}",
                self.config.api_key,
            ) from exc

    def raw(self, payload: dict, *, stream: bool = False) -> requests.Response:
        """Escape hatch for error-path / malformed-request tests."""
        return self._post("/chat/completions", payload, stream=stream)

    def chat(self, messages, **kw) -> dict:
        resp = self.raw(self._payload(messages, stream=False, **kw))
        return self._json_or_error(resp)

    def stream(self, messages, **kw) -> Iterator[dict]:
        """Yield parsed SSE delta chunks until the [DONE] sentinel."""
        resp = self.raw(self._payload(messages, stream=True, **kw), stream=True)
        if not resp.ok:
            raise ApiError(resp.status_code, resp.text, self.config.api_key)
        for line in resp.iter_lines(decode_unicode=True):  # teed by raw() to record
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                return
            yield json.loads(data)

    @staticmethod
    def _sse(resp: requests.Response) -> Iterator[tuple[str | None, str]]:
        """Yield ``(event, data)`` pairs from an SSE response."""
        event = None
        data = []
        for line in resp.iter_lines(decode_unicode=True):
            if line == "":
                if data:
                    yield event, "\n".join(data)
                event, data = None, []
            elif line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:") :].lstrip())
        if data:
            yield event, "\n".join(data)

    def chat_stream_events(self, messages, **kw) -> tuple[list[dict], bool]:
        """Return Chat Completions SSE chunks and whether ``[DONE]`` arrived."""
        resp = self.raw(self._payload(messages, stream=True, **kw), stream=True)
        if not resp.ok:
            raise ApiError(resp.status_code, resp.text, self.config.api_key)
        chunks = []
        done = False
        for _, data in self._sse(resp):
            if data == "[DONE]":
                done = True
                break
            chunks.append(json.loads(data))
        return chunks, done

    def models(self) -> dict:
        """Return the OpenAI-compatible ``/models`` response."""
        return self._json_or_error(self._request("GET", "/models"))

    def response(self, input_data, **fields) -> dict:
        """Create one non-streaming Responses API response."""
        body = {"model": self.config.model, "input": input_data, **fields}
        return self._json_or_error(self._post("/responses", body))

    def response_stream(self, input_data, **fields) -> list[tuple[str | None, dict]]:
        """Return parsed events from one streaming Responses API request."""
        body = {
            "model": self.config.model,
            "input": input_data,
            "stream": True,
            **fields,
        }
        resp = self._post("/responses", body, stream=True)
        if not resp.ok:
            raise ApiError(resp.status_code, resp.text, self.config.api_key)
        events = []
        for event, data in self._sse(resp):
            if data == "[DONE]":
                continue
            events.append((event, json.loads(data)))
        return events

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
        return self._json_or_error(resp)

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
            raise ApiError(resp.status_code, resp.text, self.config.api_key)
        pl = resp.json()["choices"][0].get("prompt_logprobs")
        if not pl:
            raise ApiError(
                resp.status_code,
                "endpoint returned no prompt_logprobs",
                self.config.api_key,
            )
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
            raise ApiError(resp.status_code, resp.text, self.config.api_key)
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
            raise ApiError(resp.status_code, resp.text, self.config.api_key)
        return resp.json()["tokens"]

    def detokenize(self, tokens: list) -> str:
        """Text for ``tokens`` via the ``/detokenize`` endpoint."""
        resp = self._post("/detokenize", {"model": self.config.model, "tokens": tokens})
        if not resp.ok:
            raise ApiError(resp.status_code, resp.text, self.config.api_key)
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
