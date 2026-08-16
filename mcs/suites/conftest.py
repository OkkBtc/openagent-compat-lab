"""Shared pytest fixtures + marker registration.

Lives in the *suites* directory (next to the test modules) so pytest always
loads it as the tests' conftest -- independent of where the rootdir lands. That
matters for the installed `mcs` / curl|bash flow: a conftest one
level up (the package root) is NOT loaded when the rootdir is the suites dir,
which surfaced as "fixture 'client' not found".

All assertions in this suite are deterministic (see SPEC.md section 6); there is
deliberately no LLM-as-judge. Semantic/quality evaluation belongs in LLM evals,
not in a functional gate.
"""

import pytest

from mcs.client import ChatClient
from mcs.config import Config

# Registered here (not only in pyproject) so markers resolve even when pytest is
# launched against the installed package, where pyproject.toml is not on disk.
_MARKERS = {
    "core": "API contract basics",
    "special_tokens": "BOS/EOS ownership across tokenization paths",
    "streaming": "SSE streaming behavior",
    "tools": "function / tool calling",
    "multimodal": "image and audio inputs",
    "multiturn": "multi-turn conversation state",
    "reasoning": "<think> / inner-monologue handling",
    "robustness": "chat-template injection & error paths",
    "perf": "performance smoke (excluded from default run)",
    "dev": "needs non-OpenAI extension endpoints (/tokenize, /detokenize); included by --spec dev",
}


def pytest_configure(config):
    for name, desc in _MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {desc}")


@pytest.fixture(scope="session")
def config() -> Config:
    cfg = Config.from_env()
    if not cfg.api_key:
        pytest.skip("no API key (set CSCS_SERVING_API or MCS_API_KEY)")
    return cfg


@pytest.fixture(scope="session")
def client(config) -> ChatClient:
    return ChatClient(config)


@pytest.fixture(autouse=True)
def _record_responses(request):
    """Record each test's request/response when --record-responses DIR is set
    (plumbed via MCS_RECORD_DIR). Passive: it never alters what is sent or
    asserted, so suite behaviour is identical with or without it."""
    import os

    from mcs import recording

    recording.configure(
        os.environ.get("MCS_RECORD_DIR"),
        request.node.name,
        os.environ.get("MCS_MODEL", "model"),
    )
    yield
    recording.reset()
