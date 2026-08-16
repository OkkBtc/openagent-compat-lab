"""multiturn suite -- multi-turn conversation state. See SPEC.md 7.6.

Deterministic via a sentinel the model can't guess: state a code early, ask for
it later, assert the exact code comes back. mt-tools (a tool call mid-
conversation) is covered by the tools suite's tools-multiturn, which currently fails
on the -tools build (server chat-template bug), so it is not duplicated here.
"""

import re

import pytest

from mcs.client import ChatClient

pytestmark = pytest.mark.multiturn

SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]*\|>|</?(?:think|inner_prefix|inner_suffix)\b")

# Thinking-safe budget: a reasoning model burns tokens on a stripped `<think>`
# block before the answer, so a tight max_tokens truncates mid-thought and
# returns empty `content`. See core.py / SPEC.md 7.7.
_THINKING_MAX_TOKENS = 1024


def test_mt_context(client):
    """mt-context: a sentinel stated in turn 1 is recalled in a later turn."""
    messages = [
        {"role": "user", "content": "Remember this: my code is 4827."},
        {"role": "assistant", "content": "Got it -- your code is 4827."},
        {"role": "user", "content": "Tell me a one-sentence fun fact about cats."},
        {"role": "assistant", "content": "Cats sleep for most of the day."},
        {
            "role": "user",
            "content": "What code did I give you? Reply with only the number.",
        },
    ]
    content = (
        ChatClient.content(client.chat(messages, max_tokens=_THINKING_MAX_TOKENS)) or ""
    ).strip()
    assert "4827" in content, f"sentinel from turn 1 not recalled: {content!r}"


def test_mt_roles(client):
    """mt-roles: a longer alternating user/assistant history is handled cleanly --
    200, non-empty content, no role bleed / special-token leak."""
    messages = [
        {"role": "user", "content": "Let's play. I say a color, you say a fruit."},
        {"role": "assistant", "content": "Deal."},
        {"role": "user", "content": "Red."},
        {"role": "assistant", "content": "Apple."},
        {"role": "user", "content": "Yellow."},
        {"role": "assistant", "content": "Banana."},
        {
            "role": "user",
            "content": "Now, in one short sentence, what game are we playing?",
        },
    ]
    content = (
        ChatClient.content(client.chat(messages, max_tokens=_THINKING_MAX_TOKENS)) or ""
    ).strip()
    assert content, "empty assistant content"
    leak = SPECIAL_TOKEN_RE.search(content)
    assert not leak, f"special-token leak in multi-turn content: {content!r}"
