"""special_tokens suite -- BOS/EOS ownership across every tokenization path.

Mostly token-level: read back what the server tokenized (via `/tokenize`,
`/detokenize`, or `/completions` `prompt_logprobs`) and assert an invariant about
the special tokens at the prompt's edges. Two checks are behavioral, because a
doubled BOS is only observable in generation on the paths `/tokenize` cannot reach.

Background (apertus-program #420, SML eval team, vLLM 0.19). A double BOS needs BOTH:

  1. the rendered text already carries a literal BOS -- the chat template emits
     `{{ bos_token }}` unconditionally, so an applied template starts with it; and
  2. that rendered string is encoded again with `add_special_tokens=True`, whose
     post-processor for a single sequence is `<bos> + Sequence`.

The result is a `<bos><bos>` bigram the model never saw in training, so on hard
prompts (medqa/math) it runs to the token budget and never emits its stop token.
vLLM's `add_special_tokens` default, before per-model overrides:

                     completion path   chat path
      text-only            True          False   <- the only carve-out
      multimodal           True          True    <- mm chat loses the carve-out

Two exposed paths, each reproduced here: multimodal chat, where the SERVER renders
and re-encodes (`bos_single_in_mm_chat*`; end-to-end in the multimodal suite's
`mm_no_degeneration_hard`), and the completion path, where the CALLER renders and
posts the string (`bos_no_double_in_rendered_completion`, `bos_rendered_prompt_stops`).

What the checks pin down:

    chat path       template owns the BOS  -> bos_single_in_chat, bos_single_in_mm_chat*
    raw/completion  tokenizer owns it      -> bos_single_in_raw_tokenize,
                                              bos_single_in_completions
    pre-rendered    nobody adds a second   -> bos_no_double_in_rendered_completion,
                                              bos_rendered_prompt_stops
    paths agree     same id, same count    -> bos_consistent_identity, bos_single_token,
                                              bos_generation_matches_tokenize
    EOS side        nobody appends one     -> eos_not_appended_to_prompt

Wherever a BOS is expected the check asserts *exactly one*, never "at most one" --
written as two bounds, `<=1` (nobody doubled it) and `>=1` (nobody dropped it), so
the failure names the direction that broke.

NO REQUEST HERE SENDS `add_special_tokens`. Every check reads the server's DEFAULT
tokenization, because the default is what a client that doesn't override the flag
actually gets, and it is the only behavior the server is answerable for. The
per-endpoint defaults (vLLM, before per-model overrides) are what the checks are
really probing:

    /tokenize chat form    False      /chat/completions   False
    /tokenize completion   True       /completions        True

READ THIS BEFORE "FIXING" A RED CHECK. Apertus 1.5 has TWO BOS owners: the template
emits `{{ bos_token }}`, and the tokenizer's `TemplateProcessing` post-processor
prepends `<s>` on `add_special_tokens=True`. That dual ownership IS the bug, so which
checks are red depends on which fix ships:

  * `bos_single_in_raw_tokenize` / `bos_single_in_completions` demand exactly one BOS
    on the raw default path -- the tokenizer keeps owning it there, because
    `/completions` and lm-eval loglikelihood never invoke the template and would
    otherwise lose the attention-sink token.
  * A template-owns fix (apertus-omni-tokenizer#18) strips the post-processor's BOS,
    making the raw default prepend ZERO. Those two go red BY DESIGN if it lands, and
    the reproductions go green.

So this suite does not describe settled behavior on the raw path; it makes the
ownership visible. Update it when the ownership question is decided -- see SPEC.md 7.2.

Model-agnostic throughout: the BOS is discovered from a rendered chat prompt (the
`bos` fixture), and a model with no BOS (e.g. Qwen) skips.
"""

import re

import pytest

from mcs.client import ApiError, ChatClient
from mcs.suites.core import (
    _CONTROL_TOKEN_RE,
    _HARD_MAX_TOKENS,
    _HARD_PROMPT,
    _THINKING_MAX_TOKENS,
    _degeneration_reason,
)
from mcs.suites.multimodal import _audio, _image, _text

pytestmark = pytest.mark.special_tokens


# --- constants ----------------------------------------------------------------

# Recognizable start-of-sequence tokens across model families. Identifies the BOS
# from a rendered chat prompt WITHOUT relying on the tokenizer auto-prepending it:
# after a template-owns fix the tokenizer may no longer add one, but the template
# still emits it as the rendered prompt's first token.
_BOS_TOKEN_RE = re.compile(
    r"^\s*(?:<s>|<\|begin_of_text\|>|<\|startoftext\|>|<bos>|\[BOS\]|<\|begin▁of▁sentence\|>)\s*$"
)

# Any short, deterministic text works; the checks are about the tokens at the
# edges, never the content. `_WORD` is used where a one-token body keeps the
# leading/trailing ids easy to read.
_PROMPT = "The capital of France is Paris."
_WORD = "Paris"

# A per-turn `{{ bos_token }}` (rather than one at the top) only doubles from the
# second turn on, so the chat check tokenizes both shapes.
_CHAT_SHAPES = {
    "single-turn": [{"role": "user", "content": _PROMPT}],
    "multi-turn": [
        {"role": "user", "content": "Hi!"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": _PROMPT},
    ],
}


# --- fixtures -----------------------------------------------------------------


@pytest.fixture
def bos(client):
    """``(bos_id, bos_str)``, read off a rendered chat prompt -- the template emits
    the BOS first -- so discovery never overrides ``add_special_tokens``. Skips the
    requesting test when the chat form / detokenize is unavailable, or the first
    token is not a recognizable BOS (a model with no BOS, e.g. Qwen).

    Yields the identity only, never the ids it saw: a test asserting on those would
    be checking the sequence this fixture already vetted, and since it skips unless
    ``ids[0]`` is a BOS, ">=1 leading BOS" would hold by construction. Every check
    re-tokenizes its own prompt.

    Function-scoped on purpose. Session scope would save a handful of tiny
    `/tokenize` calls, but `--record-responses` writes one folder per test (SPEC 8),
    so a cached fixture would record the discovery requests under whichever test ran
    first and omit them everywhere else.
    """
    try:
        ids = client.tokenize_chat([{"role": "user", "content": _WORD}])
    except ApiError as exc:
        pytest.skip(f"/tokenize does not accept chat messages: {exc}")
    if len(ids) < 2:
        pytest.skip("chat tokenization returned too few tokens")
    bos_id = ids[0]
    try:
        bos_str = client.detokenize([bos_id])
    except ApiError as exc:
        pytest.skip(f"/detokenize not available: {exc}")
    if not _BOS_TOKEN_RE.match(bos_str):
        pytest.skip(
            f"chat prompt does not begin with a recognizable BOS ({bos_str!r}); "
            f"model's chat format carries no leading BOS"
        )
    return bos_id, bos_str


# --- helpers ------------------------------------------------------------------


def _count_leading_bos(ids, bos_id):
    """How many BOS ids the sequence starts with (0, 1, or more)."""
    n = 0
    for t in ids:
        if t == bos_id:
            n += 1
        else:
            break
    return n


def _rendered_chat_prompt(client, messages):
    """The exact string the server's chat template renders for ``messages`` -- what a
    client applies locally before posting to `/completions`. Read back from the server
    (tokenize the chat form, detokenize the ids) so it is the server's own template."""
    try:
        ids = client.tokenize_chat(messages)
    except ApiError as exc:
        pytest.skip(f"/tokenize does not accept chat messages: {exc}")
    if not ids:
        pytest.skip("chat tokenization returned no tokens")
    try:
        return client.detokenize(ids)
    except ApiError as exc:
        pytest.skip(f"/detokenize not available: {exc}")


def _skip_unless_rendered_has_bos(rendered, bos_str):
    if not rendered.lstrip().startswith(bos_str):
        pytest.skip(f"rendered chat prompt does not start with {bos_str!r}")


def _assert_exactly_one_leading_bos(ids, bos, where):
    """Assert ``ids`` starts with exactly one BOS. Two bounds, so the message names
    the direction that broke."""
    bos_id, bos_str = bos
    count = _count_leading_bos(ids, bos_id)
    assert count <= 1, (
        f"{where} begins with {count} BOS tokens (id {bos_id}, {bos_str!r}), "
        f"expected 1; first ids {ids[:6]}. A repeated BOS is a bigram the model was "
        f"never trained on."
    )
    assert count >= 1, (
        f"{where} begins with no BOS (id {bos_id}, {bos_str!r}), expected 1; "
        f"first ids {ids[:6]}. The model was trained with a leading BOS."
    )


def _assert_single_bos_in_mm_chat(client, bos, part, kind):
    """Tokenize a `text + <modality>` chat through the server's own template and
    assert exactly one leading BOS. vLLM's multimodal tokenization restores
    ``add_special_tokens=True`` (via ``mm_processor.info.default_tok_params``), so a
    template that also emits ``{{ bos_token }}`` renders a doubled BOS.

    Best-effort proxy: ``/tokenize`` may not traverse the same mm code path as
    generation, but it is the only observable surface for the rendered mm prompt.
    """
    messages = [{"role": "user", "content": [_text("Describe this input."), part]}]
    try:
        ids = client.tokenize_chat(messages)
    except ApiError as exc:
        pytest.skip(f"/tokenize does not accept multimodal messages: {exc}")
    if not ids:
        pytest.skip("multimodal chat tokenization returned no tokens")
    _assert_exactly_one_leading_bos(ids, bos, f"{kind}+text chat prompt")


# --- BOS: the chat path (template owns the BOS) -------------------------------


@pytest.mark.dev
def test_bos_single_in_chat(client, bos):
    """bos-single-in-chat: a chat-templated prompt begins with exactly one BOS.

    Two means the template emitted the BOS and the server tokenized the rendered
    string with specials added on top. Zero means the template stopped emitting it.
    Both chat shapes: a per-turn `{{ bos_token }}` only doubles from the 2nd turn on.
    """
    for shape, messages in _CHAT_SHAPES.items():
        try:
            ids = client.tokenize_chat(messages)
        except ApiError as exc:
            pytest.skip(f"/tokenize does not accept chat messages: {exc}")
        if not ids:
            pytest.skip(f"{shape} chat tokenization returned no tokens")
        _assert_exactly_one_leading_bos(ids, bos, f"{shape} chat prompt")


@pytest.mark.dev
def test_bos_single_in_mm_chat(client, bos):
    """bos-single-in-mm-chat: an image+text chat prompt begins with exactly one BOS."""
    _assert_single_bos_in_mm_chat(client, bos, _image("image_4827.png"), "image")


@pytest.mark.dev
def test_bos_single_in_mm_chat_audio(client, bos):
    """bos-single-in-mm-chat-audio: an audio+text chat prompt begins with one BOS.

    Audio triggers the same `default_tok_params` mm path as image.
    """
    _assert_single_bos_in_mm_chat(client, bos, _audio("audio_fox.wav"), "audio")


# --- BOS: the raw / completion path (tokenizer owns the BOS) ------------------


@pytest.mark.dev
def test_bos_single_in_raw_tokenize(client, bos):
    """bos-single-in-raw-tokenize: raw (non-chat) /tokenize supplies exactly one BOS.

    `/completions`, offline `generate`, and lm-eval loglikelihood never invoke the
    chat template; they rely on the tokenizer for the BOS the model was pretrained
    with (the attention-sink first token). Zero is the over-correction of a
    template-owns fix; two is the double-BOS.
    """
    try:
        raw = client.tokenize(_PROMPT)
    except ApiError as exc:
        pytest.skip(f"/tokenize not available: {exc}")
    _assert_exactly_one_leading_bos(raw, bos, "default raw tokenization")


@pytest.mark.dev
def test_bos_single_in_completions(client, bos):
    """bos-single-in-completions: the /completions DEFAULT supplies exactly one BOS.

    Same invariant as `bos-single-in-raw-tokenize`, on the endpoint that actually
    generates -- a gateway may default differently there.

    `prompt_logprobs` reports None at position 0, so the first token is never directly
    observable. Anchor on the content: take the DEFAULT `/tokenize` of the same prompt
    and drop whatever BOS it prepended, leaving the body ids. That adapts to any
    `/tokenize` default, so this check stays independent of the one above.

        ids[1] == bos_id   -> positions 0 AND 1 are BOS: doubled
        ids[1] == body[0]  -> exactly one token precedes the content: one BOS
        ids[1] == body[1]  -> the content starts at position 0: no BOS at all
    """
    bos_id, bos_str = bos
    try:
        raw = client.tokenize(_PROMPT)
        ids = client.prompt_token_ids(_PROMPT)
    except ApiError as exc:
        pytest.skip(f"/tokenize or prompt_logprobs not available: {exc}")
    # `_PROMPT` is plain prose, so every leading BOS here came from the tokenizer.
    body = raw[_count_leading_bos(raw, bos_id) :]
    if len(body) < 2 or len(ids) < 2:
        pytest.skip("prompt too short to locate the content in the token ids")
    assert ids[1] != bos_id, (
        f"/completions default prepended 2 BOS (id {bos_id}, {bos_str!r}), expected 1: "
        f"position 1 is a BOS, so position 0 is one too. ids {ids}, body {body[:4]}."
    )
    assert ids[1] == body[0], (
        f"/completions default prepended no BOS (id {bos_id}, {bos_str!r}), "
        f"expected 1: the content resumes at position 1 with {ids[1]}, not "
        f"{body[0]}, so it began at position 0. ids {ids}, body {body[:4]}. (Or the "
        f"two tokenizations disagree.) Raw completion and loglikelihood paths lose "
        f"the attention-sink token."
    )


# --- BOS: an already-rendered prompt (nobody adds a second) -------------------
#
# The path a client renders itself -- OpenWebUI, lm-eval -- then posts as a string.
# vLLM defaults `add_special_tokens=True` on the completion path, so the tokenizer
# prepends a BOS on top of the template's. These two reproduce apertus-program #420
# text-only, no attachment needed. Expected RED until the BOS has a single owner:
# they reproduce a live bug, they do not describe desired behavior.


@pytest.mark.dev
def test_bos_rendered_prompt_stops(client, bos):
    """bos-rendered-prompt-stops: a hard prompt, rendered and posted to /completions
    under the server's DEFAULT tokenization, reaches its stop token instead of
    running away.

    The behavioral half: a doubled BOS only shows up in generation, and only on hard
    prompts. The default prepends a BOS onto the one the template already rendered, so
    the model sees a bigram it never trained on.

    The control is `core-no-degeneration-hard`, which sends the SAME `_HARD_PROMPT`
    text-only through the chat path (one BOS) and passes. Green there and red here
    isolates the doubled BOS. Both are cheap enough to keep separate; neither sends
    `add_special_tokens`.
    """
    _bos_id, bos_str = bos
    hard = [{"role": "user", "content": _HARD_PROMPT}]
    rendered = _rendered_chat_prompt(client, hard)
    _skip_unless_rendered_has_bos(rendered, bos_str)
    try:
        resp = client.complete(rendered, max_tokens=_HARD_MAX_TOKENS)
    except ApiError as exc:
        pytest.skip(f"/completions not available: {exc}")
    finish = resp["choices"][0]["finish_reason"]
    text = ChatClient.completion_text(resp)
    assert finish == "stop", (
        f"a rendered chat prompt posted to /completions ran to "
        f"finish_reason={finish!r} without stopping (budget {_HARD_MAX_TOKENS}). The "
        f"default prepends a second {bos_str!r} onto the template's, and the model "
        f"never reaches its stop token. The same prompt sent text-only through chat "
        f"(core-no-degeneration-hard) stops normally. Tail: {text[-200:]!r}"
    )
    reason = _degeneration_reason(text)
    assert reason is None, (
        f"a rendered chat prompt posted to /completions degenerated where the same "
        f"prompt sent text-only through chat does not: {reason}"
    )


# --- BOS: the paths must agree -------------------------------------------------


@pytest.mark.dev
def test_bos_single_token(client, bos):
    """bos-single-token: the model's BOS string encodes to exactly one BOS token.

    The prefixed tokenization must be `[bos_id] + plain`: whatever the default adds,
    it adds to both, so the difference isolates how the BOS *string* encodes. A split
    means the template (which emits that string) and the tokenizer disagree, and the
    template text will not map back to the BOS the model was trained on.
    """
    bos_id, bos_str = bos
    try:
        plain = client.tokenize(_PROMPT)
        prefixed = client.tokenize(f"{bos_str}{_PROMPT}")
    except ApiError as exc:
        pytest.skip(f"/tokenize not available: {exc}")
    assert prefixed == [bos_id] + plain, (
        f"BOS string {bos_str!r} did not encode to a single BOS token (id {bos_id}) "
        f"on the default /tokenize path: prefixed ids {prefixed[:8]} vs plain ids "
        f"{plain[:8]} -- chat-template/tokenizer mismatch."
    )


@pytest.mark.dev
def test_bos_consistent_identity(client, bos):
    """bos-consistent-identity: the chat and raw paths agree on the BOS token.

    Otherwise the two paths feed the model different 'start' tokens. Only checked when
    the raw default actually prepends a recognizable BOS -- a config with none is
    `bos-single-in-raw-tokenize`'s failure to report, not this one's.
    """
    bos_id, bos_str = bos
    try:
        raw = client.tokenize(_WORD)
        raw_first = client.detokenize([raw[0]]) if raw else ""
    except ApiError as exc:
        pytest.skip(f"/tokenize or /detokenize not available: {exc}")
    if not _BOS_TOKEN_RE.match(raw_first):
        pytest.skip("raw default path prepends no BOS (see bos-single-in-raw-tokenize)")
    assert raw[0] == bos_id, (
        f"BOS identity mismatch: the chat template emits id {bos_id} ({bos_str!r}) but "
        f"the raw default path prepends id {raw[0]} ({raw_first!r}). Both must feed "
        f"the model the same BOS it was trained with."
    )


@pytest.mark.dev
def test_bos_generation_matches_tokenize(client, bos):
    """bos-generation-matches-tokenize: the prompt the model consumed has the same
    token count as `/tokenize`'s chat form.

    Closes a proxy gap. Every other probe reads `/tokenize`, which need not traverse
    the generation path -- the doubling lives in the renderer, so one can be patched
    while the other is not. `usage.prompt_tokens` is the only observable count of what
    the model was actually fed. Text-only: a multimodal prompt's `prompt_tokens`
    includes vision-placeholder expansion, so the comparison would not be sound.
    """
    messages = [{"role": "user", "content": _WORD}]
    try:
        chat_ids = client.tokenize_chat(messages)
    except ApiError as exc:
        pytest.skip(f"/tokenize does not accept chat messages: {exc}")
    if not chat_ids:
        pytest.skip("chat tokenization returned no tokens")
    usage = client.chat(messages, max_tokens=1).get("usage") or {}
    if "prompt_tokens" not in usage:
        pytest.skip("endpoint reports no usage.prompt_tokens")
    generated = usage["prompt_tokens"]
    assert generated == len(chat_ids), (
        f"the generation path tokenized the prompt to {generated} tokens but "
        f"/tokenize's chat form gives {len(chat_ids)} (first ids {chat_ids[:6]}). A "
        f"difference of one is a BOS added on one path and not the other; every other "
        f"BOS check here reads /tokenize and is only a proxy for what the model sees."
    )


# --- EOS ----------------------------------------------------------------------


def test_eos(client):
    """eos: the model stops on its own EOS for a short, complete answer.

    A bounded question should finish with `finish_reason="stop"`, not run to the
    budget -- always finishing on "length" points at a misconfigured `eos_token_id` /
    generation config. The content must also carry no raw EOS: the template/parser
    should consume it.
    """
    resp = client.chat(
        [
            {
                "role": "user",
                "content": "What is the capital of France? Answer with just the "
                "city name.",
            }
        ],
        max_tokens=_THINKING_MAX_TOKENS,
    )
    finish = resp["choices"][0]["finish_reason"]
    assert finish == "stop", (
        f"expected natural stop, got finish_reason={finish!r} -- the model did not "
        f"reach EOS within the budget: a misconfigured eos_token_id / generation "
        f"config, or a reasoning model that never finished thinking"
    )
    content = ChatClient.content(resp) or ""
    leak = _CONTROL_TOKEN_RE.search(content)
    assert not leak, (
        f"raw control/EOS token {leak.group(0)!r} leaked into content: {content!r}"
    )


@pytest.mark.dev
def test_eos_not_appended_to_prompt(client):
    """eos-not-appended-to-prompt: default tokenization must not append an EOS to a
    raw prompt.

    The EOS-side mirror of the BOS ownership checks. A raw completion prompt is a
    prefix the model continues from; a tokenizer misconfigured with
    `add_eos_token=True` appends the EOS, so the model sees a premature stop
    mid-context -> truncated or degenerate continuations.
    """
    try:
        raw = client.tokenize(_PROMPT)
    except ApiError as exc:
        pytest.skip(f"/tokenize not available: {exc}")
    if not raw:
        pytest.skip("could not tokenize")
    try:
        last = client.detokenize([raw[-1]])
    except ApiError as exc:
        pytest.skip(f"/detokenize not available: {exc}")
    assert not _CONTROL_TOKEN_RE.search(last), (
        f"default tokenization appended a control/EOS token {last!r} to a raw prompt "
        f"(last ids {raw[-3:]}). A completion prompt must not end in an EOS -- the "
        f"model would see a premature stop mid-context."
    )
