"""Optional request/response recorder, enabled with ``--record-responses DIR``.

When a record directory is configured, every HTTP request the suite makes and
the response it gets back are written to plain-text files, organised one folder
per test:

    DIR/<test-name>/<model>_input.txt    # the request body/bodies (pretty JSON)
    DIR/<test-name>/<model>_output.txt   # the raw response body/bodies

The model id is part of the filename (``/`` sanitised), so a multi-model
comparison writes every model's I/O side by side in the same per-test folder --
handy for eyeballing what each model actually received and returned (e.g. the
reasoning split, a tool-call leak, double-BOS) without re-deriving it from
pass/fail. A test that makes several calls appends them in order.

This is a passive recorder: it never changes what is sent or asserted, so it
keeps the suite's "general capability" behaviour identical with or without it.
"""

import json
import os

# Set per-test by the autouse fixture in conftest. ``dir`` None => recording off.
_ctx = {"dir": None, "test": None, "model": None}


def _pretty(text: str):
    """(formatted_text, extension). If ``text`` is JSON, pretty-print it and use
    a ``.json`` extension; otherwise pass it through as ``.txt`` (e.g. an SSE
    stream, which is ``data: {…}`` lines, not a single JSON document)."""
    if text.lstrip()[:1] in "{[":
        try:
            return json.dumps(json.loads(text), indent=2, ensure_ascii=False), "json"
        except (ValueError, TypeError):
            pass
    return text, "txt"


def configure(record_dir, test, model) -> None:
    _ctx.update(dir=record_dir or None, test=test, model=model or "model")


def reset() -> None:
    _ctx["test"] = None


def _safe(name: str) -> str:
    """Filesystem-safe slug (keeps the model id readable: swiss-ai__Apertus-…)."""
    return "".join(c if (c.isalnum() or c in "-._") else "_" for c in name)


def record(kind: str, text: str) -> None:
    """Append one ``input`` or ``output`` payload for the current test/model.

    No-op unless ``--record-responses`` configured a directory and a test is
    active. Never raises into the caller -- recording must not affect results."""
    base = _ctx.get("dir")
    test = _ctx.get("test")
    if not base or not test:
        return
    body, ext = _pretty(text)
    try:
        folder = os.path.join(base, _safe(test))
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{_safe(_ctx['model'])}_{kind}.{ext}")
        with open(path, "a", encoding="utf-8") as fh:
            if fh.tell():  # separate multiple calls within one test
                fh.write("\n" + "-" * 60 + "\n")
            fh.write(body.rstrip("\n") + "\n")
    except OSError:
        pass
