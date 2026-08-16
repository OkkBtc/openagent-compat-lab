"""Small, dependency-free helpers for keeping credentials out of output."""

from __future__ import annotations

import re

_QUERY_SECRET = re.compile(
    r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|key|authorization)=)([^&#\s]+)"
)
_AUTH_HEADER = re.compile(r"(?i)(authorization:\s*(?:bearer\s+)?)([^\s,;]+)")
_URL_USERINFO = re.compile(r"(https?://)[^/@\s]+@", re.IGNORECASE)


def redact(value: object, *secrets: str) -> str:
    """Redact known secrets plus common URL/header credential forms."""
    text = str(value)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = _QUERY_SECRET.sub(r"\1[REDACTED]", text)
    text = _AUTH_HEADER.sub(r"\1[REDACTED]", text)
    return _URL_USERINFO.sub(r"\1[REDACTED]@", text)
