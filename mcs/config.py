"""Runtime configuration, resolved from environment variables.

See SPEC.md section 3 for the full table. Flags handled by run.sh / cli.py are
exported into these same env vars before pytest runs, so this is the single
source of truth.
"""

import os
from dataclasses import dataclass


def _env(*names, default=None):
    """Return the first set, non-empty environment variable among names."""
    for name in names:
        val = os.environ.get(name)
        if val:
            return val
    return default


@dataclass(frozen=True)
class Config:
    api_base: str
    api_key: str
    model: str
    timeout: float

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            api_base=_env("ACL_API_BASE", "MCS_API_BASE", default="").rstrip("/"),
            api_key=_env("ACL_API_KEY", "MCS_API_KEY", "CSCS_SERVING_API", default=""),
            model=_env("ACL_MODEL", "MCS_MODEL", default=""),
            timeout=float(_env("ACL_TIMEOUT", "MCS_TIMEOUT", default="60")),
        )
