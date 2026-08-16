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
            api_base=_env(
                "MCS_API_BASE", default="https://api.swissai.svc.cscs.ch/v1"
            ).rstrip("/"),
            api_key=_env("MCS_API_KEY", "CSCS_SERVING_API", default=""),
            model=_env("MCS_MODEL", default="swiss-ai/Apertus-8B-Instruct-2509"),
            timeout=float(_env("MCS_TIMEOUT", default="120")),
        )
