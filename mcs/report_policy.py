"""Offline policy-as-code gates for compatibility reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .report_diff import load_report_checks

_POLICY_FORMAT = "agent-compat-policy"
_POLICY_VERSION = 1
_MAX_POLICY_BYTES = 1024 * 1024
_STATUSES = {"pass", "fail", "broken", "skip"}


class PolicyFormatError(ValueError):
    """Raised when a compatibility policy is invalid."""


class PolicyGenerationError(ValueError):
    """Raised when a report is unsafe to promote into a strict policy."""


def generate_policy(report_path: Path) -> dict[str, Any]:
    """Create a deterministic pass-only policy from a fully passing report."""
    _, _, checks = load_report_checks(report_path)
    non_passing = [status for status in checks.values() if status != "pass"]
    if non_passing:
        counts = ", ".join(
            f"{status}={non_passing.count(status)}"
            for status in ("fail", "broken", "skip")
            if status in non_passing
        )
        raise PolicyGenerationError(
            f"cannot generate a strict policy from non-passing checks ({counts})"
        )
    return {
        "format": _POLICY_FORMAT,
        "format_version": _POLICY_VERSION,
        "requirements": [
            {"scope": scope, "check": check} for scope, check in sorted(checks)
        ],
    }


def _load_policy(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise PolicyFormatError(f"policy does not exist: {resolved}")
    if resolved.stat().st_size > _MAX_POLICY_BYTES:
        raise PolicyFormatError(f"policy exceeds 1 MiB limit: {resolved}")
    try:
        policy = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PolicyFormatError(f"invalid JSON policy {resolved}: {error}") from error
    if not isinstance(policy, dict):
        raise PolicyFormatError("policy must be a JSON object")
    if (
        policy.get("format") != _POLICY_FORMAT
        or policy.get("format_version") != _POLICY_VERSION
    ):
        raise PolicyFormatError("unsupported policy format or version")
    return resolved, policy


def _requirements(policy: dict[str, Any]) -> list[dict[str, Any]]:
    requirements = policy.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise PolicyFormatError("policy requirements must be a non-empty list")

    normalized = []
    seen: set[tuple[str, str]] = set()
    allowed_fields = {"scope", "check", "allowed_statuses"}
    for index, requirement in enumerate(requirements):
        if not isinstance(requirement, dict):
            raise PolicyFormatError(f"requirement {index} must be a JSON object")
        unknown = set(requirement) - allowed_fields
        if unknown:
            raise PolicyFormatError(
                f"requirement {index} has unsupported field: {min(unknown)}"
            )
        scope = requirement.get("scope")
        check = requirement.get("check")
        if not isinstance(scope, str) or not scope:
            raise PolicyFormatError(f"requirement {index} has an invalid scope")
        if not isinstance(check, str) or not check:
            raise PolicyFormatError(f"requirement {index} has an invalid check")
        key = (scope, check)
        if key in seen:
            raise PolicyFormatError(f"duplicate requirement: {scope}/{check}")
        seen.add(key)

        allowed = requirement.get("allowed_statuses", ["pass"])
        if (
            not isinstance(allowed, list)
            or not allowed
            or any(
                not isinstance(status, str) or status not in _STATUSES
                for status in allowed
            )
            or len(set(allowed)) != len(allowed)
        ):
            raise PolicyFormatError(
                f"requirement {scope}/{check} has invalid allowed_statuses"
            )
        normalized.append({"scope": scope, "check": check, "allowed_statuses": allowed})
    return normalized


def enforce_report_policy(policy_path: Path, report_path: Path) -> dict[str, Any]:
    """Evaluate required check statuses from a versioned JSON policy."""
    policy_file, policy = _load_policy(policy_path)
    requirements = _requirements(policy)
    report_file, report, checks = load_report_checks(report_path)

    results = []
    for requirement in requirements:
        key = (requirement["scope"], requirement["check"])
        actual = checks.get(key)
        compliant = actual in requirement["allowed_statuses"]
        result = {**requirement, "actual_status": actual, "compliant": compliant}
        if not compliant:
            result["reason"] = (
                "missing_check" if actual is None else "status_not_allowed"
            )
        results.append(result)

    failed = sum(not result["compliant"] for result in results)
    metadata = {
        key: report[key]
        for key in ("profile", "model", "models", "api_base")
        if key in report
    }
    return {
        "policy": {
            "path": str(policy_file),
            "format": _POLICY_FORMAT,
            "format_version": _POLICY_VERSION,
        },
        "report": {"path": str(report_file), **metadata},
        "summary": {
            "requirements": len(results),
            "passed": len(results) - failed,
            "failed": failed,
            "blocking": bool(failed),
        },
        "requirements": results,
    }


def render_policy_result(result: dict[str, Any]) -> str:
    """Render a concise policy-gate result for people and CI logs."""
    summary = result["summary"]
    lines = [
        "Compatibility policy gate",
        f"  policy: {result['policy']['path']}",
        f"  report: {result['report']['path']}",
        (
            f"  result: {summary['passed']}/{summary['requirements']} requirement(s) "
            f"passed; failed={summary['failed']}"
        ),
    ]
    for requirement in result["requirements"]:
        verdict = "PASS" if requirement["compliant"] else "FAIL"
        actual = requirement["actual_status"] or "missing"
        allowed = ",".join(requirement["allowed_statuses"])
        lines.append(
            f"  {verdict} {requirement['scope']}/{requirement['check']}: "
            f"{actual} (allowed: {allowed})"
        )
    return "\n".join(lines) + "\n"
