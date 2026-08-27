"""Offline regression comparison for JSON compatibility reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_STATUSES = {"pass", "fail", "broken", "skip"}
_FAILURES = {"fail", "broken"}
_MAX_REPORT_BYTES = 8 * 1024 * 1024

CheckKey = tuple[str, str]


class ReportFormatError(ValueError):
    """Raised when an input is not a supported compatibility report."""


def _add_check(
    checks: dict[CheckKey, str], scope: str, record: Any, source: str
) -> None:
    if not isinstance(record, dict):
        raise ReportFormatError(f"{source} contains a non-object check record")
    name = record.get("name")
    status = record.get("status")
    if not isinstance(name, str) or not name:
        raise ReportFormatError(f"{source} contains a check without a name")
    if status not in _STATUSES:
        raise ReportFormatError(f"{source} has an invalid status for {name!r}")
    key = (scope, name)
    if key in checks:
        raise ReportFormatError(f"{source} contains duplicate check {scope}/{name}")
    checks[key] = status


def _extract_checks(report: dict[str, Any], source: str) -> dict[CheckKey, str]:
    checks: dict[CheckKey, str] = {}
    profiles = report.get("profiles")
    raw_checks = report.get("checks")

    if isinstance(profiles, dict):
        for profile, section in profiles.items():
            if not isinstance(profile, str) or not isinstance(section, dict):
                raise ReportFormatError(f"{source} has invalid profile results")
            records = section.get("checks")
            if not isinstance(records, list):
                raise ReportFormatError(f"{source} has no check list for {profile!r}")
            for record in records:
                _add_check(checks, profile, record, source)
    elif isinstance(raw_checks, list):
        scope = report.get("profile") or report.get("model")
        if not isinstance(scope, str) or not scope:
            raise ReportFormatError(f"{source} has no profile or model identifier")
        for record in raw_checks:
            _add_check(checks, scope, record, source)
    elif isinstance(raw_checks, dict) and isinstance(report.get("models"), list):
        models = report["models"]
        if not models or any(
            not isinstance(model, str) or not model for model in models
        ):
            raise ReportFormatError(f"{source} has invalid model identifiers")
        for name, by_model in raw_checks.items():
            if not isinstance(name, str) or not isinstance(by_model, dict):
                raise ReportFormatError(f"{source} has invalid multi-model checks")
            for model in models:
                record = by_model.get(model)
                if record is not None:
                    _add_check(checks, model, record, source)
    else:
        raise ReportFormatError(f"{source} is not a supported compatibility report")

    if not checks:
        raise ReportFormatError(f"{source} contains no comparable checks")
    return checks


def _load_report(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ReportFormatError(f"report does not exist: {resolved}")
    if resolved.stat().st_size > _MAX_REPORT_BYTES:
        raise ReportFormatError(f"report exceeds 8 MiB limit: {resolved}")
    try:
        report = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReportFormatError(f"invalid JSON report {resolved}: {error}") from error
    if not isinstance(report, dict):
        raise ReportFormatError(f"report must be a JSON object: {resolved}")
    return resolved, report


def _metadata(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("profile", "model", "models", "api_base")
        if key in report
    }


def _change(
    key: CheckKey,
    baseline_status: str | None,
    current_status: str | None,
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "scope": key[0],
        "check": key[1],
        "baseline_status": baseline_status,
        "current_status": current_status,
    }
    if reason is not None:
        value["reason"] = reason
    return value


def compare_report_files(baseline_path: Path, current_path: Path) -> dict[str, Any]:
    """Compare two report files and classify compatibility changes."""
    baseline_file, baseline_report = _load_report(baseline_path)
    current_file, current_report = _load_report(current_path)
    baseline = _extract_checks(baseline_report, str(baseline_file))
    current = _extract_checks(current_report, str(current_file))

    regressions = []
    recoveries = []
    status_changes = []
    added = []
    removed = []
    unchanged = 0

    for key in sorted(set(baseline) | set(current)):
        before = baseline.get(key)
        after = current.get(key)
        if before is None:
            change = _change(key, None, after)
            added.append(change)
            if after in _FAILURES:
                regressions.append(
                    _change(key, None, after, reason="new_non_passing_check")
                )
        elif after is None:
            change = _change(key, before, None)
            removed.append(change)
            regressions.append(
                _change(key, before, None, reason="missing_check_coverage")
            )
        elif before == after:
            unchanged += 1
        elif before == "pass" and after != "pass":
            regressions.append(_change(key, before, after, reason="status_regression"))
        elif before != "pass" and after == "pass":
            recoveries.append(_change(key, before, after))
        else:
            status_changes.append(_change(key, before, after))

    summary = {
        "regressions": len(regressions),
        "recoveries": len(recoveries),
        "status_changes": len(status_changes),
        "added": len(added),
        "removed": len(removed),
        "unchanged": unchanged,
        "blocking": bool(regressions),
    }
    return {
        "baseline": {"path": str(baseline_file), **_metadata(baseline_report)},
        "current": {"path": str(current_file), **_metadata(current_report)},
        "summary": summary,
        "regressions": regressions,
        "recoveries": recoveries,
        "status_changes": status_changes,
        "added": added,
        "removed": removed,
    }


def render_report_diff(comparison: dict[str, Any]) -> str:
    """Render a concise human-readable comparison."""
    summary = comparison["summary"]
    lines = [
        "Compatibility report comparison",
        f"  baseline: {comparison['baseline']['path']}",
        f"  current:  {comparison['current']['path']}",
        (
            f"  changes:  {summary['regressions']} regression(s), "
            f"{summary['recoveries']} recoveries, "
            f"{summary['status_changes']} other status change(s), "
            f"{summary['added']} added, {summary['removed']} removed"
        ),
    ]

    for change in comparison["regressions"]:
        before = change["baseline_status"] or "missing"
        after = change["current_status"] or "missing"
        lines.append(
            f"  REGRESSION {change['scope']}/{change['check']}: {before} -> {after}"
        )
    for change in comparison["recoveries"]:
        lines.append(
            f"  RECOVERY   {change['scope']}/{change['check']}: "
            f"{change['baseline_status']} -> {change['current_status']}"
        )
    for change in comparison["status_changes"]:
        lines.append(
            f"  CHANGED    {change['scope']}/{change['check']}: "
            f"{change['baseline_status']} -> {change['current_status']}"
        )

    regression_keys = {
        (change["scope"], change["check"]) for change in comparison["regressions"]
    }
    for change in comparison["added"]:
        if (change["scope"], change["check"]) not in regression_keys:
            lines.append(
                f"  ADDED      {change['scope']}/{change['check']}: "
                f"{change['current_status']}"
            )
    return "\n".join(lines) + "\n"
