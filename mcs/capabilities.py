"""Run the deterministic suites and render the result as a capability table.

The pytest suites in `suites/` ARE the capability checks -- there is no separate
set of probes. Running mcs executes them (against the configured
model) and prints a per-check ✔/✗/⚠ table, exiting non-zero on any failure. It
can be scoped to one capability (`--capability tools`) or compared across models
(`--model A --model B`).

Status per check, derived from the pytest outcome:
    ✔ pass    the deterministic check passed
    ✗ fail    an assertion failed (a real gap in the model/endpoint)
    ⚠ broken  the test errored (e.g. the server returned an HTTP error mid-check)
    – skip    the check was skipped (capability not applicable)
"""

from __future__ import annotations

import contextlib
import io
import json
import os
from dataclasses import dataclass

from .config import Config
from .redaction import redact

PASS, FAIL, BROKEN, SKIP = "pass", "fail", "broken", "skip"
_ICON = {PASS: "✔", FAIL: "✗", BROKEN: "⚠", SKIP: "–"}

_SUITES_DIR = os.path.join(os.path.dirname(__file__), "suites")


@dataclass
class Result:
    name: str  # check name, e.g. "tools_parallel"
    status: str
    detail: str = ""


def _check_name(nodeid: str) -> str:
    """ ".../tools.py::test_tools_parallel" -> "tools_parallel"."""
    func = nodeid.split("::")[-1]
    return func.removeprefix("test_")


def _crash_message(report) -> str:
    text = getattr(report, "longreprtext", "") or ""
    if not text:
        lr = getattr(report, "longrepr", None)
        crash = getattr(lr, "reprcrash", None)
        text = (
            crash.message
            if crash is not None and getattr(crash, "message", None)
            else str(lr or "")
        )
    # pytest prefixes error/assertion lines with "E"; the FIRST is the headline
    # (e.g. "AssertionError: ..." or "ApiError: HTTP ..."); later ones are the
    # "+ where ..." assertion-rewrite detail.
    e_lines = [ln[1:].strip() for ln in text.splitlines() if ln[:2] in ("E ", "E\t")]
    if e_lines:
        return e_lines[0][:200]
    nonempty = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return (nonempty[-1] if nonempty else "")[:200]


def _skip_reason(report) -> str:
    lr = report.longrepr
    if isinstance(lr, tuple) and len(lr) == 3:
        return lr[2].replace("Skipped: ", "")[:160]
    return ""


class _Collector:
    """pytest plugin: record one Result per test (insertion-ordered)."""

    def __init__(self):
        self._by_name = {}

    def pytest_runtest_logreport(self, report):
        name = _check_name(report.nodeid)
        if report.when == "setup":
            if report.outcome == "failed":  # fixture error -> broken
                self._by_name[name] = Result(name, BROKEN, _crash_message(report))
            elif report.outcome == "skipped":
                self._by_name.setdefault(name, Result(name, SKIP, _skip_reason(report)))
        elif report.when == "call":
            if report.outcome == "passed":
                self._by_name[name] = Result(name, PASS, "")
            elif report.outcome == "failed":
                msg = _crash_message(report)
                low = msg.lower()
                status = (
                    FAIL
                    if low.startswith("assert") or "assertionerror" in low
                    else BROKEN
                )
                self._by_name[name] = Result(name, status, msg)
            elif report.outcome == "skipped":
                self._by_name.setdefault(name, Result(name, SKIP, _skip_reason(report)))

    @property
    def results(self):
        return list(self._by_name.values())


def run_checks(
    config: Config,
    capability: str | None = None,
    spec: str = "openai",
    junit: str | None = None,
) -> list:
    """Run the suites against `config`'s model; return a list of Result.

    `spec` picks the API surface under test: "openai" (default) excludes
    checks marked `dev` (they need extension endpoints such as /tokenize and
    /detokenize that the OpenAI API spec does not define); "dev" includes
    them. Explicitly requesting the capability (e.g. --capability
    special_tokens) does NOT override the spec filter -- pass --spec dev.
    """
    os.environ["MCS_MODEL"] = config.model
    os.environ["MCS_API_BASE"] = config.api_base
    os.environ["ACL_MODEL"] = config.model
    os.environ["ACL_API_BASE"] = config.api_base
    if config.api_key:
        os.environ["MCS_API_KEY"] = config.api_key
        os.environ["ACL_API_KEY"] = config.api_key

    args = [_SUITES_DIR, "-o", "python_files=*.py", "-p", "no:cacheprovider", "-q"]
    marker = f"({capability})" if capability else "not perf"
    if spec != "dev":
        marker = f"({marker}) and not dev"
    args += ["-m", marker]
    if junit:
        args += [f"--junitxml={junit}"]

    import pytest  # required at runtime now (the default command runs the suites)

    collector = _Collector()
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        pytest.main(args, plugins=[collector])
    return collector.results


def _exit_code(results) -> int:
    return 1 if any(r.status in (FAIL, BROKEN) for r in results) else 0


def report(
    config: Config,
    capability: str | None = None,
    spec: str = "openai",
    as_json: bool = False,
    junit: str | None = None,
) -> int:
    results = run_checks(config, capability, spec, junit)
    if as_json:
        print(
            json.dumps(
                {
                    "model": redact(config.model, config.api_key),
                    "api_base": redact(config.api_base, config.api_key),
                    "checks": [
                        {
                            **r.__dict__,
                            "detail": redact(r.detail, config.api_key),
                        }
                        for r in results
                    ],
                },
                indent=2,
            )
        )
        return _exit_code(results)

    print(f"Capability checks for {redact(config.model, config.api_key)}")
    print(f"  endpoint: {redact(config.api_base, config.api_key)}\n")
    if not results:
        print("  (no checks ran -- unknown --capability, or no API key?)")
        return 1
    w = max(len(r.name) for r in results)
    for r in results:
        print(f"  {_ICON[r.status]} {r.name:<{w}}  {redact(r.detail, config.api_key)}")
    n_pass = sum(1 for r in results if r.status == PASS)
    n_fail = sum(1 for r in results if r.status in (FAIL, BROKEN))
    n_skip = sum(1 for r in results if r.status == SKIP)
    print(f"\n  {len(results)} checks")
    print(f"  {n_pass} passed")
    print(f"  {n_fail} failed/broken")
    print(f"  {n_skip} skipped")
    return _exit_code(results)


def report_compare(
    configs: list,
    capability: str | None = None,
    spec: str = "openai",
    as_json: bool = False,
    detail: bool = False,
) -> int:
    """Compare >=2 models. Transposed table (checks down, models across) with one
    glyph per cell; columns M1/M2/... keep it narrow, full ids in a legend. With
    detail=True, failure reasons are listed as per-model footnotes."""
    runs = [
        (c.model, {r.name: r for r in run_checks(c, capability, spec)}) for c in configs
    ]
    safe_runs = [
        (redact(model, config.api_key), results)
        for (model, results), config in zip(runs, configs, strict=True)
    ]
    names = []
    for _, res in runs:
        for n in res:
            if n not in names:
                names.append(n)

    failed = any(r.status in (FAIL, BROKEN) for _, res in runs for r in res.values())

    if as_json:
        print(
            json.dumps(
                {
                    "api_base": redact(configs[0].api_base, configs[0].api_key),
                    "models": [model for model, _ in safe_runs],
                    "checks": {
                        n: {
                            model: (
                                {
                                    **res[n].__dict__,
                                    "detail": redact(res[n].detail, configs[0].api_key),
                                }
                                if n in res
                                else None
                            )
                            for model, res in safe_runs
                        }
                        for n in names
                    },
                },
                indent=2,
            )
        )
        return 1 if failed else 0

    if not names:
        print("(no checks ran -- unknown --capability, or no API key?)")
        return 1
    cols = [f"M{i + 1}" for i in range(len(runs))]
    colw = [len(c) for c in cols]
    w = max(len(n) for n in names + ["passed", "failed/broken", "skipped"])
    print(
        f"Capability comparison ({redact(configs[0].api_base, configs[0].api_key)})\n"
    )
    header = " | ".join(
        f"{column:<{width}}" for column, width in zip(cols, colw, strict=True)
    )
    print(f"| {'Check':<{w}} | " + header + " |")
    print(f"|{'-' * (w + 2)}|" + "".join("-" * (cw + 2) + "|" for cw in colw))
    for n in names:
        cells = []
        for _, res in runs:
            r = res.get(n)
            cells.append(_ICON.get(r.status, " ") if r else " ")
        cells = [f"{cell:<{width}}" for cell, width in zip(cells, colw, strict=True)]
        print(f"| {n:<{w}} | " + " | ".join(cells) + " |")
    # double line separating the checks from the tally
    print(f"|{'=' * (w + 2)}|" + "".join("=" * (cw + 2) + "|" for cw in colw))
    # per-model tally, one row each for passed / failed-broken / skipped
    tally_rows = [
        ("passed", lambda r: r.status == PASS),
        ("failed/broken", lambda r: r.status in (FAIL, BROKEN)),
        ("skipped", lambda r: r.status == SKIP),
    ]
    for label, match in tally_rows:
        cells = [
            f"{sum(1 for r in res.values() if match(r)):<{cw}}"
            for (_, res), cw in zip(runs, colw, strict=True)
        ]
        print(f"| {label:<{w}} | " + " | ".join(cells) + " |")
    print("\nLegend: ✔ pass · ✗ fail · ⚠ broken · – skip")
    for i, (m, _) in enumerate(runs):
        print(f"M{i + 1} = {redact(m, configs[i].api_key)}")

    if detail:
        print("\nFailure details (non-pass checks):")
        for i, (m, res) in enumerate(runs):
            fails = [res[n] for n in names if n in res and res[n].status != PASS]
            print(f"\nM{i + 1} = {redact(m, configs[i].api_key)}")
            if not fails:
                print("  (all checks passed)")
            for r in fails:
                print(
                    f"  {_ICON[r.status]} {r.name:<{w}}  "
                    f"{redact(r.detail, configs[0].api_key)}"
                )
    return 1 if failed else 0
