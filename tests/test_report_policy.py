import json

import pytest

from mcs.cli import main


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _policy(*requirements):
    return {
        "format": "agent-compat-policy",
        "format_version": 1,
        "requirements": list(requirements),
    }


def _matrix_report():
    return {
        "profile": "all",
        "profiles": {
            "codex": {
                "checks": [
                    {"name": "responses_basic", "status": "pass"},
                    {"name": "responses_stream", "status": "fail"},
                ]
            },
            "hermes": {"checks": [{"name": "tool_roundtrip", "status": "skip"}]},
        },
    }


def test_policy_gate_accepts_required_and_explicitly_allowed_statuses(tmp_path, capsys):
    policy = _write_json(
        tmp_path / "policy.json",
        _policy(
            {"scope": "codex", "check": "responses_basic"},
            {
                "scope": "hermes",
                "check": "tool_roundtrip",
                "allowed_statuses": ["pass", "skip"],
            },
        ),
    )
    report = _write_json(tmp_path / "report.json", _matrix_report())

    assert main(["--enforce-policy", str(policy), str(report), "--json"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["summary"] == {
        "requirements": 2,
        "passed": 2,
        "failed": 0,
        "blocking": False,
    }
    assert [item["actual_status"] for item in result["requirements"]] == [
        "pass",
        "skip",
    ]


def test_policy_gate_blocks_disallowed_and_missing_checks(tmp_path, capsys):
    policy = _write_json(
        tmp_path / "policy.json",
        _policy(
            {"scope": "codex", "check": "responses_stream"},
            {"scope": "openclaw", "check": "parallel_tools"},
        ),
    )
    report = _write_json(tmp_path / "report.json", _matrix_report())

    assert main(["--enforce-policy", str(policy), str(report)]) == 1

    output = capsys.readouterr().out
    assert "0/2 requirement(s) passed; failed=2" in output
    assert "FAIL codex/responses_stream: fail (allowed: pass)" in output
    assert "FAIL openclaw/parallel_tools: missing (allowed: pass)" in output


def test_policy_gate_rejects_invalid_policy_and_run_options(tmp_path, capsys):
    report = _write_json(tmp_path / "report.json", _matrix_report())
    invalid = _write_json(
        tmp_path / "invalid-policy.json",
        _policy(
            {
                "scope": "codex",
                "check": "responses_basic",
                "allowed_statuses": ["unknown"],
            }
        ),
    )

    with pytest.raises(SystemExit) as error:
        main(["--enforce-policy", str(invalid), str(report)])
    assert error.value.code == 2
    assert "invalid allowed_statuses" in capsys.readouterr().err

    invalid_value = _write_json(
        tmp_path / "invalid-value-policy.json",
        _policy(
            {
                "scope": "codex",
                "check": "responses_basic",
                "allowed_statuses": [{"status": "pass"}],
            }
        ),
    )
    with pytest.raises(SystemExit) as error:
        main(["--enforce-policy", str(invalid_value), str(report)])
    assert error.value.code == 2
    assert "invalid allowed_statuses" in capsys.readouterr().err

    valid = _write_json(
        tmp_path / "policy.json",
        _policy({"scope": "codex", "check": "responses_basic"}),
    )
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--enforce-policy",
                str(valid),
                str(report),
                "--profile",
                "codex",
            ]
        )
    assert error.value.code == 2
    assert "cannot be combined" in capsys.readouterr().err
