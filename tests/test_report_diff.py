import json

import pytest

from mcs.cli import main


def _write_report(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _single_report(statuses, *, profile="codex"):
    return {
        "profile": profile,
        "model": "provider/model",
        "api_base": "https://provider.example/v1",
        "checks": [
            {"name": name, "status": status, "detail": "", "duration_ms": 1.0}
            for name, status in statuses.items()
        ],
    }


def test_compare_reports_classifies_regressions_and_recoveries(tmp_path, capsys):
    baseline = _write_report(
        tmp_path / "baseline.json",
        _single_report({"stable": "pass", "recovering": "fail", "changed": "broken"}),
    )
    current = _write_report(
        tmp_path / "current.json",
        _single_report(
            {
                "stable": "fail",
                "recovering": "pass",
                "changed": "fail",
                "new_check": "pass",
            }
        ),
    )

    assert main(["--compare-reports", str(baseline), str(current), "--json"]) == 1

    comparison = json.loads(capsys.readouterr().out)
    assert comparison["summary"] == {
        "regressions": 1,
        "recoveries": 1,
        "status_changes": 1,
        "added": 1,
        "removed": 0,
        "unchanged": 0,
        "blocking": True,
    }
    assert comparison["regressions"][0] == {
        "scope": "codex",
        "check": "stable",
        "baseline_status": "pass",
        "current_status": "fail",
        "reason": "status_regression",
    }
    assert comparison["recoveries"][0]["check"] == "recovering"
    assert comparison["status_changes"][0]["check"] == "changed"


def test_compare_reports_blocks_missing_coverage_and_new_failures(tmp_path, capsys):
    baseline = _write_report(
        tmp_path / "baseline.json",
        _single_report({"unchanged": "pass", "removed": "pass"}),
    )
    current = _write_report(
        tmp_path / "current.json",
        _single_report({"unchanged": "pass", "introduced": "broken"}),
    )

    assert main(["--compare-reports", str(baseline), str(current)]) == 1

    output = capsys.readouterr().out
    assert "REGRESSION codex/removed: pass -> missing" in output
    assert "REGRESSION codex/introduced: missing -> broken" in output


def test_compare_reports_supports_matrix_and_multi_model_formats(tmp_path, capsys):
    matrix_baseline = {
        "profile": "all",
        "profiles": {
            "codex": {"checks": [{"name": "responses", "status": "pass"}]},
            "hermes": {"checks": [{"name": "roundtrip", "status": "fail"}]},
        },
    }
    matrix_current = {
        "profile": "all",
        "profiles": {
            "codex": {"checks": [{"name": "responses", "status": "pass"}]},
            "hermes": {"checks": [{"name": "roundtrip", "status": "pass"}]},
        },
    }
    baseline = _write_report(tmp_path / "matrix-baseline.json", matrix_baseline)
    current = _write_report(tmp_path / "matrix-current.json", matrix_current)
    assert main(["--compare-reports", str(baseline), str(current), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["summary"]["recoveries"] == 1

    multi_baseline = {
        "models": ["model-a", "model-b"],
        "checks": {
            "tools": {
                "model-a": {"name": "tools", "status": "pass"},
                "model-b": {"name": "tools", "status": "broken"},
            }
        },
    }
    multi_current = {
        "models": ["model-a", "model-b"],
        "checks": {
            "tools": {
                "model-a": {"name": "tools", "status": "pass"},
                "model-b": {"name": "tools", "status": "pass"},
            }
        },
    }
    baseline = _write_report(tmp_path / "multi-baseline.json", multi_baseline)
    current = _write_report(tmp_path / "multi-current.json", multi_current)
    assert main(["--compare-reports", str(baseline), str(current), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["recoveries"][0]["scope"] == "model-b"


def test_compare_reports_rejects_invalid_input_and_run_options(tmp_path, capsys):
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json", encoding="utf-8")
    valid = _write_report(tmp_path / "valid.json", _single_report({"basic": "pass"}))

    with pytest.raises(SystemExit) as error:
        main(["--compare-reports", str(invalid), str(valid)])
    assert error.value.code == 2
    assert "invalid JSON report" in capsys.readouterr().err

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--compare-reports",
                str(valid),
                str(valid),
                "--profile",
                "codex",
            ]
        )
    assert error.value.code == 2
    assert "cannot be combined" in capsys.readouterr().err
