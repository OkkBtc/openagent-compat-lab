import json
import os
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from mcs.agent_checks import run_agent_checks, run_agent_matrix
from mcs.capabilities import FAIL, PASS
from mcs.cli import main
from mcs.client import ApiError
from mcs.config import Config
from mcs.redaction import redact


class _MockHandler(BaseHTTPRequestHandler):
    model = "mock-model"
    required_key = "test-secret"
    chat_done = True
    response_text = "agent-compat-ok"
    request_bodies: ClassVar[list[dict]] = []

    def log_message(self, format, *args):
        pass

    def _authorized(self):
        if not self.required_key:
            return True
        return self.headers.get("Authorization") == f"Bearer {self.required_key}"

    def _send_json(self, payload, status=200):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self):
        size = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(size))

    def do_GET(self):
        if not self._authorized():
            self._send_json({"error": "unauthorized"}, 401)
        elif self.path == "/v1/models":
            self._send_json({"object": "list", "data": [{"id": self.model}]})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if not self._authorized():
            self._send_json({"error": "unauthorized"}, 401)
            return
        body = self._body()
        self.request_bodies.append(body)
        if self.path == "/v1/chat/completions":
            self._chat(body)
        elif self.path == "/v1/responses":
            self._responses(body)
        else:
            self._send_json({"error": "not found"}, 404)

    def _chat(self, body):
        if body.get("stream"):
            if body.get("tools"):
                events = [
                    (
                        'data: {"choices":[{"delta":{"tool_calls":'
                        '[{"index":0,"id":"call_order","type":"function",'
                        '"function":{"name":"lookup_order",'
                        '"arguments":"{\\"order_"}}]}}]}\n\n'
                    ),
                    (
                        'data: {"choices":[{"delta":{"tool_calls":'
                        '[{"index":1,"id":"call_weather","type":"function",'
                        '"function":{"name":"get_weather",'
                        '"arguments":"{\\"loca"}}]}}]}\n\n'
                    ),
                    (
                        'data: {"choices":[{"delta":{"tool_calls":'
                        '[{"index":0,"function":'
                        '{"arguments":"id\\":\\"A-100\\"}"}},'
                        '{"index":1,"function":'
                        '{"arguments":"tion\\":\\"Paris\\"}"}}]}}]}\n\n'
                    ),
                ]
            else:
                events = [
                    'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
                ]
            if self.chat_done:
                events.append("data: [DONE]\n\n")
            raw = "".join(events).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        messages = body.get("messages") or []
        if messages and messages[-1].get("role") == "tool":
            message = {
                "role": "assistant",
                "content": "Order A-100 is ready.",
            }
            finish_reason = "stop"
        elif body.get("tools"):
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup_order",
                            "arguments": '{"order_id":"A-100"}',
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        elif body.get("response_format"):
            message = {"role": "assistant", "content": '{"compatible":true}'}
            finish_reason = "stop"
        else:
            message = {"role": "assistant", "content": self.response_text}
            finish_reason = "stop"
        self._send_json(
            {"choices": [{"message": message, "finish_reason": finish_reason}]}
        )

    @staticmethod
    def _response_message(text):
        return {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }

    def _responses(self, body):
        if body.get("stream"):
            raw = (
                b"event: response.output_text.delta\n"
                b'data: {"type":"response.output_text.delta","delta":"ok"}\n\n'
                b"event: response.completed\n"
                b'data: {"type":"response.completed","response":{"id":"resp_1"}}\n\n'
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        input_items = body.get("input")
        if (
            isinstance(input_items, list)
            and input_items
            and input_items[0].get("type") == "function_call_output"
        ):
            output = [self._response_message("Order A-100 is ready.")]
        elif body.get("tools"):
            output = [
                {
                    "id": "call_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup_order",
                    "arguments": '{"order_id":"A-100"}',
                }
            ]
        elif body.get("text", {}).get("format"):
            output = [self._response_message('{"compatible":true}')]
        else:
            output = [self._response_message(self.response_text)]
        self._send_json({"id": "resp_1", "object": "response", "output": output})


@pytest.fixture
def mock_endpoint():
    _MockHandler.required_key = "test-secret"
    _MockHandler.chat_done = True
    _MockHandler.response_text = "agent-compat-ok"
    _MockHandler.request_bodies = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", server
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _config(base_url, key="test-secret"):
    return Config(base_url, key, "mock-model", 2)


@pytest.mark.parametrize(
    ("profile", "expected_count", "specific_check"),
    [
        ("generic", 7, None),
        ("hermes", 8, "hermes_tool_result_roundtrip"),
        ("openclaw", 8, "openclaw_streamed_parallel_tools"),
    ],
)
def test_chat_profiles_pass(mock_endpoint, profile, expected_count, specific_check):
    base_url, _ = mock_endpoint
    results = run_agent_checks(_config(base_url), profile)
    assert len(results) == expected_count
    assert {result.status for result in results} == {PASS}
    assert all(result.duration_ms >= 0 for result in results)
    if specific_check:
        assert specific_check in {result.name for result in results}


def test_codex_profile_passes(mock_endpoint):
    base_url, _ = mock_endpoint
    results = run_agent_checks(_config(base_url), "codex")
    assert len(results) == 8
    assert {result.status for result in results} == {PASS}
    assert "responses_tool_result_roundtrip" in {result.name for result in results}


def test_hermes_roundtrip_uses_strict_role_order(mock_endpoint):
    base_url, _ = mock_endpoint
    run_agent_checks(_config(base_url), "hermes")
    roundtrip = next(
        body
        for body in _MockHandler.request_bodies
        if len(body.get("messages") or []) == 3
    )
    messages = roundtrip["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant", "tool"]
    assert messages[2]["tool_call_id"] == messages[1]["tool_calls"][0]["id"]


def test_codex_roundtrip_pairs_call_id_and_previous_response(mock_endpoint):
    base_url, _ = mock_endpoint
    run_agent_checks(_config(base_url), "codex")
    roundtrip = next(
        body
        for body in _MockHandler.request_bodies
        if isinstance(body.get("input"), list)
        and body["input"]
        and body["input"][0].get("type") == "function_call_output"
    )
    assert roundtrip["previous_response_id"] == "resp_1"
    assert roundtrip["input"][0]["call_id"] == "call_1"


def test_agent_matrix_runs_named_profiles(mock_endpoint):
    base_url, _ = mock_endpoint
    matrix = run_agent_matrix(_config(base_url))
    assert list(matrix) == ["codex", "hermes", "openclaw"]
    assert all(
        result.status == PASS for results in matrix.values() for result in results
    )


def test_agent_matrix_runs_only_selected_profiles(mock_endpoint):
    base_url, _ = mock_endpoint
    matrix = run_agent_matrix(_config(base_url), ["hermes", "codex"])

    assert list(matrix) == ["hermes", "codex"]
    assert all(
        result.status == PASS for results in matrix.values() for result in results
    )


def test_cli_lists_checks_without_endpoint(capsys):
    status = main(["--profile", "codex", "--list-checks", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert list(payload["profiles"]) == ["codex"]
    assert payload["profiles"]["codex"][0] == "models_auth_and_target"
    assert payload["profiles"]["codex"][-1] == "responses_tool_result_roundtrip"


def test_cli_reports_installed_version(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--version"])

    assert error.value.code == 0
    assert capsys.readouterr().out == "agent-compat 0.2.0\n"


def test_chat_profile_detects_missing_done(mock_endpoint):
    base_url, _ = mock_endpoint
    _MockHandler.chat_done = False
    results = run_agent_checks(_config(base_url), "hermes")
    stream = next(result for result in results if result.name == "chat_stream_done")
    assert len(results) == 8
    assert stream.status == FAIL
    assert "[DONE]" in stream.detail


def test_fail_fast_stops_after_first_failed_check(mock_endpoint):
    base_url, _ = mock_endpoint
    _MockHandler.chat_done = False

    results = run_agent_checks(_config(base_url), "hermes", fail_fast=True)

    assert [result.name for result in results] == [
        "models_auth_and_target",
        "chat_basic",
        "chat_stream_done",
    ]
    assert results[-1].status == FAIL


def test_cli_fail_fast_stops_later_profiles(mock_endpoint, monkeypatch, capsys):
    base_url, _ = mock_endpoint
    monkeypatch.setenv("ACL_API_KEY", "test-secret")
    _MockHandler.required_key = "different-secret"

    status = main(
        [
            "--profile",
            "all",
            "--base-url",
            base_url,
            "--model",
            "mock-model",
            "--fail-fast",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert status == 1
    assert payload["fail_fast"] is True
    assert list(payload["profiles"]) == ["codex"]
    assert payload["profiles"]["codex"]["checks"][-1]["name"] == (
        "models_auth_and_target"
    )


def test_cli_writes_json_and_markdown(mock_endpoint, monkeypatch, capsys, tmp_path):
    base_url, _ = mock_endpoint
    monkeypatch.setenv("ACL_API_KEY", "test-secret")
    report = tmp_path / "report.md"
    status = main(
        [
            "--profile",
            "codex",
            "--base-url",
            base_url,
            "--model",
            "mock-model",
            "--json",
            "--markdown",
            str(report),
        ]
    )
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert status == 0
    assert payload["profile"] == "codex"
    assert "test-secret" not in output
    assert "test-secret" not in report.read_text()
    assert "8 passed, 0 failed/broken" in report.read_text()
    assert "Probe time" in report.read_text()
    assert payload["summary"]["compatible"] is True
    assert "duration_ms" in payload["checks"][0]


def test_cli_writes_all_profile_matrix(mock_endpoint, monkeypatch, capsys, tmp_path):
    base_url, _ = mock_endpoint
    monkeypatch.setenv("ACL_API_KEY", "test-secret")
    report = tmp_path / "matrix.md"
    status = main(
        [
            "--profile",
            "all",
            "--base-url",
            base_url,
            "--model",
            "mock-model",
            "--json",
            "--markdown",
            str(report),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert [row["profile"] for row in payload["matrix"]] == [
        "codex",
        "hermes",
        "openclaw",
    ]
    assert all(row["compatible"] for row in payload["matrix"])
    markdown = report.read_text()
    assert "openagent-compat-lab matrix" in markdown
    assert "responses_tool_result_roundtrip" in markdown
    assert "openclaw_streamed_parallel_tools" in markdown


def test_cli_writes_selected_profile_matrix(
    mock_endpoint, monkeypatch, capsys, tmp_path
):
    base_url, _ = mock_endpoint
    monkeypatch.setenv("ACL_API_KEY", "test-secret")
    markdown_report = tmp_path / "selected.md"
    junit_report = tmp_path / "selected.xml"

    status = main(
        [
            "--profile",
            "codex",
            "--profile",
            "hermes",
            "--base-url",
            base_url,
            "--model",
            "mock-model",
            "--json",
            "--markdown",
            str(markdown_report),
            "--junit",
            str(junit_report),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert payload["profile"] == "selected"
    assert [row["profile"] for row in payload["matrix"]] == ["codex", "hermes"]
    assert list(payload["profiles"]) == ["codex", "hermes"]
    markdown = markdown_report.read_text()
    assert "## codex" in markdown
    assert "## hermes" in markdown
    assert "openclaw" not in markdown
    assert [
        suite.attrib["name"] for suite in ET.parse(junit_report).findall("testsuite")
    ] == [
        "openagent-compat-lab.codex",
        "openagent-compat-lab.hermes",
    ]


@pytest.mark.parametrize(
    "profiles",
    [
        ["--profile", "codex", "--profile", "codex"],
        ["--profile", "all", "--profile", "hermes"],
        ["--profile", "model", "--profile", "codex"],
    ],
)
def test_cli_rejects_ambiguous_profile_combinations(profiles, capsys):
    with pytest.raises(SystemExit):
        main(profiles)

    assert "--profile" in capsys.readouterr().err


def test_cli_writes_agent_matrix_junit(mock_endpoint, monkeypatch, tmp_path):
    base_url, _ = mock_endpoint
    monkeypatch.setenv("ACL_API_KEY", "test-secret")
    report = tmp_path / "agent-compat.xml"
    status = main(
        [
            "--profile",
            "all",
            "--base-url",
            base_url,
            "--model",
            "mock-model",
            "--junit",
            str(report),
            "--json",
        ]
    )

    root = ET.parse(report).getroot()
    assert status == 0
    assert root.tag == "testsuites"
    assert root.attrib == {
        "tests": "24",
        "failures": "0",
        "errors": "0",
        "time": root.attrib["time"],
    }
    assert [suite.attrib["name"] for suite in root.findall("testsuite")] == [
        "openagent-compat-lab.codex",
        "openagent-compat-lab.hermes",
        "openagent-compat-lab.openclaw",
    ]
    assert len(root.findall(".//testcase")) == 24
    assert "test-secret" not in report.read_text()


def test_agent_junit_marks_protocol_failure(mock_endpoint, monkeypatch, tmp_path):
    base_url, _ = mock_endpoint
    monkeypatch.setenv("ACL_API_KEY", "test-secret")
    _MockHandler.chat_done = False
    report = tmp_path / "hermes.xml"
    status = main(
        [
            "--profile",
            "hermes",
            "--base-url",
            base_url,
            "--model",
            "mock-model",
            "--junit",
            str(report),
            "--json",
        ]
    )

    root = ET.parse(report).getroot()
    failure = root.find(".//testcase[@name='chat_stream_done']/failure")
    assert status == 1
    assert root.attrib["failures"] == "1"
    assert root.attrib["errors"] == "0"
    assert failure is not None
    assert "[DONE]" in failure.attrib["message"]


def test_cli_allows_explicit_no_auth(mock_endpoint, monkeypatch):
    base_url, _ = mock_endpoint
    _MockHandler.required_key = ""
    monkeypatch.delenv("ACL_API_KEY", raising=False)
    monkeypatch.delenv("MCS_API_KEY", raising=False)
    monkeypatch.delenv("CSCS_SERVING_API", raising=False)
    status = main(
        [
            "--profile",
            "generic",
            "--base-url",
            base_url,
            "--model",
            "mock-model",
            "--allow-no-auth",
            "--json",
        ]
    )
    assert status == 0


def test_cli_timeout_overrides_environment(mock_endpoint, monkeypatch):
    base_url, _ = mock_endpoint
    monkeypatch.setenv("ACL_API_KEY", "test-secret")
    monkeypatch.setenv("ACL_TIMEOUT", "60")
    monkeypatch.setenv("MCS_TIMEOUT", "60")

    status = main(
        [
            "--profile",
            "generic",
            "--base-url",
            base_url,
            "--model",
            "mock-model",
            "--timeout",
            "1.5",
            "--json",
        ]
    )

    assert status == 0
    assert os.environ["ACL_TIMEOUT"] == "1.5"
    assert os.environ["MCS_TIMEOUT"] == "1.5"
    assert Config.from_env().timeout == 1.5


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_cli_rejects_invalid_timeout(value, capsys):
    with pytest.raises(SystemExit):
        main(["--timeout", value])

    assert "finite number greater than zero" in capsys.readouterr().err


@pytest.mark.parametrize("profile", ["hermes", "codex", "all"])
def test_installed_console_script_end_to_end(mock_endpoint, profile):
    base_url, _ = mock_endpoint
    executable = Path(sys.executable).with_name("agent-compat")
    env = {**os.environ, "ACL_API_KEY": "test-secret"}
    completed = subprocess.run(
        [
            str(executable),
            "--profile",
            profile,
            "--base-url",
            base_url,
            "--model",
            "mock-model",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["profile"] == profile
    assert "test-secret" not in completed.stdout


def test_recorded_response_bodies_are_redacted(mock_endpoint, monkeypatch, tmp_path):
    base_url, _ = mock_endpoint
    _MockHandler.response_text = "test-secret"
    monkeypatch.setenv("ACL_API_KEY", "test-secret")
    record_dir = tmp_path / "wire"
    status = main(
        [
            "--profile",
            "generic",
            "--base-url",
            base_url,
            "--model",
            "mock-model",
            "--record-responses",
            str(record_dir),
            "--json",
        ]
    )
    recorded = "".join(path.read_text() for path in record_dir.rglob("*.*"))
    assert status == 0
    assert "test-secret" not in recorded
    assert "[REDACTED]" in recorded


def test_redaction_covers_known_and_url_secrets():
    text = (
        "Authorization: Bearer secret-value "
        "https://user:pass@example.test/v1?api_key=another-secret&x=1"
    )
    safe = redact(text, "secret-value")
    assert "secret-value" not in safe
    assert "another-secret" not in safe
    assert "user:pass" not in safe
    assert safe.count("[REDACTED]") == 3

    error = ApiError(401, "provider echoed secret-value", "secret-value")
    assert "secret-value" not in str(error)
