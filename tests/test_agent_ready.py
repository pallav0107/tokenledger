"""Agent-ready release gate: schemas, tool definitions, MCP, closed stdin, doc budgets, conformance."""
from __future__ import annotations

import http.server
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "tokenledger"
CONF = ROOT / "conformance"
ENV = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
CMD = [sys.executable, "-m", "tokenledger"]
SAMPLE = str(ROOT / "examples" / "sample_logs" / "sample.jsonl")
ANTHROPIC = str(ROOT / "examples" / "sample_logs" / "anthropic.jsonl")
OPENAI = str(ROOT / "examples" / "sample_logs" / "openai.jsonl")


def run(args, cwd=None, stdin=subprocess.DEVNULL):
    return subprocess.run(CMD + list(args), capture_output=True, text=True, encoding="utf-8",
                          stdin=stdin, env=ENV, cwd=cwd, timeout=60)


def schema(name):
    return json.loads((ROOT / "schemas" / f"{name}.schema.json").read_text(encoding="utf-8"))


def validate(instance, name):
    jsonschema.validate(instance, schema(name), cls=jsonschema.Draft202012Validator)


# ---------------------------------------------------------------- schemas for every --json output
@pytest.mark.parametrize("args", [
    ["report", SAMPLE, "--json"],
    ["report", SAMPLE, "--json", "--quiet"],
    ["report", ANTHROPIC, "--format", "anthropic", "--json"],
    ["report", OPENAI, "--format", "openai", "--turn-key", "turn", "--step-key", "step", "--json"],
])
def test_report_json_matches_schema(args):
    r = run(args)
    assert r.returncode == 0, r.stderr
    validate(json.loads(r.stdout), "report")


@pytest.mark.parametrize("extra", [[], ["--quiet"]])
def test_what_if_json_matches_schema(extra):
    r = run(["what-if", SAMPLE, "--turn", "1", "--step", "0", "--json"] + extra)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    validate(out, "what_if")
    assert out == {"turn": 1, "step": 0, "tokens_saved": 22216}


def test_report_json_output_unchanged_from_0_1_1():
    """Published numeric output must be identical: compare with the 0.1.1 conformance fixture."""
    exp = json.loads((CONF / "expected.json").read_text(encoding="utf-8"))
    case = next(c for c in exp["cli"] if c["args"] == ["report", "logs/sample.jsonl", "--json"])
    assert run(["report", "logs/sample.jsonl", "--json"], cwd=CONF).stdout == case["stdout"]
    assert case["stdout"].startswith('{\n  "n_requests": 4,')


@pytest.mark.parametrize("args,code", [
    (["report", "nope.jsonl", "--json"], "TL_E_FILE_NOT_FOUND"),
    (["report", str(CONF / "logs" / "bad_json.jsonl"), "--json"], "TL_E_BAD_JSON"),
    (["report", str(CONF / "logs" / "missing_field.jsonl"), "--json"], "TL_E_MISSING_FIELD"),
    (["report", str(CONF / "logs" / "empty.jsonl"), "--json"], "TL_E_NO_REQUESTS"),
    (["report", str(CONF / "logs" / "bad_int.jsonl"), "--format", "openai", "--json"], "TL_E_BAD_VALUE"),
    (["what-if", SAMPLE, "--turn", "99", "--step", "0", "--json"], "TL_E_STEP_NOT_FOUND"),
    (["report", SAMPLE, "--format", "bogus", "--json"], "TL_E_USAGE"),
    (["what-if", SAMPLE, "--json"], "TL_E_USAGE"),
])
def test_errors_are_structured(args, code):
    r = run(args)
    assert r.returncode in (1, 2)
    assert "Traceback" not in r.stdout + r.stderr
    body = json.loads(r.stdout)
    validate(body, "error")
    assert body["error"]["code"] == code


def test_plain_error_goes_to_stderr_without_traceback():
    r = run(["report", "nope.jsonl"])
    assert r.returncode == 1 and r.stdout == ""
    assert "TL_E_FILE_NOT_FOUND" in r.stderr and "hint:" in r.stderr and "Traceback" not in r.stderr


def test_describe_validates():
    r = run(["describe", "--json"])
    assert r.returncode == 0
    d = json.loads(r.stdout)
    validate(d, "describe")
    for t in d["anthropic"]:
        validate(t, "tool_anthropic")
        jsonschema.Draft202012Validator.check_schema(t["input_schema"])
    for t in d["openai"]:
        validate(t, "tool_openai")
        jsonschema.Draft202012Validator.check_schema(t["function"]["parameters"])
    for t in d["mcp"]:
        validate(t, "tool_mcp")
        jsonschema.Draft202012Validator.check_schema(t["inputSchema"])
    # all three formats come from the one tools.json
    src = json.loads((PKG / "tools.json").read_text(encoding="utf-8"))["tools"]
    assert [t["name"] for t in d["anthropic"]] == [t["name"] for t in src] == [t["name"] for t in d["mcp"]]
    assert [t["function"]["name"] for t in d["openai"]] == [t["name"] for t in src]
    assert [t["input_schema"] for t in d["anthropic"]] == [t["inputSchema"] for t in d["mcp"]]
    assert [t["input_schema"] for t in d["anthropic"]] == [t["function"]["parameters"] for t in d["openai"]]


def test_describe_is_deterministic():
    assert run(["describe"]).stdout == run(["describe", "--json"]).stdout


# ---------------------------------------------------------------- MCP
class Mcp:
    def __init__(self, cwd=None):
        self.p = subprocess.Popen(CMD + ["mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, encoding="utf-8", env=ENV, cwd=cwd)
        self.n = 0

    def rpc(self, method, params=None):
        self.n += 1
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.n, "method": method,
                                       "params": params or {}}) + "\n")
        self.p.stdin.flush()
        msg = json.loads(self.p.stdout.readline())
        assert msg["id"] == self.n
        return msg

    def close(self):
        self.p.stdin.close()
        return self.p.wait(timeout=20)


def test_mcp_initialize_list_call():
    m = Mcp()
    try:
        init = m.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                    "clientInfo": {"name": "t", "version": "1"}})["result"]
        assert init["protocolVersion"] == "2025-06-18"
        assert "tools" in init["capabilities"] and init["serverInfo"]["name"] == "tokenledger"
        m.p.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        m.p.stdin.flush()
        tools = m.rpc("tools/list")["result"]["tools"]
        assert {t["name"] for t in tools} == {"tokenledger_report", "tokenledger_what_if"}
        for t in tools:
            validate(t, "tool_mcp")
        res = m.rpc("tools/call", {"name": "tokenledger_report", "arguments": {"logfile": SAMPLE}})["result"]
        assert res["isError"] is False
        validate(res["structuredContent"], "report")
        assert json.loads(res["content"][0]["text"]) == res["structuredContent"]
        assert res["structuredContent"]["total_input"] == 89532
        res = m.rpc("tools/call", {"name": "tokenledger_what_if",
                                   "arguments": {"logfile": SAMPLE, "turn": 1, "step": 0}})["result"]
        assert res["structuredContent"]["tokens_saved"] == 22216
        bad = m.rpc("tools/call", {"name": "tokenledger_report", "arguments": {"logfile": "nope"}})["result"]
        assert bad["isError"] is True
        validate(bad["structuredContent"], "error")
        assert bad["structuredContent"]["error"]["code"] == "TL_E_FILE_NOT_FOUND"
        assert m.rpc("bogus")["error"]["code"] == -32601
    finally:
        assert m.close() == 0


# ---------------------------------------------------------------- never blocks on stdin
@pytest.mark.parametrize("args", [
    [], ["--help"], ["report", SAMPLE], ["report", SAMPLE, "--json"], ["report", SAMPLE, "--quiet"],
    ["what-if", SAMPLE, "--turn", "1", "--step", "0"], ["describe"], ["describe", "--json"],
    ["report", "nope.jsonl", "--json"], ["mcp"],
])
def test_commands_complete_with_stdin_closed(args):
    p = subprocess.Popen(CMD + args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, env=ENV)
    p.stdin.close()
    p.communicate(timeout=30)
    assert p.returncode in (0, 1, 2)
    r = run(args, stdin=subprocess.DEVNULL)
    assert r.returncode in (0, 1, 2)


def test_no_ansi_codes_and_no_color_env():
    for env_extra in ({}, {"NO_COLOR": "1"}):
        r = subprocess.run(CMD + ["report", SAMPLE], capture_output=True, text=True, stdin=subprocess.DEVNULL,
                           env={**ENV, **env_extra})
        assert "\x1b[" not in r.stdout + r.stderr


# ---------------------------------------------------------------- docs and packaging
def tokens(text: str) -> int:
    """Conservative estimate: the larger of chars/4 and words*1.3."""
    return int(max(len(text) / 4, len(text.split()) * 1.3))


def test_agents_md_within_token_budget():
    assert tokens((ROOT / "AGENTS.md").read_text(encoding="utf-8")) <= 1500


def test_help_text_within_token_budget():
    for args in ([], ["report", "--help"], ["what-if", "--help"], ["describe", "--help"]):
        assert tokens(run(args).stdout) <= 500


def test_llms_txt_format():
    t = (ROOT / "llms.txt").read_text(encoding="utf-8")
    lines = t.splitlines()
    assert lines[0].startswith("# ") and any(l.startswith("> ") for l in lines)
    assert re.search(r"^- \[[^\]]+\]\([^)]+\)", t, re.M)


def test_package_copies_identical():
    for name in ("AGENTS.md", "llms.txt"):
        assert (ROOT / name).read_bytes() == (PKG / name).read_bytes(), name
    root_schemas = sorted(p.name for p in (ROOT / "schemas").glob("*.json"))
    assert root_schemas and root_schemas == sorted(p.name for p in (PKG / "schemas").glob("*.json"))
    for n in root_schemas:
        assert (ROOT / "schemas" / n).read_bytes() == (PKG / "schemas" / n).read_bytes(), n
    assert (PKG / "py.typed").exists()


def test_version_is_0_2_0():
    import tokenledger
    assert tokenledger.__version__ == "0.2.0"
    assert 'version = "0.2.0"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")


# ---------------------------------------------------------------- shared conformance (Python = reference)
def test_conformance_fixtures_reproduce(monkeypatch):
    monkeypatch.chdir(CONF)
    exp = json.loads((CONF / "expected.json").read_text(encoding="utf-8"))
    for c in exp["cli"] + exp["agent_cli"]:
        r = run(c["args"], cwd=CONF)
        assert r.returncode == c["exit_code"], c["args"]
        if "error" in c:
            e = json.loads(r.stdout)["error"]
            assert {"code": e["code"], "hint": e["hint"]} == c["error"], c["args"]
        else:
            assert r.stdout == c["stdout"], c["args"]
    from tokenledger import agent
    import io
    buf = io.StringIO()
    agent.serve(io.StringIO("".join(json.dumps(m) + "\n" for m in exp["mcp"]["input"])), buf)
    out = [json.loads(l) for l in buf.getvalue().splitlines()]
    for m in out:
        c = m.get("result", {}).get("structuredContent", {})
        if "error" in c:
            c["error"]["message"] = "<impl>"
    assert out == exp["mcp"]["output"]


# ---------------------------------------------------------------- eval harness (not run against a model)
def _runner():
    spec = importlib.util.spec_from_file_location("run_agent_eval", ROOT / "eval" / "run_agent_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_eval_tasks_are_valid_and_fixed_settings():
    run_mod = _runner()
    tasks = run_mod.load_tasks(ROOT / "eval" / "agent_tasks")
    assert len(tasks) >= 5
    for t in tasks:
        assert {"id", "prompt", "fixtures", "expect"} <= set(t)
        assert t["expect"].get("tool_called") or t["expect"].get("tool_calls")
        assert t["expect"].get("answer_contains")
    assert run_mod.DEFAULT_ENDPOINT == os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
    assert run_mod.DEFAULT_MODELS == ["gpt-oss:20b", "glm-4.7-flash"]
    assert run_mod.TEMPERATURE == 0.2 and run_mod.MAX_TOOL_CALLS == 20 and run_mod.MIN_RUNS == 3


def test_eval_score_rejects_wrong_and_accepts_right(tmp_path):
    s = _runner().score
    exp = {"tool_called": "t", "args_subset": {"a": 1}, "answer_contains": [["10", "10,0"], "x"]}
    ok_call = [{"name": "t", "arguments": {"a": 1, "b": 2}, "is_error": False}]
    assert s(exp, ok_call, "10 x", tmp_path)[0]
    assert not s(exp, ok_call, "11 x", tmp_path)[0]
    assert not s(exp, [{"name": "t", "arguments": {"a": 2}, "is_error": False}], "10 x", tmp_path)[0]
    assert not s(exp, [{"name": "t", "arguments": {"a": 1}, "is_error": True}], "10 x", tmp_path)[0]
    assert s({**exp, "allow_error": True}, [{"name": "t", "arguments": {"a": 1}, "is_error": True}],
             "10 x", tmp_path)[0]


def _stub_server(answer):
    seen = {}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen["req"] = body
            if body["messages"][-1]["role"] == "tool":
                msg = {"role": "assistant", "content": answer}
            else:
                msg = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "c1", "type": "function", "function": {
                        "name": "tokenledger_report", "arguments": json.dumps({"logfile": "log.jsonl"})}}]}
            data = json.dumps({"choices": [{"message": msg}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


@pytest.mark.parametrize("answer,verdict", [("total_input 61700 cache 40500", "PASS"), ("I do not know", "FAIL")])
def test_eval_harness_loop_with_stub_endpoint(tmp_path, answer, verdict):
    run_mod = _runner()
    tdir = tmp_path / "tasks"
    tdir.mkdir()
    src = next(t for t in run_mod.load_tasks(ROOT / "eval" / "agent_tasks") if t["id"] == "cost_report_json")
    (tdir / "t.yaml").write_text(json.dumps(src), encoding="utf-8")
    srv, seen = _stub_server(answer)
    old = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + old
    try:
        out, sc = tmp_path / "r.json", tmp_path / "s.json"
        code = run_mod.main(["--tool-cmd", f"{sys.executable} -m tokenledger",
                             "--endpoint", f"http://127.0.0.1:{srv.server_port}/v1", "--model", "stub",
                             "--tasks", str(tdir), "--out", str(out), "--scores", str(sc)])
    finally:
        srv.shutdown()
        os.environ["PYTHONPATH"] = old
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["verdict"] == verdict and code == (0 if verdict == "PASS" else 1)
    assert res["temperature"] == 0.2 and res["models"][0]["tasks"][0]["runs"][0]["transcript"]
    assert len(res["models"][0]["tasks"][0]["runs"]) == 3
    assert json.loads(sc.read_text(encoding="utf-8"))["primary_model"] == "stub"
    assert seen["req"]["temperature"] == 0.2
    assert "tokenledger" in seen["req"]["messages"][0]["content"]
    assert seen["req"]["tools"] == json.loads(run(["describe", "--json"]).stdout)["openai"]
