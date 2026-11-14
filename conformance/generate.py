"""Regenerate conformance/expected.json from the REFERENCE Python implementation (tokenledger-llm).

Usage:  PYTHONPATH=/path/to/python/tokenledger-llm python3 conformance/generate.py

The Python package is the source of truth. The TypeScript tests (test/conformance.test.ts) read
expected.json and assert the JS port reproduces every value, byte for byte on CLI output.
"""
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import asdict

import tokenledger
from tokenledger import adapters, parse
from tokenledger.account import account, what_if_remove_step
from tokenledger.report import main as cli_main

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

NATIVE = ["sample", "tie_rounding", "zero_input", "empty", "missing_field", "mapped"]

# ---------------------------------------------------------------- 0.2.0 robustness logs (bytes matter)
L1 = '{"turn": 1, "step": 0, "input_tokens": 1000, "cache_read_tokens": 0, "tool_result_tokens": 200}'
L2 = '{"turn": 1, "step": 1, "input_tokens": 1400, "cache_read_tokens": 1000, "output_tokens": 7}'
L3 = '{"turn": 2, "step": 0, "input_tokens": 90, "priming_tokens": 50}'
BAD_ONE = '{"turn": 1, "step": 0, "input_tokens": %s}'
ROBUST_LOGS = {
    # accepted
    "bom.jsonl": b"\xef\xbb\xbf" + "\n".join([L1, L2, L3]).encode() + b"\n",
    "crlf.jsonl": "\r\n".join([L1, L2, L3]).encode() + b"\r\n",
    "cr_only.jsonl": "\r".join([L1, L2, L3]).encode() + b"\r",
    "mixed_eol.jsonl": (L1 + "\r\n" + L2 + "\r" + L3 + "\n").encode(),
    "unicode_ws.jsonl": (" " + L1 + "　\n\x1c" + L2 + "\x1f\n\x85" + L3 + " \n \n").encode(),
    "bom_crlf_comments.jsonl": b"\xef\xbb\xbf# c\r\n\r\n" + L1.encode() + b"\r\n  # indented comment\r\n" + L2.encode() + b"\r\n",
    "big_ints.jsonl": ("\n".join([
        '{"turn": 1, "step": 0, "input_tokens": 9007199254740993, "cache_read_tokens": 9007199254740991, "priming_tokens": 9007199254740992, "conversation_tokens": 1, "tool_result_tokens": 9007199254740993, "output_tokens": 9007199254740995}',
        '{"turn": 1, "step": 1, "input_tokens": 9007199254740999, "cache_read_tokens": 9007199254740993, "priming_tokens": 9007199254740993, "conversation_tokens": 3, "tool_result_tokens": 5, "output_tokens": 9007199254740997}',
        '{"turn": 18446744073709551616, "step": 0, "input_tokens": 123456789012345678901234567890, "cache_read_tokens": 100000000000000000000000000001, "output_tokens": 1}',
    ]) + "\n").encode(),
    "huge_mean.jsonl": ('{"turn": 1, "step": 0, "input_tokens": 1' + "0" * 22 + '}\n{"turn": 1, "step": 1, "input_tokens": 1' + "0" * 22 + '}\n').encode(),
    "neg_zero.jsonl": (L1 + '\n{"turn": 1, "step": -0, "input_tokens": 5}\n').encode(),
    "extra_float_field.jsonl": (L1 + '\n{"turn": 1, "step": 1, "input_tokens": 1, "note": 1e999, "x": 1.5}\n').encode(),
    "duplicate_keys.jsonl": b'{"turn": 1, "step": 0, "input_tokens": 5, "input_tokens": 9}\n',
    "unicode_label.jsonl": ('{"turn": 1, "step": 0, "input_tokens": 5, "label": "café \U0001F600"}\n').encode(),
    "anthropic_big.jsonl": ('{"usage": {"input_tokens": 9007199254740993, "cache_creation_input_tokens": 9007199254740993, "cache_read_input_tokens": 9007199254740993, "output_tokens": 1}}\n').encode(),
    # rejected: token counts
    "float_tokens.jsonl": (BAD_ONE % "1.5" + "\n").encode(),
    "float_integral.jsonl": (BAD_ONE % "1.0" + "\n").encode(),
    "exp_tokens.jsonl": (BAD_ONE % "1e3" + "\n").encode(),
    "bool_tokens.jsonl": (BAD_ONE % "true" + "\n").encode(),
    "str_tokens.jsonl": (BAD_ONE % '"100"' + "\n").encode(),
    "neg_tokens.jsonl": (BAD_ONE % "-1" + "\n").encode(),
    "list_tokens.jsonl": (BAD_ONE % "[1]" + "\n").encode(),
    "float_optional.jsonl": (L1 + '\n{"turn": 1, "step": 1, "input_tokens": 5, "cache_read_tokens": 2.5}\n').encode(),
    "float_turn.jsonl": b'{"turn": 1.0, "step": 0, "input_tokens": 5}\n',
    "bool_turn.jsonl": b'{"turn": true, "step": 0, "input_tokens": 5}\n',
    "str_output.jsonl": b'{"turn": 1, "step": 0, "input_tokens": 5, "output_tokens": "7"}\n',
    "numeric_label.jsonl": b'{"turn": 1, "step": 0, "input_tokens": 5, "label": 7}\n',
    "null_input.jsonl": b'{"turn": 1, "step": 0, "input_tokens": null}\n',
    "float_openai.jsonl": b'{"usage": {"prompt_tokens": 12.5, "completion_tokens": 1}}\n',
    "str_anthropic.jsonl": b'{"usage": {"input_tokens": "100", "output_tokens": 1}}\n',
    "bool_openai.jsonl": b'{"usage": {"prompt_tokens": true}}\n',
    "str_turn_key.jsonl": b'{"turn": "1", "step": 0, "usage": {"prompt_tokens": 5}}\n',
    "missing_turn_key.jsonl": b'{"step": 0, "usage": {"prompt_tokens": 5}}\n',
    "huge_digits.jsonl": ('{"turn": 1, "step": 0, "input_tokens": ' + "9" * 4301 + "}\n").encode(),
    # rejected: JSON
    "nan_literal.jsonl": (BAD_ONE % "NaN" + "\n").encode(),
    "infinity_literal.jsonl": (BAD_ONE % "Infinity" + "\n").encode(),
    "neg_infinity_literal.jsonl": (BAD_ONE % "-Infinity" + "\n").encode(),
    "nan_extra_field.jsonl": (L1 + '\n{"turn": 1, "step": 1, "input_tokens": 1, "note": NaN}\n').encode(),
    "leading_zero.jsonl": (BAD_ONE % "01" + "\n").encode(),
    "plus_sign.jsonl": (BAD_ONE % "+1" + "\n").encode(),
    "trailing_garbage.jsonl": (L1 + " x\n").encode(),
    "trailing_comma.jsonl": b'{"turn": 1, "step": 0, "input_tokens": 5,}\n',
    "single_quotes.jsonl": b"{'turn': 1, 'step': 0, 'input_tokens': 5}\n",
    "non_object_list.jsonl": b"[1, 2]\n",
    "non_object_scalar.jsonl": b"5\n",
    "non_object_null.jsonl": b"null\n",
    "deep_nesting.jsonl": ('{"turn": 1, "step": 0, "input_tokens": 5, "x": ' + "[" * 101 + "]" * 101 + "}\n").encode(),
    "nesting_ok.jsonl": ('{"turn": 1, "step": 0, "input_tokens": 5, "x": ' + "[" * 99 + "]" * 99 + "}\n").encode(),
    "control_char_string.jsonl": b'{"turn": 1, "step": 0, "input_tokens": 5, "label": "a\tb"}\n',
    "bom_twice.jsonl": b"\xef\xbb\xbf\xef\xbb\xbf" + L1.encode() + b"\n",
    "bom_mid_file.jsonl": (L1 + "\n").encode() + b"\xef\xbb\xbf" + L2.encode() + b"\n",
    # rejected: encoding
    "invalid_utf8.jsonl": (L1 + "\n").encode() + b'{"turn": 1, "step": 1, "input_tokens": 5, "label": "\xff\xfe"}\n',
    "invalid_utf8_comment.jsonl": b"# \xff\xfe comment\n" + L1.encode() + b"\n",
    "latin1.jsonl": '{"turn": 1, "step": 0, "input_tokens": 5, "label": "café"}\n'.encode("latin-1"),
    "utf16.jsonl": (L1 + "\n").encode("utf-16"),
    "overlong_utf8.jsonl": b'{"turn": 1, "step": 0, "input_tokens": 5, "label": "\xc0\xaf"}\n',
    "surrogate_utf8.jsonl": b'{"turn": 1, "step": 0, "input_tokens": 5, "label": "\xed\xa0\x80"}\n',
    "whitespace_only.jsonl": b" \n\t\r\n\xc2\xa0\n",
    "only_bom.jsonl": b"\xef\xbb\xbf",
}
for _name, _data in ROBUST_LOGS.items():
    with open(os.path.join("logs", _name), "wb") as _f:
        _f.write(_data)


def cli(args):
    buf = io.StringIO()
    err = io.StringIO()
    try:
        sys.stderr, old = err, sys.stderr
        with redirect_stdout(buf):
            try:
                code = cli_main(args)
            except SystemExit as e:  # argparse errors
                code = e.code
            except Exception:
                code = 1
    finally:
        sys.stderr = old
    return {"args": args, "exit_code": code, "stdout": buf.getvalue()}


def attempt(fn):
    try:
        return {"ok": fn()}
    except Exception as e:  # recorded as an error class only; message text is not compared
        return {"error": True}


def reqs_to_json(reqs):
    return [asdict(r) for r in reqs]


out = {"reference": {"package": "tokenledger-llm", "version": tokenledger.__version__}, "cli": [], "api": []}

# --- CLI: byte-identical stdout + exit code
cli_cases = []
for n in ["sample", "tie_rounding", "zero_input", "empty", "missing_field"]:
    cli_cases += [["report", f"logs/{n}.jsonl"], ["report", f"logs/{n}.jsonl", "--json"]]
for n, fmt in [("anthropic", "anthropic"), ("openai", "openai")]:
    cli_cases += [
        ["report", f"logs/{n}.jsonl", "--format", fmt],
        ["report", f"logs/{n}.jsonl", "--format", fmt, "--json"],
        ["report", f"logs/{n}.jsonl", "--format", fmt, "--turn-key", "turn", "--step-key", "step"],
        ["report", f"logs/{n}.jsonl", "--format", fmt, "--turn-key", "turn", "--step-key", "step", "--json"],
    ]
cli_cases += [
    ["report", "logs/anthropic_nested.jsonl", "--format", "anthropic", "--usage-key", "meta",
     "--turn-key", "t", "--step-key", "s"],
    ["report", "logs/anthropic_nested.jsonl", "--format", "anthropic", "--usage-key", "meta",
     "--turn-key", "t", "--step-key", "s", "--json"],
    ["report", "logs/openai_nested.jsonl", "--format", "openai"],
    ["report", "logs/openai_nested.jsonl", "--format", "openai", "--json"],
    ["report", "logs/bad_int.jsonl", "--format", "openai"],
    ["report", "logs/sample.jsonl", "--format", "bogus"],
    ["report"],
    [],
]
for a in cli_cases:
    out["cli"].append(cli(a))

# --- API: what-if for every (turn, step) in each native log plus a missing pair
for n in ["sample", "tie_rounding", "big_ints"]:
    reqs = parse.load_requests(f"logs/{n}.jsonl")
    pairs = [(r.turn, r.step) for r in reqs] + [(99, 0)]
    out["api"].append({
        "kind": "what_if", "log": f"logs/{n}.jsonl",
        "cases": [{"turn": t, "step": s, **attempt(lambda t=t, s=s: what_if_remove_step(reqs, t, s))}
                  for t, s in pairs],
    })

# --- API: load_requests (native, mapped, error), account
for n in ["sample", "tie_rounding", "zero_input", "bom", "crlf", "cr_only", "mixed_eol", "unicode_ws", "bom_crlf_comments", "big_ints", "huge_mean", "neg_zero", "duplicate_keys", "unicode_label", "extra_float_field", "nesting_ok"]:
    reqs = parse.load_requests(f"logs/{n}.jsonl")
    out["api"].append({"kind": "load_requests", "log": f"logs/{n}.jsonl", "mapping": None,
                       "requests": reqs_to_json(reqs), "metrics": account(reqs).to_dict(),
                       "fresh": [r.fresh_tokens for r in reqs]})
mapping = {"turn": "iteration", "step": "idx", "input_tokens": "prompt_tok", "cache_read_tokens": "cached"}
reqs = parse.load_requests("logs/mapped.jsonl", mapping)
out["api"].append({"kind": "load_requests", "log": "logs/mapped.jsonl", "mapping": mapping,
                   "requests": reqs_to_json(reqs), "metrics": account(reqs).to_dict(),
                   "fresh": [r.fresh_tokens for r in reqs]})
out["api"].append({"kind": "load_requests_error", "log": "logs/missing_field.jsonl"})
assert attempt(lambda: parse.load_requests("logs/missing_field.jsonl")).get("error")
for n in ["float_tokens", "float_integral", "exp_tokens", "bool_tokens", "str_tokens", "neg_tokens", "list_tokens",
          "float_optional", "float_turn", "bool_turn", "str_output", "numeric_label", "null_input", "nan_literal",
          "infinity_literal", "neg_infinity_literal", "nan_extra_field", "leading_zero", "plus_sign",
          "trailing_garbage", "non_object_list", "non_object_scalar", "deep_nesting", "bom_twice", "bom_mid_file",
          "invalid_utf8", "invalid_utf8_comment", "latin1", "utf16", "overlong_utf8", "surrogate_utf8",
          "huge_digits", "control_char_string"]:
    try:
        parse.load_requests(f"logs/{n}.jsonl")
    except Exception as e:  # noqa: BLE001
        from tokenledger.errors import classify
        out["api"].append({"kind": "load_requests_error", "log": f"logs/{n}.jsonl", "code": classify(e).code})
    else:
        raise AssertionError(f"{n} should be rejected")
out["api"].append({"kind": "account_empty_error"})
assert attempt(lambda: account([])).get("error")

# --- API: provider adapters with key variants
prov = [
    ("logs/anthropic.jsonl", "anthropic", {"turn_key": "turn", "step_key": "step"}),
    ("logs/anthropic.jsonl", "anthropic", {}),
    ("logs/openai.jsonl", "openai", {"turn_key": "turn", "step_key": "step"}),
    ("logs/openai.jsonl", "openai", {}),
    ("logs/anthropic_nested.jsonl", "anthropic", {"turn_key": "t", "step_key": "s", "usage_key": "meta"}),
    ("logs/openai_nested.jsonl", "openai", {}),
]
for log, fmt, kw in prov:
    reqs = adapters.load_provider(log, fmt, **kw)
    out["api"].append({"kind": "load_provider", "log": log, "fmt": fmt, "opts": kw,
                       "requests": reqs_to_json(reqs), "metrics": account(reqs).to_dict()})
out["api"].append({"kind": "load_provider_error", "log": "logs/bad_int.jsonl", "fmt": "openai", "opts": {}})
assert attempt(lambda: adapters.load_provider("logs/bad_int.jsonl", "openai")).get("error")
out["api"].append({"kind": "unknown_format_error", "fmt": "cohere"})
assert attempt(lambda: adapters.from_records([], "cohere")).get("error")

def _errorize(entry):
    """JSON error bodies embed implementation-specific message text: compare code + hint only."""
    try:
        e = json.loads(entry["stdout"])["error"]
    except (ValueError, KeyError, TypeError):
        return entry
    return {"args": entry["args"], "exit_code": entry["exit_code"],
            "error": {"code": e["code"], "hint": e["hint"]}}


out["cli"] = [_errorize(c) for c in out["cli"]]

# --- 0.2.0 behaviour: robustness (every previously diverging input) and CLI parity.
# Everything below compares stdout byte for byte (error messages included) plus the exit code.
def _record(a, full=True):
    """Full stdout comparison: error objects (message included) must be identical in both languages."""
    return cli(a)


out["robustness"] = []
for name in sorted(ROBUST_LOGS):
    fmt = []
    if name in ("float_openai.jsonl", "bool_openai.jsonl", "str_turn_key.jsonl", "missing_turn_key.jsonl"):
        fmt = ["--format", "openai"]
    if name in ("str_anthropic.jsonl", "anthropic_big.jsonl"):
        fmt = ["--format", "anthropic"]
    if name in ("str_turn_key.jsonl", "missing_turn_key.jsonl"):
        fmt += ["--turn-key", "turn", "--step-key", "step"]
    p = f"logs/{name}"
    out["robustness"] += [_record(["report", p] + fmt), _record(["report", p] + fmt + ["--json"]),
                          _record(["report", p] + fmt + ["--quiet"])]
for t, st in [(1, 0), (18446744073709551616, 0)]:
    base = ["what-if", "logs/big_ints.jsonl", "--turn", str(t), "--step", str(st)]
    out["robustness"] += [_record(base), _record(base + ["--json"]), _record(base + ["--quiet"])]
out["robustness"] += [
    _record(["what-if", "logs/neg_zero.jsonl", "--turn", "1", "--step", "0", "--json"]),
    _record(["what-if", "logs/bom.jsonl", "--turn", "1", "--step", "0"]),
]

SAMPLE = "logs/sample.jsonl"
parity = [
    ["x"], ["x", "--json"], ["--bogus"], ["--bogus", "--json"], ["--json"], ["--quiet"], ["--help"], ["-h"],
    ["-h", "--json"], ["x", "--help"], ["report", "--help"], ["report", "-h"], ["report", "-h", "--json"],
    ["what-if", "--help"], ["what-if", "-h"], ["describe", "--help"], ["describe", "-h"], ["mcp", "-h"],
    ["mcp", "--help"], ["report", SAMPLE, "--help"], ["report", "--format", "bogus", "-h"],
    ["report", SAMPLE, "--js"], ["report", SAMPLE, "--js", "--json"], ["report", SAMPLE, "--form", "openai"],
    ["report", SAMPLE, "--quie"], ["report", SAMPLE, "--turn-k", "t"], ["report", SAMPLE, "--format=openai"],
    ["report", SAMPLE, "--format=bogus"], ["report", SAMPLE, "--format"], ["report", SAMPLE, "--format", "--json"],
    ["report", SAMPLE, "--format", "bogus"], ["report", SAMPLE, "--format", "bogus", "--json"],
    ["report", SAMPLE, "--json=1"], ["report", SAMPLE, "--quiet=x", "--json"],
    ["report", "logs/nope.jsonl"], ["report", "logs/nope.jsonl", "--json"],
    ["report", ""], ["report", "", "--json"], ["report"], ["report", "--json"], ["report", SAMPLE, "extra"],
    ["report", SAMPLE, "extra", "--json"], ["report", "a", "b", "c", "--json"],
    ["what-if", SAMPLE], ["what-if", SAMPLE, "--json"], ["what-if", SAMPLE, "--turn", "1"],
    ["what-if", SAMPLE, "--turn", "1", "--json"], ["what-if", "--json"],
    ["what-if", SAMPLE, "--turn", "x", "--step", "0"], ["what-if", SAMPLE, "--turn", "x", "--step", "0", "--json"],
    ["what-if", SAMPLE, "--turn", "1.5", "--step", "0", "--json"], ["what-if", SAMPLE, "--turn", "", "--step", "0", "--json"],
    ["what-if", SAMPLE, "--turn", "1_0", "--step", "0", "--json"], ["what-if", SAMPLE, "--turn", " 1", "--step", "0", "--json"],
    ["what-if", SAMPLE, "--turn", "١", "--step", "0", "--json"],
    ["what-if", SAMPLE, "--turn", "+1", "--step", "0"], ["what-if", SAMPLE, "--turn", "01", "--step", "-0", "--json"],
    ["what-if", SAMPLE, "--turn", "-1", "--step", "0"], ["what-if", SAMPLE, "--turn", "-1", "--step", "0", "--json"],
    ["what-if", SAMPLE, "--turn=1", "--step=0"], ["what-if", SAMPLE, "--turn", "1", "--step", "0", "--turn", "2"],
    ["what-if", SAMPLE, "--tur", "1", "--step", "0", "--json"], ["what-if", SAMPLE, "--turn", "1", "--step"],
    ["what-if", "--turn", "1", "--step", "0", SAMPLE], ["what-if", SAMPLE, "--turn", "1", "--step", "0", "--format", "x"],
    ["what-if", SAMPLE, "--turn", "1", "--step", "0", "-x", "--json"], ["describe", "x"], ["describe", "x", "--json"],
    ["describe", "--format", "native"], ["describe", "--js"], ["describe", "--quiet"], ["mcp", "x"], ["mcp", "--json"],
    ["describe", "--help", "--bogus"], ["describe", "--bogus", "--help"], ["report", SAMPLE, "-q"], ["report", "-"],
    ["report", SAMPLE, "-"], ["report", SAMPLE, "--turn-key", "-1", "--format", "openai"],
    ["report", SAMPLE, "--turn-key", "-x"], ["report", SAMPLE, "--turn-key", "a b"], ["--", "report", SAMPLE],
    ["report", "--", SAMPLE], ["report", SAMPLE, "--"], ["Report", SAMPLE], ["report ", SAMPLE], ["what_if", SAMPLE],
]
out["cli_parity"] = [_record(a, full=True) for a in parity]

# --- agent surface (0.2.0): describe, what-if, --quiet, JSON errors, MCP session
from tokenledger import agent

out["agent_cli"] = []
agent_ok = [
    ["describe"], ["describe", "--json"], ["describe", "--quiet"],
    ["report", "logs/sample.jsonl", "--quiet"], ["report", "logs/sample.jsonl", "--json", "--quiet"],
    ["report", "logs/anthropic.jsonl", "--format", "anthropic", "--json", "--quiet"],
]
for t, st in [(1, 0), (1, 1), (2, 0)]:
    base = ["what-if", "logs/sample.jsonl", "--turn", str(t), "--step", str(st)]
    agent_ok += [base, base + ["--json"], base + ["--quiet"], base + ["--json", "--quiet"]]
agent_ok += [["what-if", "logs/anthropic.jsonl", "--format", "anthropic", "--turn-key", "turn",
              "--step-key", "step", "--turn", "1", "--step", "0", "--json"]]
for a in agent_ok:
    out["agent_cli"].append(cli(a))

# error cases: exit code + JSON {code, hint} compared; message text is implementation-specific
agent_err = [
    ["report", "logs/nope.jsonl", "--json"],
    ["report", "logs/bad_json.jsonl", "--json"],
    ["report", "logs/missing_field.jsonl", "--json"],
    ["report", "logs/empty.jsonl", "--json"],
    ["report", "logs/bad_int.jsonl", "--format", "openai", "--json"],
    ["what-if", "logs/sample.jsonl", "--turn", "99", "--step", "0", "--json"],
    ["report", "logs/sample.jsonl", "--format", "bogus", "--json"],
    ["report", "--json"],
    ["what-if", "logs/sample.jsonl", "--json"],
    ["report", "logs/nope.jsonl"],
    ["what-if", "logs/sample.jsonl", "--turn", "99", "--step", "0"],
]
for a in agent_err:
    r = cli(a)
    try:
        e = json.loads(r["stdout"])["error"]
        out["agent_cli"].append({"args": a, "exit_code": r["exit_code"],
                                 "error": {"code": e["code"], "hint": e["hint"]}})
    except ValueError:
        assert r["stdout"] == "" and r["exit_code"] == 1
        out["agent_cli"].append({"args": a, "exit_code": r["exit_code"], "stdout": ""})

mcp_in = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
     "params": {"name": "tokenledger_report", "arguments": {"logfile": "logs/sample.jsonl"}}},
    {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
     "params": {"name": "tokenledger_what_if", "arguments": {"logfile": "logs/sample.jsonl", "turn": 1, "step": 0}}},
    {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
     "params": {"name": "tokenledger_what_if", "arguments": {"logfile": "logs/sample.jsonl", "turn": 9, "step": 9}}},
    {"jsonrpc": "2.0", "id": 6, "method": "tools/call",
     "params": {"name": "tokenledger_report", "arguments": {"logfile": "logs/nope.jsonl"}}},
    {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "nope", "arguments": {}}},
    {"jsonrpc": "2.0", "id": 8, "method": "bogus/method"},
    {"jsonrpc": "2.0", "id": 9, "method": "ping"},
]
buf = io.StringIO()
agent.serve(io.StringIO("".join(json.dumps(m) + "\n" for m in mcp_in)), buf)
mcp_out = [json.loads(l) for l in buf.getvalue().splitlines()]
for m in mcp_out:  # message text of file errors is implementation-specific
    c = m.get("result", {}).get("structuredContent", {})
    if "error" in c:
        c["error"]["message"] = "<impl>"
# exact-text session: big integers, float ids, NaN, junk lines, Unicode whitespace around lines
mcp_extra_in = [
    json.dumps({"jsonrpc": "2.0", "id": 10, "method": "tools/call",
                "params": {"name": "tokenledger_report", "arguments": {"logfile": "logs/big_ints.jsonl"}}}),
    json.dumps({"jsonrpc": "2.0", "id": 11, "method": "tools/call",
                "params": {"name": "tokenledger_what_if",
                           "arguments": {"logfile": "logs/big_ints.jsonl", "turn": 18446744073709551616, "step": 0}}}),
    '{"jsonrpc": "2.0", "id": 1.0, "method": "ping"}',
    '{"jsonrpc": "2.0", "id": 12345678901234567890123, "method": "ping"}',
    '{"jsonrpc": "2.0", "id": 14, "method": "ping", "params": {"x": NaN}}',
    '{bad json',
    ' {"jsonrpc": "2.0", "id": 16, "method": "ping"}　',
    json.dumps({"jsonrpc": "2.0", "id": 17, "method": "tools/call",
                "params": {"name": "tokenledger_what_if",
                           "arguments": {"logfile": "logs/sample.jsonl", "turn": 1.0, "step": 0}}}),
    json.dumps({"jsonrpc": "2.0", "id": 18, "method": "tools/call",
                "params": {"name": "tokenledger_what_if",
                           "arguments": {"logfile": "logs/sample.jsonl", "turn": True, "step": 0}}}),
    json.dumps({"jsonrpc": "2.0", "id": 19, "method": "tools/call",
                "params": {"name": "tokenledger_report", "arguments": {"logfile": "logs/float_tokens.jsonl"}}}),
]
buf2 = io.StringIO()
agent.serve(io.StringIO("".join(x + chr(10) for x in mcp_extra_in)), buf2)
mcp_extra_out = buf2.getvalue().splitlines()
out["mcp"] = {"input": mcp_in, "output": mcp_out, "extra_input": mcp_extra_in, "extra_output": mcp_extra_out}

with open(os.path.join(HERE, "expected.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
    f.write("\n")
print(f"wrote expected.json: {len(out['cli'])} CLI cases, {len(out['api'])} API cases")
