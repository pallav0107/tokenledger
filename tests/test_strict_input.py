"""0.2.0 behaviour: strict token counts, input robustness and CLI parity (shared with the JS port)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tokenledger import adapters, parse
from tokenledger.account import account, what_if_remove_step
from tokenledger.errors import TokenledgerError, classify
from tokenledger.report import main

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "conformance" / "logs"
L1 = b'{"turn": 1, "step": 0, "input_tokens": 1000, "tool_result_tokens": 200}'
L2 = b'{"turn": 1, "step": 1, "input_tokens": 1400, "cache_read_tokens": 1000}'


def w(tmp_path, data: bytes, name="log.jsonl"):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def code_of(path, fn=parse.load_requests, **kw):
    with pytest.raises(Exception) as e:
        fn(path, **kw)
    return classify(e.value).code


@pytest.mark.parametrize("value", ["1.5", "1.0", "1e3", "true", "false", '"100"', "-1", "[1]", "{}"])
def test_non_integer_token_counts_rejected(tmp_path, value):
    p = w(tmp_path, b'{"turn": 1, "step": 0, "input_tokens": ' + value.encode() + b"}\n")
    assert code_of(p) == "TL_E_BAD_VALUE"


@pytest.mark.parametrize("field", ["cache_read_tokens", "priming_tokens", "conversation_tokens",
                                   "tool_result_tokens", "output_tokens", "turn", "step"])
def test_every_count_field_is_validated(tmp_path, field):
    row = {"turn": 1, "step": 0, "input_tokens": 5, field: 2.5}
    assert code_of(w(tmp_path, json.dumps(row).encode() + b"\n")) == "TL_E_BAD_VALUE"


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_nan_and_infinity_literals_are_invalid_json(tmp_path, literal):
    p = w(tmp_path, b'{"turn": 1, "step": 0, "input_tokens": ' + literal.encode() + b"}\n")
    assert code_of(p) == "TL_E_BAD_JSON"


def test_null_means_absent(tmp_path):
    assert code_of(w(tmp_path, b'{"turn": 1, "step": 0, "input_tokens": null}\n')) == "TL_E_MISSING_FIELD"
    reqs = parse.load_requests(w(tmp_path, b'{"turn": 1, "step": 0, "input_tokens": 5, "output_tokens": null}\n'))
    assert reqs[0].output_tokens == 0


def test_provider_counts_are_strict(tmp_path):
    for body, fmt in [(b'{"usage": {"prompt_tokens": 12.5}}', "openai"), (b'{"usage": {"prompt_tokens": true}}', "openai"),
                      (b'{"usage": {"input_tokens": "100"}}', "anthropic"),
                      (b'{"usage": {"input_tokens": 1, "cache_read_input_tokens": -4}}', "anthropic")]:
        assert code_of(w(tmp_path, body + b"\n"), adapters.load_provider, fmt=fmt) == "TL_E_BAD_VALUE"
    p = w(tmp_path, b'{"turn": "1", "step": 0, "usage": {"prompt_tokens": 5}}\n')
    assert code_of(p, adapters.load_provider, fmt="openai", turn_key="turn", step_key="step") == "TL_E_BAD_VALUE"
    with pytest.raises(TokenledgerError):
        adapters.from_records([{"usage": {"prompt_tokens": 5}}], "openai", turn_fn=lambda r: 1.5)


def test_request_dataclass_validates():
    for bad in (1.5, True, "5", -1):
        with pytest.raises(TokenledgerError) as e:
            parse.Request(turn=1, step=0, input_tokens=bad)
        assert e.value.code == "TL_E_BAD_VALUE"
    with pytest.raises(TokenledgerError):
        parse.Request(turn=1, step=0, input_tokens=1, label=5)


def test_bom_crlf_cr_and_unicode_whitespace_are_accepted(tmp_path):
    want = parse.load_requests(w(tmp_path, L1 + b"\n" + L2 + b"\n", "lf.jsonl"))
    assert len(want) == 2
    variants = {
        "bom": b"\xef\xbb\xbf" + L1 + b"\n" + L2 + b"\n",
        "crlf": L1 + b"\r\n" + L2 + b"\r\n",
        "cr": L1 + b"\r" + L2 + b"\r",
        "mixed": L1 + b"\r\n" + L2 + b"\r",
        "unicode_ws": " ".encode() + L1 + "　".encode() + b"\n\x1c" + L2 + b"\x1f\n\xc2\x85\n",
    }
    for name, data in variants.items():
        assert parse.load_requests(w(tmp_path, data, name + ".jsonl")) == want, name


@pytest.mark.parametrize("data", [
    b"\xef\xbb\xbf\xef\xbb\xbf" + L1 + b"\n",   # two BOMs: only one is accepted
    L1 + b"\n\xef\xbb\xbf" + L2 + b"\n",         # BOM in the middle
    b"\xef\xbb\xbf" + L1 + b"\n#c\n[1]\n",      # a non-object line
])
def test_extra_boms_and_non_objects_are_bad_json(tmp_path, data):
    assert code_of(w(tmp_path, data)) == "TL_E_BAD_JSON"


@pytest.mark.parametrize("data", [
    L1 + b'\n{"turn": 1, "step": 1, "input_tokens": 5, "label": "\xff\xfe"}\n',
    b"# \xff comment\n" + L1 + b"\n",
    '{"turn": 1, "step": 0, "input_tokens": 5, "label": "café"}\n'.encode("latin-1"),
    (L1.decode() + "\n").encode("utf-16"),
    b'{"turn": 1, "step": 0, "input_tokens": 5, "label": "\xc0\xaf"}\n',
    b'{"turn": 1, "step": 0, "input_tokens": 5, "label": "\xed\xa0\x80"}\n',
])
def test_invalid_utf8_is_rejected_with_a_clear_code(tmp_path, data):
    assert code_of(w(tmp_path, data)) == "TL_E_BAD_ENCODING"
    assert code_of(w(tmp_path, data), adapters.load_provider, fmt="openai") == "TL_E_BAD_ENCODING"


def test_sums_are_exact_above_2_53():
    reqs = parse.load_requests(LOGS / "big_ints.jsonl")
    m = account(reqs)
    assert m.total_input == 9007199254740993 + 9007199254740999 + 123456789012345678901234567890
    assert m.total_output == 9007199254740995 + 9007199254740997 + 1
    assert what_if_remove_step(reqs, 1, 0) == 9007199254740993 + 9007199254740993 * 1


def test_huge_digit_and_deep_nesting_limits(tmp_path):
    assert code_of(w(tmp_path, b'{"turn": 1, "step": 0, "input_tokens": ' + b"9" * 4301 + b"}\n")) == "TL_E_BAD_VALUE"
    assert parse.load_requests(w(tmp_path, b'{"turn": 1, "step": 0, "input_tokens": ' + b"9" * 4300 + b"}\n"))
    deep = b'{"turn": 1, "step": 0, "input_tokens": 5, "x": ' + b"[" * 101 + b"]" * 101 + b"}\n"
    assert code_of(w(tmp_path, deep)) == "TL_E_BAD_JSON"
    ok = b'{"turn": 1, "step": 0, "input_tokens": 5, "x": ' + b"[" * 99 + b"]" * 99 + b"}\n"
    assert parse.load_requests(w(tmp_path, ok))


# ---------------------------------------------------------------- CLI parity
def run_cli(args, capsys):
    try:
        code = main(args)
    except SystemExit as e:
        code = e.code
    cap = capsys.readouterr()
    return code, cap.out, cap.err


SAMPLE = str(ROOT / "examples" / "sample_logs" / "sample.jsonl")


@pytest.mark.parametrize("args", [
    ["report", SAMPLE, "--js"], ["report", SAMPLE, "--form", "openai"], ["report", SAMPLE, "--quie"],
    ["report", SAMPLE, "--turn-k", "t"], ["what-if", SAMPLE, "--tur", "1", "--step", "0"],
    ["describe", "--js"], ["x"], ["rep", SAMPLE],
])
def test_abbreviations_and_unknown_commands_are_usage_errors(args, capsys):
    code, out, err = run_cli(args, capsys)
    assert code == 2 and out == "" and "error:" in err


@pytest.mark.parametrize("args", [["--help"], ["-h"], ["report", "-h"], ["what-if", "--help"], ["describe", "-h"], ["mcp", "-h"]])
def test_help_comes_from_the_shared_tools_json(args, capsys):
    code, out, _ = run_cli(args, capsys)
    cli = json.loads((ROOT / "tokenledger" / "tools.json").read_text(encoding="utf-8"))["cli"]
    assert code == 0
    assert out == cli["help"]["main" if len(args) == 1 else args[0]]


def test_shared_help_matches_argparse_native_output():
    """tools.json 'cli' must be exactly what argparse prints (regenerate: conformance/gen_cli_help.py)."""
    if sys.version_info < (3, 10):
        pytest.skip("argparse wording differs before 3.10")
    r = subprocess.run([sys.executable, str(ROOT / "conformance" / "gen_cli_help.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, "tokenledger/tools.json is stale: run python conformance/gen_cli_help.py"


def test_unknown_subcommand_and_bad_format_exit_codes(capsys):
    assert run_cli([], capsys)[0] == 1
    code, out, _ = run_cli(["x", "--json"], capsys)
    assert code == 2 and json.loads(out)["error"]["code"] == "TL_E_USAGE"
    code, out, _ = run_cli(["report", SAMPLE, "--format", "bogus", "--json"], capsys)
    assert code == 2 and json.loads(out)["error"]["code"] == "TL_E_USAGE"
    code, out, err = run_cli(["report", "nope.jsonl"], capsys)
    assert code == 1 and out == "" and "TL_E_FILE_NOT_FOUND" in err


def test_turn_and_step_are_ascii_integers(capsys):
    for bad in ["1.5", "1_0", " 1", "x", "", "١"]:
        code, out, _ = run_cli(["what-if", SAMPLE, "--turn", bad, "--step", "0"], capsys)
        assert code == 2 and out == "", bad
    code, out, _ = run_cli(["what-if", SAMPLE, "--turn", "+01", "--step", "-0", "--json"], capsys)
    assert code == 0 and json.loads(out) == {"turn": 1, "step": 0, "tokens_saved": 22216}
