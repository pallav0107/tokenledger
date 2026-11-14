"""Regenerate the shared CLI help/usage text (tools.json -> "cli") from argparse's own output.

Usage:  python conformance/gen_cli_help.py        (writes tokenledger/tools.json; copy it to the JS repo)
        python conformance/gen_cli_help.py --check (exit 1 if tools.json is stale)

Both implementations print THIS text for -h/--help and in usage errors, so it cannot drift.
"""
import json
import os
import sys
from pathlib import Path

os.environ["COLUMNS"] = "80"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tokenledger import report  # noqa: E402

TOOLS = ROOT / "tokenledger" / "tools.json"
NAMES = ["main", "report", "what-if", "describe", "mcp"]


def native_texts() -> dict:
    report._Parser.native = True
    try:
        p = report.build_parser()
        subs = next(a for a in p._actions if a.dest == "cmd").choices
        parsers = {"main": p, **subs}
        return {
            "help": {n: parsers[n].format_help() for n in NAMES},
            "usage": {n: parsers[n].format_usage() for n in NAMES},
        }
    finally:
        report._Parser.native = False


def render(text: str, cli: dict) -> str:
    head = text[: text.index(',\n  "cli"')] if ',\n  "cli"' in text else text.rstrip().rstrip("}").rstrip()
    return head + ',\n  "cli": ' + json.dumps(cli, indent=2, ensure_ascii=False).replace("\n", "\n  ") + "\n}\n"


if __name__ == "__main__":
    text = TOOLS.read_text(encoding="utf-8")
    new = render(text, native_texts())
    if "--check" in sys.argv:
        sys.exit(0 if new == text else 1)
    TOOLS.write_text(new, encoding="utf-8", newline="\n")
    print("updated", TOOLS)
