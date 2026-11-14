"""CLI: report / what-if / describe / mcp. Human text by default, JSON with --json."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

from .account import Metrics, account, what_if_remove_step
from .agent import describe, load_log, serve
from .errors import TokenledgerError, classify, error_dict


def format_report(m: Metrics) -> str:
    lines = [
        "tokenledger — cost structure report",
        "=" * 42,
        f"requests (steps): {m.n_requests}",
        f"turns:            {m.n_turns}",
        f"steps per turn:   {m.steps_per_turn}",
        "",
        f"total input tokens:  {m.total_input:,}",
        f"total output tokens: {m.total_output:,}",
        "",
        f"cache-read:               {m.total_cache_read:,} ({m.cache_read_pct:.1%} of input)",
        f"priming re-sent:          {m.priming_total:,} ({m.priming_pct:.1%} of input)",
        f"re-carried conversation:  {m.conversation_total:,} ({m.conversation_pct:.1%} of input)",
        "",
        f"mean prefix / request: {m.mean_prefix:,.0f}    max prefix: {m.max_prefix:,}",
        f"prefix re-read cost:   {m.prefix_reread_cost:,} tokens",
        "",
        "takeaway: cost tracks (prefix size x requests), dominated by cache re-reads —",
        "not prompt size. Cut steps and reset context to move the bill.",
    ]
    return "\n".join(lines)


def format_quiet(m: Metrics) -> str:
    return (
        f"requests={m.n_requests} turns={m.n_turns} input={m.total_input} "
        f"output={m.total_output} cache_read_pct={m.cache_read_pct:.1%}"
    )


def _cli_int(value: str) -> int:
    """--turn/--step: an optionally signed ASCII integer (the JS port accepts exactly the same set)."""
    if not re.fullmatch(r"[+-]?[0-9]+", value):
        raise ValueError(value)
    return int(value)  # > 4300 digits raises ValueError on new Pythons; the JS port rejects it too


_cli_int.__name__ = "int"  # argparse says "invalid int value: 'x'"


def _shared_cli() -> dict:
    """Help/usage text shared with the JS port: tools.json -> "cli" (regenerate with conformance/gen_cli_help.py)."""
    with open(Path(__file__).with_name("tools.json"), "r", encoding="utf-8") as f:
        return json.load(f)["cli"]


class _Parser(argparse.ArgumentParser):
    """argparse that reports usage errors as JSON on stdout when --json was requested.

    Abbreviated options are rejected (allow_abbrev=False) and help/usage text comes from the shared
    tools.json so both languages print identical text. ``native=True`` uses argparse's own text
    (used only to generate and verify the shared text).
    """

    json_requested = False
    native = False

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)

    def _key(self) -> str:
        return "main" if self.prog == "tokenledger" else self.prog.split(" ", 1)[1]

    def format_help(self) -> str:
        return super().format_help() if _Parser.native else _shared_cli()["help"][self._key()]

    def format_usage(self) -> str:
        return super().format_usage() if _Parser.native else _shared_cli()["usage"][self._key()]

    def error(self, message: str):  # type: ignore[override]
        if _Parser.json_requested:
            exc = TokenledgerError("TL_E_USAGE", message)
            print(json.dumps({"error": error_dict(exc)}, indent=2))
            sys.exit(2)
        super().error(message)


def _add_log_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("logfile", help="path to a JSONL request log")
    sp.add_argument("--json", action="store_true", help="emit JSON instead of text")
    sp.add_argument("--quiet", action="store_true", help="compact output (one line / minimal JSON)")
    sp.add_argument(
        "--format",
        choices=["native", "anthropic", "openai"],
        default="native",
        help="log format: 'native' tokenledger fields (default), or a provider usage log",
    )
    sp.add_argument("--turn-key", help="record field naming the turn (provider formats)")
    sp.add_argument("--step-key", help="record field naming the step (provider formats)")
    sp.add_argument(
        "--usage-key",
        help="record field holding the usage sub-object (default: 'usage' if present, else top level)",
    )


def _fail(exc: BaseException, as_json: bool) -> int:
    err = classify(exc)
    if as_json:
        print(json.dumps({"error": error_dict(err)}, indent=2))
    else:
        print(f"tokenledger: error: {err.code}: {err.message}\nhint: {err.hint}", file=sys.stderr)
    return 1


def _dump(obj: dict, quiet: bool) -> str:
    return json.dumps(obj, separators=(",", ":")) if quiet else json.dumps(obj, indent=2)


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(
        prog="tokenledger",
        description="Measure the cost structure of multi-agent LLM systems from request logs.",
    )
    sub = p.add_subparsers(dest="cmd")
    rep = sub.add_parser("report", help="print a cost-structure report for a JSONL log")
    _add_log_args(rep)
    wi = sub.add_parser("what-if", help="input tokens saved by removing one (turn, step)")
    _add_log_args(wi)
    wi.add_argument("--turn", type=_cli_int, required=True, help="turn number as in the log")
    wi.add_argument("--step", type=_cli_int, required=True, help="step index within the turn")
    de = sub.add_parser("describe", help="print tool definitions (Anthropic, OpenAI, MCP) as JSON")
    de.add_argument("--json", action="store_true", help="accepted for uniformity; output is always JSON")
    de.add_argument("--quiet", action="store_true", help="compact JSON")
    sub.add_parser("mcp", help="run an MCP server on stdio exposing the tools")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    _Parser.json_requested = "--json" in argv_list
    p = build_parser()
    args = p.parse_args(argv_list)

    if args.cmd == "describe":
        print(_dump(describe(), args.quiet))
        return 0
    if args.cmd == "mcp":
        return serve(sys.stdin, sys.stdout)
    if args.cmd not in ("report", "what-if"):
        p.print_help()
        return 1

    try:
        requests = load_log(args.logfile, args.format, args.turn_key, args.step_key, args.usage_key)
        if args.cmd == "what-if":
            saved = what_if_remove_step(requests, args.turn, args.step)
            if args.json:
                text = _dump({"turn": args.turn, "step": args.step, "tokens_saved": saved}, args.quiet)
            elif args.quiet:
                text = str(saved)
            else:
                text = f"turn {args.turn} step {args.step}: removing it saves {saved:,} input tokens"
        else:
            metrics = account(requests)
            if args.json:
                text = _dump(metrics.to_dict(), args.quiet)
            elif args.quiet:
                text = format_quiet(metrics)
            else:
                text = format_report(metrics)
    except Exception as exc:  # noqa: BLE001 - classified into a stable code
        return _fail(exc, args.json)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
