"""CLI: `tokenledger report <log.jsonl>` → human or JSON cost-structure report."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from .account import Metrics, account
from .adapters import load_provider
from .parse import load_requests


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


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="tokenledger",
        description="Measure the cost structure of multi-agent LLM systems from request logs.",
    )
    sub = p.add_subparsers(dest="cmd")
    rep = sub.add_parser("report", help="print a cost-structure report for a JSONL log")
    rep.add_argument("logfile", help="path to a JSONL request log")
    rep.add_argument("--json", action="store_true", help="emit JSON instead of text")
    rep.add_argument(
        "--format",
        choices=["native", "anthropic", "openai"],
        default="native",
        help="log format: 'native' tokenledger fields (default), or a provider usage log",
    )
    rep.add_argument("--turn-key", help="record field naming the turn (provider formats)")
    rep.add_argument("--step-key", help="record field naming the step (provider formats)")
    rep.add_argument(
        "--usage-key",
        help="record field holding the usage sub-object (default: 'usage' if present, else top level)",
    )
    args = p.parse_args(argv)

    if args.cmd != "report":
        p.print_help()
        return 1

    if args.format == "native":
        requests = load_requests(args.logfile)
    else:
        requests = load_provider(
            args.logfile,
            args.format,
            turn_key=args.turn_key,
            step_key=args.step_key,
            usage_key=args.usage_key,
        )
    metrics = account(requests)
    if args.json:
        print(json.dumps(metrics.to_dict(), indent=2))
    else:
        print(format_report(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
