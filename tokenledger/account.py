"""The accounting math: where the tokens actually go.

Core claims this quantifies (see docs/METHODOLOGY.md):
  1. input cost tracks (prefix size x number of requests), dominated by cache re-reads;
  2. cost is superlinear in step count, because each step's priming AND its tool
     result are re-carried into every later step of the turn.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

from .errors import StepNotFoundError, TokenledgerError
from .parse import Request


@dataclass
class Metrics:
    n_requests: int
    n_turns: int
    steps_per_turn: Dict[int, int]
    total_input: int
    total_output: int
    total_cache_read: int
    cache_read_pct: float
    priming_total: int
    priming_pct: float
    conversation_total: int
    conversation_pct: float
    mean_prefix: float
    max_prefix: int
    prefix_reread_cost: int  # tokens spent re-reading already-seen (cached) context

    def to_dict(self) -> dict:
        return asdict(self)


def _pct(part: int, whole: int) -> float:
    return (part / whole) if whole else 0.0


def account(requests: List[Request]) -> Metrics:
    if not requests:
        raise TokenledgerError("TL_E_NO_REQUESTS", "no requests to account")

    total_input = sum(r.input_tokens for r in requests)
    total_output = sum(r.output_tokens for r in requests)
    total_cache = sum(r.cache_read_tokens for r in requests)
    priming_total = sum(r.priming_tokens for r in requests)
    conversation_total = sum(r.conversation_tokens for r in requests)

    turns: Dict[int, List[Request]] = {}
    for r in requests:
        turns.setdefault(r.turn, []).append(r)
    steps_per_turn = {t: len(rs) for t, rs in sorted(turns.items())}

    prefixes = [r.input_tokens for r in requests]
    return Metrics(
        n_requests=len(requests),
        n_turns=len(turns),
        steps_per_turn=steps_per_turn,
        total_input=total_input,
        total_output=total_output,
        total_cache_read=total_cache,
        cache_read_pct=_pct(total_cache, total_input),
        priming_total=priming_total,
        priming_pct=_pct(priming_total, total_input),
        conversation_total=conversation_total,
        conversation_pct=_pct(conversation_total, total_input),
        mean_prefix=sum(prefixes) / len(prefixes),
        max_prefix=max(prefixes),
        prefix_reread_cost=total_cache,
    )


def what_if_remove_step(requests: List[Request], turn: int, step: int) -> int:
    """Estimated input tokens saved by removing (turn, step).

    Saving = the removed request's own input, PLUS its tool result, which would
    have been re-carried into every later step of the same turn. This is why
    removing an EARLY step saves more than a late one: the superlinear effect.
    """
    same_turn = [r for r in requests if r.turn == turn]
    target = next((r for r in same_turn if r.step == step), None)
    if target is None:
        raise StepNotFoundError(f"no step {step} in turn {turn}")
    later_steps = [r for r in same_turn if r.step > step]
    return target.input_tokens + target.tool_result_tokens * len(later_steps)
