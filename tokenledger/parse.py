"""Ingest request logs into a normalized Request model.

A log is JSONL: one JSON object per line, one object per LLM request (one step
within a turn). Unknown keys are ignored; missing optional fields default to 0.
Map a provider's own field names with the `mapping` argument.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class Request:
    """One LLM request = one step within a conversation turn."""
    turn: int                      # which turn (main-loop iteration) this belongs to
    step: int                      # step index within the turn (0-based)
    input_tokens: int              # total input tokens sent on this request (the context)
    cache_read_tokens: int = 0     # input tokens served from cache (re-read prefix)
    priming_tokens: int = 0        # fixed re-sent block: system prompt, tool schemas, agreements
    conversation_tokens: int = 0   # re-carried prior messages + tool results
    tool_result_tokens: int = 0    # tokens this step emitted as tool output (carried forward)
    output_tokens: int = 0
    label: str = ""

    @property
    def fresh_tokens(self) -> int:
        """Input tokens that were NOT served from cache on this request."""
        return max(self.input_tokens - self.cache_read_tokens, 0)


_FIELDS = {
    "turn", "step", "input_tokens", "cache_read_tokens", "priming_tokens",
    "conversation_tokens", "tool_result_tokens", "output_tokens", "label",
}


def _coerce(obj: dict, mapping: Optional[dict]) -> Request:
    if mapping:
        obj = {ours: obj.get(theirs) for ours, theirs in mapping.items()}
    kwargs = {k: v for k, v in obj.items() if k in _FIELDS and v is not None}
    for required in ("turn", "step", "input_tokens"):
        if required not in kwargs:
            raise ValueError(f"record missing required field '{required}': {obj}")
    return Request(**kwargs)


def load_requests(path: Union[str, Path], mapping: Optional[dict] = None) -> List[Request]:
    """Load a JSONL request log. Blank lines and lines starting with '#' are skipped."""
    out: List[Request] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(_coerce(json.loads(line), mapping))
    return out
