"""Adapters: turn real provider usage logs into normalized Request records.

The point of tokenledger is that you measure *your own* system, so it has to
read the logs you already have. Providers report token usage differently, and
one difference matters for correctness:

  * Anthropic Messages API: `usage.input_tokens` is the FRESH (uncached) input
    only. Cached context is reported SEPARATELY in `cache_read_input_tokens`
    and `cache_creation_input_tokens`. So the *total* context sent on a request
    is the sum of all three — a naive reader who takes `input_tokens` alone
    undercounts the prefix, which is exactly the cost this tool is about.

  * OpenAI Chat Completions: `usage.prompt_tokens` is ALREADY the total input
    (it includes cached tokens); the cached portion is broken out under
    `usage.prompt_tokens_details.cached_tokens`.

These adapters normalize both into Request.input_tokens = total context sent.

Turn/step structure usually is NOT in a raw usage record. The superlinearity
metric (what_if_remove_step, steps_per_turn) depends on it, so you supply it:
pass turn_key/step_key (field names present in each record) or turn_fn/step_fn
(callables over the record). If you supply neither, every record is treated as
its own step within a single turn (step = file order), which still gives you
correct totals, cache-read % and priming %, but a flat step structure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union

from ._jsonl import read_jsonl, show, to_count
from .errors import TokenledgerError
from .parse import Request


def _usage(record: dict, usage_key: Optional[str]) -> dict:
    """Return the sub-dict holding token counts, or the record itself."""
    if usage_key and isinstance(record.get(usage_key), dict):
        return record[usage_key]
    if usage_key is None and isinstance(record.get("usage"), dict):
        return record["usage"]
    return record


def _int(d: dict, key: str, default: int = 0) -> int:
    v = d.get(key, default)
    return to_count(v, key) if v is not None else default


def _label(record: dict, label: str) -> str:
    if label:
        return label
    v = record.get("label")
    if v is None:
        return ""
    if not isinstance(v, str):
        raise TokenledgerError("TL_E_BAD_VALUE", f"field 'label' must be a string, got {show(v)}")
    return v


def _key_count(record: dict, key: str) -> int:
    v = record.get(key)
    if v is None:
        raise TokenledgerError("TL_E_BAD_VALUE", f"record has no value for key '{key}'")
    return to_count(v, key)


def from_anthropic_usage(
    record: dict, *, turn: int, step: int, label: str = "", usage_key: Optional[str] = None
) -> Request:
    """One Anthropic Messages API record -> Request.

    total input = input_tokens + cache_creation_input_tokens + cache_read_input_tokens
    (because Anthropic's `input_tokens` excludes cached context).
    """
    u = _usage(record, usage_key)
    fresh = _int(u, "input_tokens")
    cache_creation = _int(u, "cache_creation_input_tokens")
    cache_read = _int(u, "cache_read_input_tokens")
    total_input = fresh + cache_creation + cache_read
    return Request(
        turn=turn,
        step=step,
        input_tokens=total_input,
        cache_read_tokens=cache_read,
        output_tokens=_int(u, "output_tokens"),
        label=_label(record, label),
    )


def from_openai_usage(
    record: dict, *, turn: int, step: int, label: str = "", usage_key: Optional[str] = None
) -> Request:
    """One OpenAI Chat Completions record -> Request.

    `prompt_tokens` is already the total input; the cached portion is in
    `prompt_tokens_details.cached_tokens`.
    """
    u = _usage(record, usage_key)
    total_input = _int(u, "prompt_tokens")
    details = u.get("prompt_tokens_details") or {}
    cache_read = _int(details, "cached_tokens") if isinstance(details, dict) else 0
    return Request(
        turn=turn,
        step=step,
        input_tokens=total_input,
        cache_read_tokens=cache_read,
        output_tokens=_int(u, "completion_tokens"),
        label=_label(record, label),
    )


_CONVERTERS = {
    "anthropic": from_anthropic_usage,
    "openai": from_openai_usage,
}


def _turn_step_resolver(
    turn_key: Optional[str],
    step_key: Optional[str],
    turn_fn: Optional[Callable[[dict], int]],
    step_fn: Optional[Callable[[dict], int]],
):
    def resolve(record: dict, idx: int):
        if turn_fn is not None:
            turn = to_count(turn_fn(record), "turn")
        elif turn_key is not None:
            turn = _key_count(record, turn_key)
        else:
            turn = 1
        if step_fn is not None:
            step = to_count(step_fn(record), "step")
        elif step_key is not None:
            step = _key_count(record, step_key)
        else:
            step = idx
        return turn, step

    return resolve


def from_records(
    records: Iterable[dict],
    fmt: str,
    *,
    turn_key: Optional[str] = None,
    step_key: Optional[str] = None,
    turn_fn: Optional[Callable[[dict], int]] = None,
    step_fn: Optional[Callable[[dict], int]] = None,
    usage_key: Optional[str] = None,
) -> List[Request]:
    """Convert an iterable of provider usage dicts into Request records."""
    if fmt not in _CONVERTERS:
        raise TokenledgerError(
            "TL_E_UNKNOWN_FORMAT", f"unknown format '{fmt}'; expected one of {', '.join(sorted(_CONVERTERS))}"
        )
    convert = _CONVERTERS[fmt]
    resolve = _turn_step_resolver(turn_key, step_key, turn_fn, step_fn)
    out: List[Request] = []
    for idx, record in enumerate(records):
        turn, step = resolve(record, idx)
        out.append(convert(record, turn=turn, step=step, usage_key=usage_key))
    return out


def load_provider(
    path: Union[str, Path],
    fmt: str,
    *,
    turn_key: Optional[str] = None,
    step_key: Optional[str] = None,
    turn_fn: Optional[Callable[[dict], int]] = None,
    step_fn: Optional[Callable[[dict], int]] = None,
    usage_key: Optional[str] = None,
) -> List[Request]:
    """Load a JSONL usage log in a provider format into Request records.

    Same file rules as load_requests (docs/SPEC.md). Each remaining line is one JSON
    object carrying a usage record (either at top level or under a `usage` key); every
    token count must be a non-negative JSON integer.
    """
    records = read_jsonl(path)
    return from_records(
        records,
        fmt,
        turn_key=turn_key,
        step_key=step_key,
        turn_fn=turn_fn,
        step_fn=step_fn,
        usage_key=usage_key,
    )
