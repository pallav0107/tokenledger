# tokenledger

**Measure where the tokens actually go in a multi-agent / multi-model LLM system.**

Most teams reason about LLM cost as if it scales with *prompt size*. In a multi-step agent it
doesn't — it scales with **(context size × number of requests)**, because almost all input is a
**cache re-read of a growing prefix**, and it is **superlinear in step count**, because each step's
priming *and* its tool result get re-carried into every later step. `tokenledger` reads your request
logs and shows you exactly that, so you optimize the right thing.

> Status: **v0.1 — early scaffold, feedback very welcome.** Author: Pallav Pokhriyal.
> Zero dependencies, zero telemetry. It reads logs you already have.

## Install
```bash
pip install tokenledger-llm     # on PyPI; the import name and CLI are both `tokenledger`
# or from source:
pip install -e .
```

## Quickstart
```bash
tokenledger report examples/sample_logs/sample.jsonl
# or JSON:
tokenledger report examples/sample_logs/sample.jsonl --json
```

Example output:
```
tokenledger — cost structure report
==========================================
requests (steps): 4
turns:            2
steps per turn:   {1: 3, 2: 1}

total input tokens:  89,532
total output tokens: 505

cache-read:               88,432 (98.8% of input)
priming re-sent:          80,464 (89.9% of input)
re-carried conversation:  7,968 (8.9% of input)

mean prefix / request: 22,383    max prefix: 24,000
prefix re-read cost:   88,432 tokens

takeaway: cost tracks (prefix size x requests), dominated by cache re-reads —
not prompt size. Cut steps and reset context to move the bill.
```

## Log format
JSONL, one request (one step within a turn) per line. Required: `turn`, `step`, `input_tokens`.
Everything else is optional and defaults to 0:

| field | meaning |
|---|---|
| `turn` / `step` | which turn, and the step index within it |
| `input_tokens` | total input sent on this request (the context) |
| `cache_read_tokens` | input served from cache (the re-read prefix) |
| `priming_tokens` | fixed re-sent block: system prompt, tool schemas |
| `conversation_tokens` | re-carried prior messages + tool results |
| `tool_result_tokens` | tokens this step emitted as tool output (carried forward) |
| `output_tokens` | completion tokens |

Have a different log shape? Map it:
```python
from tokenledger import load_requests, account
reqs = load_requests("mylog.jsonl", mapping={"input_tokens": "prompt_tok", "turn": "iteration"})
print(account(reqs).cache_read_pct)
```

## Provider usage logs (no reshaping)

Point it straight at Anthropic or OpenAI usage records:
```bash
tokenledger report anthropic_usage.jsonl --format anthropic --turn-key turn --step-key step
tokenledger report openai_usage.jsonl   --format openai   --turn-key turn --step-key step
```
Each line is one request whose token counts live at top level or under a `usage` key.
One correctness detail the adapters handle for you:

- **Anthropic** reports `usage.input_tokens` as the *fresh* (uncached) input only; cached
  context is reported separately in `cache_read_input_tokens` / `cache_creation_input_tokens`.
  The adapter sums all three so `input_tokens` is the *true* per-request context — taking
  `input_tokens` alone silently undercounts the prefix, which is the whole thing being measured.
- **OpenAI** reports `usage.prompt_tokens` as the total input already, with the cached portion
  in `prompt_tokens_details.cached_tokens`.

If your records don't carry turn/step, omit the keys — each line becomes its own step in a single
turn (totals and percentages stay correct; the per-turn step structure just goes flat). In Python:
```python
from tokenledger import load_provider, account
reqs = load_provider("anthropic_usage.jsonl", "anthropic", turn_key="turn", step_key="step")
print(account(reqs).cache_read_pct)
```

## What-if: which step to cut
```python
from tokenledger import load_requests, what_if_remove_step
reqs = load_requests("examples/sample_logs/sample.jsonl")
print(what_if_remove_step(reqs, turn=1, step=0))  # tokens saved by removing an early step
```
Removing an **early** step saves more than a late one — it drops that step's own cost *and* its
tool result from every later step. That's the superlinearity, made concrete.

## Why it exists
See [`docs/METHODOLOGY.md`](https://github.com/pallav0107/tokenledger/blob/main/docs/METHODOLOGY.md) for the measured findings this operationalizes.

## Contributing
Issues and PRs welcome — especially log-format adapters for different providers/harnesses. Keep
contributions genuinely useful; no telemetry, no secrets in sample data (synthetic only).

## License
MIT — see [LICENSE](https://github.com/pallav0107/tokenledger/blob/main/LICENSE).
