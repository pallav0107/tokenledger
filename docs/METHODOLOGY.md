# Methodology — the cost model tokenledger measures

These are the findings `tokenledger` operationalizes. The absolute numbers come from one production
harness and are **not** universal constants — the tool exists so you can measure *your own*.

## 1. Cost ≈ (context size × number of requests)
In a multi-step agent, the dominant input cost is not the marginal prompt of one call — it's the
**repeated re-processing of a conversation that only grows**. On the reference harness, **~97% of
input tokens were cache-reads**: the model re-reading a cached prefix it had already seen. Each
request re-reads the whole prefix that exists at that point, so:

```
input cost ≈ Σ_requests (prefix size at that request) ≈ (typical context) × (number of requests)
```

Consequence: **requests/turns are a first-class cost driver**, independent of prompt wording; and a
large prefix is paid for *every* subsequent request.

## 2. Steps are superlinear
Every request re-sends a fixed **priming** block (system prompt, tool schemas, agreements) —
measured at **~20,116 tokens/request** on the reference harness. And prior steps' messages and
**tool results** are re-carried into each later step (measured ~70% of input). So adding a step
adds its priming *and* pins its tool output into every following step. Removing an **early** step
therefore saves more than removing a late one — `what_if_remove_step()` quantifies this.

## 3. The levers that actually move the bill
- reset context when switching tasks (don't re-pay a big prefix on every later request);
- compact sessions once the prefix dominates;
- prefer ranged reads over whole-file reads (a whole file is re-sent every later step of the turn);
- collapse many small steps into fewer, larger-value ones.

## 4. Measure before optimizing
Every claim above began as an assumption the logs corrected — at least once in the *opposite*
direction from the prediction. That's the whole point of measuring rather than guessing.
