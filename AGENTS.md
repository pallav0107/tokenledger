# tokenledger — agent guide

Reads LLM request logs (JSONL) and reports where input tokens go. Read-only: it never writes files,
never reads stdin (except `mcp`, which speaks JSON-RPC on stdin/stdout), never prompts, never uses
the network. Install: `pip install tokenledger-llm` (CLI and import name: `tokenledger`).

## Calling as a tool (MCP / function calls) - read first

- The log files named in the task are in the working directory: CALL THE TOOL, never ask the user to paste a log, never guess a number.
- A failing log is still a tool call: call `tokenledger_report` on it and quote `error.code` exactly from the result.
- Quote numbers as plain digits (`61700`, not `61,700`). Send only the arguments you need.

Worked examples (tool arguments -> result to read):
```
# "total_input and total_cache_read of log.jsonl (native)"
tokenledger_report {"logfile":"log.jsonl"}
  -> {"n_requests":3,"n_turns":1,...,"total_input":61700,"total_output":350,"total_cache_read":40500,...}
# "the exact error code for bad.jsonl" (a line lacks `step`)
tokenledger_report {"logfile":"bad.jsonl"}
  -> isError: {"error":{"code":"TL_E_MISSING_FIELD","message":"record missing required field 'step'","hint":"..."}}   answer: TL_E_MISSING_FIELD
# OpenAI usage log with turn/step fields: format + both keys (prompt_tokens summed into total_input)
tokenledger_report {"logfile":"oa.jsonl","format":"openai","turn_key":"turn","step_key":"step"}
  -> {"n_turns":2,"total_input":4300,"total_cache_read":2000,...}
# Anthropic usage log: format anthropic counts cached input (total_input 20300 for the example)
tokenledger_report {"logfile":"usage.jsonl","format":"anthropic","turn_key":"turn","step_key":"step"}
```

## Log format (JSONL, one request per line)
Required per line: `turn`, `step`, `input_tokens` (integers). Optional, default 0:
`cache_read_tokens`, `priming_tokens`, `conversation_tokens`, `tool_result_tokens`,
`output_tokens`, `label`. Blank lines and lines starting with `#` are skipped.
Counts, `turn` and `step` must be non-negative JSON integers: floats (`1.0`, `1e3`), booleans, strings
(`"100"`), negatives -> TL_E_BAD_VALUE (exit 1); `NaN`/`Infinity` -> TL_E_BAD_JSON; non-UTF-8 ->
TL_E_BAD_ENCODING. A UTF-8 BOM, CRLF and lone CR are accepted. Totals are exact above 2^53. Rules: docs/SPEC.md.
Provider logs: `--format anthropic|openai` (token counts at top level or under `usage`;
add `--turn-key`/`--step-key` if the records carry turn and step, else every line is its own step).

## Commands (all accept `--json` and `--quiet`)
- `tokenledger report LOGFILE [--format F] [--turn-key K] [--step-key K] [--usage-key K]`
  Prints the cost report. `--json` prints the object in `schemas/report.schema.json`
  (`n_requests, n_turns, steps_per_turn, total_input, total_output, total_cache_read,
  cache_read_pct, priming_total, priming_pct, conversation_total, conversation_pct,
  mean_prefix, max_prefix, prefix_reread_cost`; the `*_pct` values are fractions 0..1).
  `--quiet` prints one line (text) or compact JSON.
- `tokenledger what-if LOGFILE --turn T --step S [same format flags]`
  Input tokens saved by removing that step: `{"turn","step","tokens_saved"}`
  (schemas/what_if.schema.json). Saving = its input + its tool result times later steps in the turn.
- `tokenledger describe [--json]` Tool definitions: `{"anthropic":[...],"openai":[...],"mcp":[...]}`.
- `tokenledger mcp` MCP server on stdio (newline-delimited JSON-RPC). Tools:
  `tokenledger_report`, `tokenledger_what_if` (same arguments, snake_case: `logfile`, `format`,
  `turn_key`, `step_key`, `usage_key`, `turn`, `step`). Exits 0 when stdin closes.
All commands are read-only, so there is no `--dry-run`. Output is deterministic (no timestamps).
No colour is ever emitted; NO_COLOR and non-TTY need no handling.

## Exit codes
0 success. 1 expected failure (error object below). 2 usage error (bad or missing arguments).

## Errors
With `--json`, failures print to stdout (exit 1 or 2):
`{"error":{"code":"TL_E_...","message":"...","hint":"..."}}` (schemas/error.schema.json).
Without `--json` they print `tokenledger: error: CODE: message` and `hint: ...` to stderr.
Codes: TL_E_FILE_NOT_FOUND, TL_E_BAD_JSON, TL_E_MISSING_FIELD (native line lacks turn/step/
input_tokens), TL_E_BAD_VALUE, TL_E_BAD_ENCODING, TL_E_NO_REQUESTS (empty log), TL_E_UNKNOWN_FORMAT,
TL_E_STEP_NOT_FOUND (what-if turn/step not in log), TL_E_USAGE, TL_E_INTERNAL,
TL_E_OVERFLOW (JS library API only; Python ints are exact).
In MCP, failures return `isError: true` with the same error object as the text content.

## Examples
```
printf '{"turn":1,"step":0,"input_tokens":1000,"cache_read_tokens":0,"tool_result_tokens":200}\n{"turn":1,"step":1,"input_tokens":1400,"cache_read_tokens":1000}\n' > log.jsonl
tokenledger report log.jsonl --json          # total_input 2400, cache_read_pct 0.4166...
tokenledger report log.jsonl --quiet         # requests=2 turns=1 input=2400 output=0 cache_read_pct=41.7%
tokenledger what-if log.jsonl --turn 1 --step 0 --json   # {"tokens_saved": 1200, ...}
tokenledger report usage.jsonl --format anthropic --turn-key turn --step-key step --json
tokenledger describe --json
```

## Pitfalls
- Anthropic `usage.input_tokens` excludes cached input; use `--format anthropic` so cache tokens are summed.
- `turn` and `step` in `what-if` are the values in the log, not positions; step indexes are 0-based per turn.
- A native log with a line missing `turn`, `step` or `input_tokens` fails entirely (TL_E_MISSING_FIELD).
- Paths are relative to the current working directory; `--json` errors go to stdout, not stderr.
- Percent fields in JSON are fractions (0.988), in text they are shown as percentages (98.8%).
