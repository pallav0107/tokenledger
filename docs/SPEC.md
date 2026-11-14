# tokenledger behaviour specification (0.2.0)

The Python package (`tokenledger-llm` on PyPI) and the JavaScript package (`tokenledger-llm` on npm) implement
exactly this. `conformance/expected.json` is generated from the Python reference and replayed against both;
a divergence is a bug in whichever side differs from this document.

## Log files

- **Encoding.** Strict UTF-8. Anything else (Latin-1, UTF-16, overlong or surrogate sequences) fails with
  `TL_E_BAD_ENCODING`; bytes are never silently replaced. One leading UTF-8 BOM is accepted and ignored; a BOM
  anywhere else is invalid JSON.
- **Lines.** LF, CRLF and a lone CR all end a line (files may mix them). Each line is trimmed of Unicode
  whitespace as Python's `str.strip()` does (tab, VT, FF, U+001C..U+001F, NBSP, U+0085, U+2000..U+200A,
  U+2028/2029, U+202F, U+205F, U+3000 ...; **not** U+FEFF). Blank lines and lines starting with `#` are skipped.
- **JSON.** Every other line is one JSON object (`TL_E_BAD_JSON` if it is not valid JSON or not an object).
  `NaN`, `Infinity` and `-Infinity` are invalid. Duplicate keys: the last wins. Integer literals have at most
  4300 digits (`TL_E_BAD_VALUE` beyond that); nesting deeper than 100 levels is `TL_E_BAD_JSON`.
- **Empty log.** No records after skipping: `TL_E_NO_REQUESTS`.

## Token counts

A valid token count (and `turn` / `step`) is a **non-negative JSON integer**. Rejected with `TL_E_BAD_VALUE`
(exit code 1, in both languages): floats including `1.0` and `1e3`, booleans, strings (even `"100"`), lists,
objects and negative numbers. `null` means "absent": optional fields default to 0, a required field that is
`null` is `TL_E_MISSING_FIELD`. `label` must be a string or absent. This applies to native logs, the provider
formats (`--format anthropic|openai`), `--turn-key` / `--step-key` values, `turn_fn` / `step_fn` results and
the `Request` constructor.

## Arithmetic and output

- Sums are exact integers. Python uses `int` everywhere. JavaScript sums with `BigInt`; the **CLI and the
  MCP server** print the exact digits (identical to Python, any size), so totals above 2^53 match Python.
- **JS library API (documented divergence).** `loadRequests`, `account`, `whatIfRemoveStep`, the adapters and
  the tool functions return plain `number`s only, so `JSON.stringify(result)` always works. Any count, total
  or result above `Number.MAX_SAFE_INTEGER` (2^53-1) throws `TL_E_OVERFLOW` (hint: use the CLI with `--json`,
  which prints exact digits). The Python API has no such limit and returns exact `int`s.
- Ratios (`*_pct`, `mean_prefix`) are the correctly rounded IEEE double of the exact quotient (Python's
  `int / int`); a quotient too large for a double is `TL_E_BAD_VALUE`.
- Output is byte-identical: `json.dumps` formatting (`indent=2`, or compact separators with `--quiet`),
  floats as Python `repr` (`0.0`, `1e-05`, `1e+22`), integers in full, `{:,}` grouping in text.
  Printing an integer of more than 4300 digits is `TL_E_BAD_VALUE` (Python refuses to).

## Command line

- Exit codes: `0` success (and `-h`), `1` expected failure (a `TL_E_*` error object) or no subcommand,
  `2` usage error. A usage error prints nothing on stdout unless `--json` was given, then a `TL_E_USAGE` error.
- No abbreviations: `--js`, `--form`, `--turn-k` and subcommand prefixes are usage errors (argparse
  `allow_abbrev=False` in Python; the JS parser implements the same grammar).
- `--turn` / `--step` take an optionally signed ASCII integer (`+1`, `-0`, `007`); `1.5`, `1_0`, ` 1` and
  non-ASCII digits are usage errors.
- Help and usage text is stored once in `tools.json` (`"cli"`), generated from argparse by
  `conformance/gen_cli_help.py` and printed verbatim by both implementations.

## Error codes

`TL_E_FILE_NOT_FOUND`, `TL_E_BAD_JSON`, `TL_E_BAD_ENCODING`, `TL_E_MISSING_FIELD`, `TL_E_BAD_VALUE`,
`TL_E_NO_REQUESTS`, `TL_E_UNKNOWN_FORMAT`, `TL_E_STEP_NOT_FOUND`, `TL_E_USAGE`, `TL_E_INTERNAL`, and
`TL_E_OVERFLOW` (JavaScript library API only; the CLI, the MCP server and Python never raise it).
Error messages are part of the contract too: the `--json` error object (code, message, hint) and every usage
message are byte-identical in both implementations, and conformance compares them. File errors read
`<strerror>: '<path>'` (e.g. `No such file or directory: 'x.jsonl'`) and JSON errors name the file and 1-based line.
