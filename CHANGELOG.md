# Changelog

## 0.2.0

Agent-ready release (describe, MCP server, what-if, JSON errors, schemas, AGENTS.md / llms.txt). It also makes
the Python and JavaScript packages behave identically on every input found by independent acceptance testing.
Behaviour changes versus 0.1.1 (documented in `docs/SPEC.md`):

- **Token counts must be non-negative JSON integers** (both languages, `TL_E_BAD_VALUE`, exit 1). 0.1.1 coerced
  floats, booleans and numeric strings with `int()`. Logs with `"input_tokens": "100"` or `1000.9` now fail.
- **`NaN` / `Infinity` literals are rejected** (`TL_E_BAD_JSON`); 0.1.1 Python accepted them, JS rejected them.
- **Invalid UTF-8 is rejected** with the new `TL_E_BAD_ENCODING` (0.1.1 JS substituted U+FFFD).
- **A UTF-8 BOM is accepted** (0.1.1 Python failed), as are CRLF and lone-CR line endings and Unicode
  whitespace around a line (0.1.1 JS failed on lone CR and on U+001C / NBSP padding).
- **A line that is not a JSON object** is `TL_E_BAD_JSON` (was `TL_E_BAD_VALUE`).
- **`label` must be a string** when present.
- **Exact totals above 2^53**: the JS CLI and MCP server sum with BigInt and print exact digits, identical to
  Python. The JS **library API** returns plain `number`s only (so `JSON.stringify` of any result works); a
  count or total above 2^53-1 throws the new `TL_E_OVERFLOW` (hint: use the CLI `--json`). The Python API
  keeps exact ints, so the JS API range is a documented divergence. Values that fit are unchanged.
- **Large floats** such as a `mean_prefix` of 1e22 now print every digit in JS, like Python.
- **CLI parity**: `allow_abbrev=False` in Python, so `--js` / `--form` are usage errors in both; an unknown
  subcommand exits 2 with empty stdout in both; help and usage text is shared (`tools.json` -> `cli`);
  `--turn` / `--step` accept only optionally signed ASCII integers.
- Error messages are identical in both languages (JSON errors name the file and line; file errors read
  `No such file or directory: 'path'`; `MISSING_FIELD` no longer dumps the record).
- Output is now built before anything is printed, so a late failure never leaves partial stdout.
