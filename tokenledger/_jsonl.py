"""Strict, shared JSONL reading and token-count validation (see docs/SPEC.md).

The JavaScript port implements the same rules byte for byte; conformance/expected.json locks them.

Rules
-----
* The file must be valid UTF-8 (``TL_E_BAD_ENCODING`` otherwise). One leading BOM is accepted.
* Lines end with LF, CRLF or a lone CR. Each line is trimmed of Unicode whitespace (Python's
  ``str.strip``); blank lines and lines starting with ``#`` are skipped.
* Every other line must be one JSON object. ``NaN``/``Infinity`` literals are invalid JSON here, an
  integer literal may have at most 4300 digits, and nesting deeper than 100 levels is rejected.
* A valid token count is a JSON integer >= 0: floats (``1.0``, ``1e3``), booleans, strings and negative
  numbers are ``TL_E_BAD_VALUE`` (``null`` means "absent").
"""
from __future__ import annotations

import json
import re
from typing import Any, List

from .errors import TokenledgerError

MAX_INT_DIGITS = 4300
MAX_DEPTH = 100
_LINE_BREAKS = re.compile(r"\r\n|\r|\n")


def _bad_constant(name: str) -> Any:
    raise json.JSONDecodeError(f"{name} is not valid JSON", name, 0)


def _parse_int(s: str) -> int:
    if len(s.lstrip("-")) > MAX_INT_DIGITS:
        raise TokenledgerError("TL_E_BAD_VALUE", f"integer literal has more than {MAX_INT_DIGITS} digits")
    return int(s)


def _check_depth(line: str) -> None:
    depth = 0
    in_str = False
    esc = False
    for ch in line:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "[{":
            depth += 1
            if depth > MAX_DEPTH:
                raise json.JSONDecodeError(f"nesting deeper than {MAX_DEPTH} levels", line, 0)
        elif ch in "]}":
            depth -= 1


def loads_strict(text: str) -> Any:
    """json.loads with the shared strictness (no NaN/Infinity, bounded integers and depth)."""
    _check_depth(text)
    return json.loads(text, parse_constant=_bad_constant, parse_int=_parse_int)


def show(value: Any) -> str:
    """A value as it appears in error messages (JSON text; dict/list by type name). Same in the JS port."""
    if isinstance(value, (dict, list)):
        return type(value).__name__
    return json.dumps(value, ensure_ascii=False)


def to_count(value: Any, field: str) -> int:
    """Validate a token count / turn / step: a JSON integer >= 0 (never bool, float or string)."""
    if type(value) is not int or value < 0:
        raise TokenledgerError(
            "TL_E_BAD_VALUE",
            f"field '{field}' must be a non-negative integer (JSON integer, not float/bool/string), got {show(value)}",
        )
    return value


def read_lines(path: Any) -> List[str]:
    """File -> lines, per the rules above (strict UTF-8, BOM, LF/CRLF/CR)."""
    with open(path, "rb") as f:
        data = f.read()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise TokenledgerError("TL_E_BAD_ENCODING", f"{path}: not valid UTF-8") from e
    if text.startswith("\ufeff"):
        text = text[1:]
    return _LINE_BREAKS.split(text)


def read_jsonl(path: Any) -> List[dict]:
    """The JSON objects of a JSONL file, in order."""
    out: List[dict] = []
    for n, raw in enumerate(read_lines(path), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        where = f"{path}: line {n}"
        try:
            obj = loads_strict(line)
        except json.JSONDecodeError as e:
            raise TokenledgerError("TL_E_BAD_JSON", f"{where}: invalid JSON") from e
        except TokenledgerError as e:
            raise TokenledgerError(e.code, f"{where}: {e.message}") from e
        if not isinstance(obj, dict):
            raise TokenledgerError(
                "TL_E_BAD_JSON", f"{where}: each line must be a JSON object, got {type(obj).__name__}"
            )
        out.append(obj)
    return out
