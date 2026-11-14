"""Stable, machine-readable errors: every expected failure is {code, message, hint}."""
from __future__ import annotations

import json
from typing import Dict

HINTS: Dict[str, str] = {
    "TL_E_FILE_NOT_FOUND": "Check the path; it is resolved against the current working directory.",
    "TL_E_BAD_JSON": "Each non-blank line must be one JSON object (JSONL); lines starting with # are skipped.",
    "TL_E_MISSING_FIELD": "Native logs need turn, step and input_tokens per line; for provider logs pass --format anthropic or openai.",
    "TL_E_BAD_VALUE": "Token counts, turn and step must be non-negative JSON integers (no floats, booleans, strings or NaN/Infinity).",
    "TL_E_OVERFLOW": "The value exceeds the JavaScript API's safe integer range (2^53-1); use the CLI with --json, which prints exact digits (or the Python package).",
    "TL_E_BAD_ENCODING": "The log must be valid UTF-8 (a leading BOM is fine); re-export it as UTF-8.",
    "TL_E_NO_REQUESTS": "The log has no records; check the file and the --format value.",
    "TL_E_UNKNOWN_FORMAT": "Use one of: native, anthropic, openai.",
    "TL_E_STEP_NOT_FOUND": "Run `tokenledger report LOG --json` and read steps_per_turn for valid turn/step pairs.",
    "TL_E_USAGE": "Run `tokenledger describe --json` or `tokenledger --help` for the exact arguments.",
    "TL_E_INTERNAL": "Unexpected failure; re-run with the same input and report it with the full command.",
}


class TokenledgerError(ValueError):
    """Expected failure with a stable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def __str__(self) -> str:
        return self.message

    @property
    def hint(self) -> str:
        return HINTS[self.code]


class StepNotFoundError(TokenledgerError, KeyError):
    """what_if_remove_step on a (turn, step) that is not in the log (also a KeyError)."""

    def __init__(self, message: str) -> None:
        super().__init__("TL_E_STEP_NOT_FOUND", message)


def classify(exc: BaseException) -> TokenledgerError:
    """Map any exception raised while loading/accounting to a TokenledgerError."""
    if isinstance(exc, TokenledgerError):
        return exc
    if isinstance(exc, json.JSONDecodeError):
        return TokenledgerError("TL_E_BAD_JSON", str(exc))
    if isinstance(exc, OSError):
        if exc.strerror:
            return TokenledgerError("TL_E_FILE_NOT_FOUND", f"{exc.strerror}: '{exc.filename}'")
        return TokenledgerError("TL_E_FILE_NOT_FOUND", str(exc))
    if isinstance(exc, RecursionError):
        return TokenledgerError("TL_E_BAD_JSON", "JSON nesting too deep")
    if isinstance(exc, OverflowError):
        return TokenledgerError("TL_E_BAD_VALUE", "integer too large to convert to a float")
    if isinstance(exc, (ValueError, TypeError, AttributeError, KeyError)):
        return TokenledgerError("TL_E_BAD_VALUE", str(exc))
    return TokenledgerError("TL_E_INTERNAL", f"{type(exc).__name__}: {exc}")


def error_dict(exc: TokenledgerError) -> dict:
    return {"code": exc.code, "message": exc.message, "hint": exc.hint}
