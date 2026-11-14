"""Agent surface: tool definitions (one source), tool execution, and the stdio MCP server."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, IO, List, Optional

from ._jsonl import loads_strict
from .account import account, what_if_remove_step
from .adapters import load_provider
from .errors import TokenledgerError, classify, error_dict
from .parse import Request, load_requests

TOOLS_PATH = Path(__file__).with_name("tools.json")
PROTOCOL_VERSION = "2025-06-18"
FORMATS = ("native", "anthropic", "openai")


def load_tools() -> List[Dict[str, Any]]:
    """The single source of truth for tool definitions (shared byte-for-byte with the JS port)."""
    with open(TOOLS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["tools"]


def describe() -> Dict[str, List[Dict[str, Any]]]:
    """Tool definitions in Anthropic, OpenAI and MCP formats, generated from tools.json."""
    tools = load_tools()
    return {
        "anthropic": [
            {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
            for t in tools
        ],
        "openai": [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ],
        "mcp": [
            {"name": t["name"], "description": t["description"], "inputSchema": t["input_schema"]}
            for t in tools
        ],
    }


def load_log(
    logfile: str,
    fmt: str = "native",
    turn_key: Optional[str] = None,
    step_key: Optional[str] = None,
    usage_key: Optional[str] = None,
) -> List[Request]:
    """Load a log; every failure is raised as TokenledgerError."""
    try:
        if fmt == "native":
            return load_requests(logfile)
        return load_provider(logfile, fmt, turn_key=turn_key, step_key=step_key, usage_key=usage_key)
    except Exception as exc:  # noqa: BLE001 - classified into a stable code
        raise classify(exc) from exc


def _opt(args: dict, key: str) -> Optional[str]:
    v = args.get(key)
    if v is not None and not isinstance(v, str):
        raise TokenledgerError("TL_E_USAGE", f"argument '{key}' must be a string")
    return v


def _int_arg(args: dict, key: str) -> int:
    v = args.get(key)
    if isinstance(v, bool) or not isinstance(v, int):
        raise TokenledgerError("TL_E_USAGE", f"argument '{key}' is required and must be an integer")
    return v


def _requests_from_args(args: dict) -> List[Request]:
    logfile = args.get("logfile")
    if not isinstance(logfile, str) or not logfile:
        raise TokenledgerError("TL_E_USAGE", "argument 'logfile' is required and must be a string")
    fmt = args.get("format") or "native"
    if fmt not in FORMATS:
        raise TokenledgerError(
            "TL_E_UNKNOWN_FORMAT", f"unknown format '{fmt}'; expected one of {', '.join(sorted(FORMATS))}"
        )
    return load_log(logfile, fmt, _opt(args, "turn_key"), _opt(args, "step_key"), _opt(args, "usage_key"))


def tool_report(args: dict) -> dict:
    try:
        return account(_requests_from_args(args)).to_dict()
    except Exception as exc:  # noqa: BLE001
        raise classify(exc) from exc


def tool_what_if(args: dict) -> dict:
    turn, step = _int_arg(args, "turn"), _int_arg(args, "step")
    try:
        saved = what_if_remove_step(_requests_from_args(args), turn, step)
    except Exception as exc:  # noqa: BLE001
        raise classify(exc) from exc
    return {"turn": turn, "step": step, "tokens_saved": saved}


TOOL_FUNCS = {"tokenledger_report": tool_report, "tokenledger_what_if": tool_what_if}


def call_tool(name: str, args: Optional[dict]) -> dict:
    """Run one tool; raises TokenledgerError on any expected failure."""
    if name not in TOOL_FUNCS:
        raise TokenledgerError("TL_E_USAGE", f"unknown tool '{name}'; expected one of {', '.join(sorted(TOOL_FUNCS))}")
    if args is not None and not isinstance(args, dict):
        raise TokenledgerError("TL_E_USAGE", "tool arguments must be an object")
    return TOOL_FUNCS[name](args or {})


def _compact(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _version() -> str:
    from . import __version__
    return __version__


def handle_message(msg: Any) -> Optional[dict]:
    """Handle one JSON-RPC message; returns the response, or None for notifications."""
    if not isinstance(msg, dict) or "method" not in msg:
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}
    method, mid, params = msg["method"], msg.get("id"), msg.get("params") or {}
    if "id" not in msg:
        return None  # notification (e.g. notifications/initialized)

    def ok(result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    if method == "initialize":
        proto = params.get("protocolVersion") if isinstance(params, dict) else None
        return ok({
            "protocolVersion": proto if isinstance(proto, str) else PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "tokenledger", "version": _version()},
        })
    if method == "ping":
        return ok({})
    if method == "tools/list":
        return ok({"tools": describe()["mcp"]})
    if method == "tools/call":
        if not isinstance(params, dict) or not isinstance(params.get("name"), str):
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": "name is required"}}
        try:
            result = call_tool(params["name"], params.get("arguments"))
            return ok({"content": [{"type": "text", "text": _compact(result)}],
                       "structuredContent": result, "isError": False})
        except TokenledgerError as exc:
            err = error_dict(exc)
            return ok({"content": [{"type": "text", "text": _compact({"error": err})}],
                       "structuredContent": {"error": err}, "isError": True})
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Method not found: {method}"}}


def serve(stdin: IO[str], stdout: IO[str]) -> int:
    """Newline-delimited JSON-RPC over stdio; returns 0 on stdin EOF."""
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = loads_strict(line)
        except (json.JSONDecodeError, TokenledgerError):
            resp: Optional[dict] = {"jsonrpc": "2.0", "id": None,
                                    "error": {"code": -32700, "message": "Parse error"}}
        else:
            resp = handle_message(msg)
        if resp is not None:
            stdout.write(_compact(resp) + "\n")
            stdout.flush()
    return 0
