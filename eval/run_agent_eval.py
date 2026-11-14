#!/usr/bin/env python3
"""AGENT TASK EVAL harness (AGENT_UX_SPEC item 8). Stdlib only.

A local open-weight model behind an OpenAI-compatible endpoint gets ONLY llms.txt + AGENTS.md
(system prompt) and the tool definitions (`<tool> describe --json`, OpenAI format). Tool calls are
executed against the real MCP server (`<tool> mcp`) with its cwd set to a throwaway sandbox that
holds the task's fixtures. Each task is run --runs times (>= 3); a task counts as passed in a run
when every expectation holds. The pass bar (default 0.80 of task-runs) is fixed on the command
line / in this file BEFORE running and written into the results next to the model name and prompts.

Task files (eval/agent_tasks/*.yaml) are JSON-compatible YAML (JSON is YAML 1.2):
  {"id": str, "prompt": str, "fixtures": {"rel/path": "text content"},
   "expect": {"tool_called": "name", "args_subset": {"k": v}, "answer_contains": ["str" | ["any", "of"]], "tool_calls": [{"name", "args_subset"}], "allow_error": bool,
              "files_exist": ["rel"], "files_absent": ["rel"],
              "files_unchanged": {"rel": "text content"}}}

Usage: python eval/run_agent_eval.py --tool-cmd "tokenledger" --endpoint http://HOST:PORT/v1 \
         --model NAME [--runs 3] [--bar 0.8] [--out eval/results.json]
(--tool-cmd is the CLI prefix; the runner appends `describe --json` and `mcp`.)
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
PASS_BAR = 0.80
MIN_RUNS = 3
DEFAULT_ENDPOINT = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
DEFAULT_MODELS = ["gpt-oss:20b", "glm-4.7-flash"]
TEMPERATURE = 0.2
MAX_TOOL_CALLS = 20


def load_tasks(task_dir: Path) -> List[Dict[str, Any]]:
    tasks = []
    for p in sorted(task_dir.glob("*.yaml")):
        text = p.read_text(encoding="utf-8")
        try:
            import yaml  # type: ignore
            tasks.append(yaml.safe_load(text))
        except ImportError:
            tasks.append(json.loads(text))
    return tasks


class McpClient:
    """Minimal newline-delimited JSON-RPC client for a stdio MCP server."""

    def __init__(self, cmd: List[str], cwd: str) -> None:
        self.p = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, encoding="utf-8")
        self._id = 0
        self.call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                 "clientInfo": {"name": "agent-eval", "version": "1"}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _send(self, msg: dict) -> None:
        assert self.p.stdin is not None
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()

    def call(self, method: str, params: Optional[dict] = None) -> dict:
        self._id += 1
        self._send({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}})
        assert self.p.stdout is not None
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("MCP server closed stdout")
            msg = json.loads(line)
            if msg.get("id") == self._id:
                return msg

    def close(self) -> None:
        try:
            if self.p.stdin:
                self.p.stdin.close()
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def chat(endpoint: str, model: str, messages: list, tools: list, timeout: float) -> dict:
    body = json.dumps({"model": model, "messages": messages, "tools": tools,
                       "temperature": TEMPERATURE, "stream": False}).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]


def run_once(task: dict, base_cmd: List[str], system: str, tools: list, endpoint: str,
             model: str, max_calls: int, timeout: float) -> dict:
    with tempfile.TemporaryDirectory(prefix="agent_eval_") as sandbox:
        for rel, content in (task.get("fixtures") or {}).items():
            fp = Path(sandbox, rel)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
        mcp = McpClient(base_cmd + ["mcp"], sandbox)
        calls: List[dict] = []
        messages: list = [{"role": "system", "content": system},
                          {"role": "user", "content": task["prompt"]}]
        answer = ""
        try:
            while True:
                msg = chat(endpoint, model, messages, tools, timeout)
                messages.append(msg)
                tcs = msg.get("tool_calls") or []
                if not tcs:
                    answer = msg.get("content") or ""
                    break
                for tc in tcs:
                    if len(calls) >= max_calls:
                        break
                    fn = tc["function"]
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    res = mcp.call("tools/call", {"name": fn["name"], "arguments": args})
                    out = res.get("result") or {"isError": True, "content": [
                        {"type": "text", "text": json.dumps(res.get("error"))}]}
                    text = "".join(c.get("text", "") for c in out.get("content", []))
                    calls.append({"name": fn["name"], "arguments": args,
                                  "is_error": bool(out.get("isError"))})
                    messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": text})
                if len(calls) >= max_calls:
                    break
        finally:
            mcp.close()
        ok, why = score(task.get("expect") or {}, calls, answer, Path(sandbox))
        return {"ok": ok, "reason": why, "tool_calls": calls, "answer": answer, "transcript": messages}


def _call_ok(calls: List[dict], name: str, subset: dict, allow_error: bool) -> bool:
    return any(
        c["name"] == name and (allow_error or not c["is_error"])
        and all(c["arguments"].get(k) == v for k, v in subset.items())
        for c in calls
    )


def score(expect: dict, calls: List[dict], answer: str, sandbox: Path) -> Tuple[bool, str]:
    wanted = list(expect.get("tool_calls") or [])
    if expect.get("tool_called"):
        wanted.append({"name": expect["tool_called"], "args_subset": expect.get("args_subset") or {}})
    for w in wanted:
        sub = w.get("args_subset") or {}
        if not _call_ok(calls, w["name"], sub, bool(expect.get("allow_error"))):
            return False, f"no matching call to {w['name']} with args {sub}"
    for s in expect.get("answer_contains") or []:
        alts = s if isinstance(s, list) else [s]  # a list means: any one of these
        if not any(x in answer for x in alts):
            return False, f"answer missing {alts!r}"
    for rel in expect.get("files_exist") or []:
        if not (sandbox / rel).exists():
            return False, f"expected file {rel} to exist"
    for rel in expect.get("files_absent") or []:
        if (sandbox / rel).exists():
            return False, f"expected file {rel} to be absent"
    for rel, content in (expect.get("files_unchanged") or {}).items():
        fp = sandbox / rel
        if not fp.exists() or fp.read_text(encoding="utf-8") != content:
            return False, f"file {rel} changed"
    return True, "ok"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tool-cmd", required=True, help="CLI prefix, e.g. 'tokenledger'")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="OpenAI-compatible base URL")
    ap.add_argument("--model", action="append", help="repeatable; first is PRIMARY (default: both fixed models)")
    ap.add_argument("--runs", type=int, default=MIN_RUNS)
    ap.add_argument("--bar", type=float, default=PASS_BAR)
    ap.add_argument("--tasks", default=str(ROOT / "eval" / "agent_tasks"))
    ap.add_argument("--docs", nargs="*", default=[str(ROOT / "llms.txt"), str(ROOT / "AGENTS.md")])
    ap.add_argument("--out", default=str(ROOT / "eval" / "results.json"))
    ap.add_argument("--scores", default=str(ROOT / "eval" / "scores.json"))
    ap.add_argument("--max-tool-calls", type=int, default=MAX_TOOL_CALLS)
    ap.add_argument("--timeout", type=float, default=600.0)
    a = ap.parse_args(argv)
    if a.runs < MIN_RUNS:
        ap.error(f"--runs must be >= {MIN_RUNS}")
    if a.bar < PASS_BAR:
        ap.error(f"--bar must be >= {PASS_BAR} (fixed before running)")
    models = a.model or DEFAULT_MODELS

    base = shlex.split(a.tool_cmd, posix=os.name != "nt")
    system = "\n\n".join(Path(d).read_text(encoding="utf-8") for d in a.docs)
    desc = subprocess.run(base + ["describe", "--json"], capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, check=True).stdout
    tools = json.loads(desc)["openai"]
    tasks = load_tasks(Path(a.tasks))
    per_model = []
    for model in models:
        results = []
        passed = total = 0
        for t in tasks:
            runs = [run_once(t, base, system, tools, a.endpoint, model, a.max_tool_calls, a.timeout)
                    for _ in range(a.runs)]
            passed += sum(r["ok"] for r in runs)
            total += len(runs)
            results.append({"id": t["id"], "prompt": t["prompt"], "runs": runs})
            print(f"{model} {t['id']}: {sum(r['ok'] for r in runs)}/{len(runs)}", file=sys.stderr)
        rate = passed / total if total else 0.0
        per_model.append({"model": model, "pass_rate": rate, "passed": passed, "attempts": total,
                          "tasks": results})
    primary = per_model[0]
    verdict = "PASS" if primary["pass_rate"] >= a.bar else "FAIL"
    meta = {"endpoint": a.endpoint, "primary_model": models[0], "pass_bar": a.bar,
            "runs_per_task": a.runs, "temperature": TEMPERATURE, "max_tool_calls": a.max_tool_calls,
            "system_prompt": system, "tool_definitions": tools, "verdict": verdict}
    Path(a.out).write_text(json.dumps({**meta, "models": per_model}, indent=1, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    scores = {k: v for k, v in meta.items() if k not in ("system_prompt", "tool_definitions")}
    scores["models"] = [{"model": m["model"], "pass_rate": m["pass_rate"], "passed": m["passed"],
                         "attempts": m["attempts"],
                         "per_task": {t["id"]: [r["ok"] for r in t["runs"]] for t in m["tasks"]}}
                        for m in per_model]
    Path(a.scores).write_text(json.dumps(scores, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "primary_pass_rate": primary["pass_rate"], "scores": a.scores}))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
