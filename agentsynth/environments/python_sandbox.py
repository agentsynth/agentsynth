"""A Python execution environment.

The `python` tool runs a snippet in a fresh, isolated subprocess (`python -I`)
with a timeout and returns its real stdout.

WARNING: subprocess isolation plus a timeout is not a security sandbox. The code
can still touch the filesystem and network. Don't run untrusted code with it on a
host you care about; for that, put a real sandbox (gVisor, Firecracker, e2b, …)
behind this same interface.
"""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Any, Dict, List

from ..schemas import ToolSpec
from .base import Environment

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _snippet(query: str, seed: int) -> str:
    """A small, valid stdlib-only snippet, tied to the query when it has numbers."""
    nums = [float(n) for n in _NUM_RE.findall(query or "")]
    if len(nums) >= 2:
        as_ints = all(n.is_integer() for n in nums)
        values = [int(n) for n in nums] if as_ints else nums
        return (
            "import statistics\n"
            f"values = {values}\n"
            'print(f"n={len(values)} total={sum(values)} '
            'mean={statistics.mean(values):.2f}")'
        )
    n = (seed % 8) + 5
    return (
        "import math\n"
        f"n = {n}\n"
        "squares = sum(i * i for i in range(1, n + 1))\n"
        'print(f"sum_of_squares={squares} sqrt={math.sqrt(squares):.3f}")'
    )


class PythonSandbox(Environment):
    name = "python"

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout

    def tools(self) -> List[ToolSpec]:
        return [
            ToolSpec(
                name="python",
                description="Execute a short Python snippet and return its stdout.",
                parameters={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python source to run"}
                    },
                    "required": ["code"],
                },
            )
        ]

    def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        if tool_name != "python":
            raise KeyError(tool_name)
        code = str((args or {}).get("code", ""))
        if not code.strip():
            return ""
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return f"TimeoutError: code exceeded {self.timeout:g}s"
        out = (proc.stdout or "").strip()
        if proc.returncode != 0:
            err_lines = (proc.stderr or "").strip().splitlines()
            tail = err_lines[-1] if err_lines else "non-zero exit"
            return (out + ("\n" if out else "") + tail).strip()
        return out

    def sample_args(self, tool_name: str, query: str, seed: int) -> Dict[str, Any]:
        return {"code": _snippet(query, seed)}


__all__ = ["PythonSandbox"]
