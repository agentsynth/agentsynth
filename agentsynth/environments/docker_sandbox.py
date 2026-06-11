"""Run Python in a throwaway Docker container.

Same tool surface as PythonSandbox, but each snippet runs in its own container
with networking off — the right backend when the generated code can't be trusted
on the host. Needs a working `docker` on PATH; `DockerSandbox.available()` tells
you up front.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any, Dict, List

from ..schemas import ToolSpec
from .base import Environment
from .python_sandbox import _snippet


class DockerSandbox(Environment):
    name = "docker"

    def __init__(self, image: str = "python:3.12-alpine", timeout: float = 30.0) -> None:
        self.image = image
        self.timeout = timeout

    @staticmethod
    def available() -> bool:
        """True when the docker binary exists and the daemon answers."""
        if shutil.which("docker") is None:
            return False
        try:
            probe = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
            return probe.returncode == 0
        except Exception:
            return False

    def tools(self) -> List[ToolSpec]:
        return [
            ToolSpec(
                name="python",
                description="Execute a short Python snippet in an isolated container.",
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
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            self.image,
            "python",
            "-I",
            "-c",
            code,
        ]
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout)
        except FileNotFoundError:
            return "DockerError: docker not found on PATH"
        except subprocess.TimeoutExpired:
            return f"DockerError: code exceeded {self.timeout:g}s"
        out = (proc.stdout or "").strip()
        if proc.returncode != 0:
            err_lines = (proc.stderr or "").strip().splitlines()
            tail = err_lines[-1] if err_lines else "non-zero exit"
            return (out + ("\n" if out else "") + tail).strip()
        return out

    def sample_args(self, tool_name: str, query: str, seed: int) -> Dict[str, Any]:
        return {"code": _snippet(query, seed)}


__all__ = ["DockerSandbox"]
