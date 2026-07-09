"""Bench an agent that lives outside this process.

Two transports, one payload. Each step the harness sends one JSON object:

    {"task": "...", "observation": "...", "step": 1, "max_steps": 8,
     "tools": [{"name": ..., "description": ..., "parameters": {...}}, ...],
     "transcript": "call sql_query({...})\\n-> ..."}

and the agent replies with one JSON object — ``{"tool": ..., "args": {...}}``
to act, or ``{"answer": "..."}`` to finish. That is the whole integration
surface: any language, any framework, no SDK import.

    agentsynth bench --pack core_v2 --agent "python my_agent.py"
    agentsynth bench --pack core_v2 --agent http://localhost:8088/act

A subprocess agent reads one line from stdin and prints one line to stdout per
step; a long-lived loop keeps state across steps, and a script that exits after
one reply is respawned per step, so both styles work unchanged. An HTTP agent
receives the same payload as a POST body. stderr passes through to the console
either way, so agents can log freely.
"""

from __future__ import annotations

import json
import select
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Union

Policy = Callable[[str, Any], Any]


def step_payload(observation: str, gym: Any, max_transcript_chars: int = 4000) -> Dict[str, Any]:
    """The JSON object an external agent sees each step."""
    return {
        "task": gym.task,
        "observation": observation,
        "step": gym.step_count + 1,
        "max_steps": gym.max_steps,
        "tools": [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in gym.environment.tools()
        ],
        "transcript": gym.transcript(max_transcript_chars),
    }


def _parse_reply(raw: str) -> Any:
    """A reply should be JSON; anything else is taken as a final answer."""
    raw = raw.strip()
    if not raw:
        return {"answer": ""}
    try:
        return json.loads(raw)
    except ValueError:
        return raw  # the episode coercion treats unparsable text as the answer


class _SubprocessAgent:
    """One line of JSON out, one line back, over a child process's stdio."""

    def __init__(self, cmd: Union[str, List[str]], timeout: float = 120.0) -> None:
        self.argv = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
        if not self.argv:
            raise ValueError("empty agent command")
        self.timeout = timeout
        self._proc: Optional[subprocess.Popen] = None

    def _spawn(self) -> subprocess.Popen:
        return subprocess.Popen(
            self.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # the agent's logging stays visible
            text=True,
            bufsize=1,
        )

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _read_line(self, proc: subprocess.Popen) -> str:
        stdout = proc.stdout
        assert stdout is not None
        if hasattr(select, "select") and sys.platform != "win32":
            ready, _, _ = select.select([stdout], [], [], self.timeout)
            if not ready:
                proc.kill()
                raise RuntimeError(
                    f"agent {self.argv[0]!r} sent nothing for {self.timeout:.0f}s — "
                    "it must print one JSON line per request"
                )
        return stdout.readline()

    def __call__(self, observation: str, gym: Any) -> Any:
        payload = json.dumps(step_payload(observation, gym)) + "\n"
        # One retry: a one-shot agent exits after each reply, and the exit may not
        # be visible yet when the next step arrives — respawn and resend once.
        for attempt in (0, 1):
            if not self._alive():
                self.close()
                self._proc = self._spawn()
            proc = self._proc
            assert proc is not None and proc.stdin is not None
            try:
                proc.stdin.write(payload)
                proc.stdin.flush()
                line = self._read_line(proc)
            except (BrokenPipeError, OSError):
                line = ""
            if line:
                return _parse_reply(line)
            code = proc.wait()
            self.close()
            if attempt:
                raise RuntimeError(f"agent {self.argv[0]!r} exited (code {code}) without replying")
        raise AssertionError("unreachable")

    def close(self) -> None:
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            for stream in (self._proc.stdin, self._proc.stdout):
                if stream is not None:
                    stream.close()
            self._proc = None


class _HTTPAgent:
    """POST the step payload to an endpoint, read the action from the body."""

    def __init__(
        self, url: str, timeout: float = 120.0, headers: Optional[Dict[str, str]] = None
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        self.headers.update(headers or {})

    def __call__(self, observation: str, gym: Any) -> Any:
        body = json.dumps(step_payload(observation, gym)).encode("utf-8")
        request = urllib.request.Request(self.url, data=body, headers=self.headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"agent endpoint {self.url} unreachable: {exc}") from None
        return _parse_reply(raw)

    def close(self) -> None:  # symmetry with the subprocess agent; nothing to release
        return None


def subprocess_policy(cmd: Union[str, List[str]], timeout: float = 120.0) -> Policy:
    """A bench policy that forwards each step to ``cmd`` over stdin/stdout."""
    return _SubprocessAgent(cmd, timeout=timeout)


def http_policy(
    url: str, timeout: float = 120.0, headers: Optional[Dict[str, str]] = None
) -> Policy:
    """A bench policy that forwards each step to an HTTP endpoint."""
    return _HTTPAgent(url, timeout=timeout, headers=headers)


def agent_policy(spec: str, timeout: float = 120.0) -> Policy:
    """Resolve ``--agent`` — an http(s) URL or a shell command."""
    if spec.startswith(("http://", "https://")):
        return http_policy(spec, timeout=timeout)
    return subprocess_policy(spec, timeout=timeout)


__all__ = ["agent_policy", "http_policy", "step_payload", "subprocess_policy"]
