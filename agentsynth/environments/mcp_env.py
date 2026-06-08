"""An environment backed by a Model Context Protocol (MCP) server.

Point it at any MCP server (a local stdio command, or an HTTP/SSE URL), and its
tools become available to the generator and run for real. The MCP SDK is async
and optional; we bridge it to the sync Environment interface with a background
event loop, and only import `mcp` when an MCPEnvironment is actually constructed.

    env = MCPEnvironment(command="python", args=["examples/mcp_server.py"])
    gen = AgentTrajectoryGenerator(environment=env)

Requires `pip install "agentsynth[mcp]"` (Python 3.10+).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, List, Optional

from ..schemas import ToolSpec
from .base import Environment


def _tool_to_spec(tool: Any) -> ToolSpec:
    schema = getattr(tool, "inputSchema", None) or {
        "type": "object",
        "properties": {},
        "required": [],
    }
    return ToolSpec(
        name=tool.name,
        description=getattr(tool, "description", "") or "",
        parameters=schema,
    )


def _extract_text(call_result: Any) -> str:
    parts: List[str] = []
    for block in getattr(call_result, "content", None) or []:
        text = getattr(block, "text", None)
        parts.append(text if text is not None else str(getattr(block, "data", block)))
    out = "\n".join(parts).strip()
    if getattr(call_result, "isError", False):
        return f"MCPError: {out}" if out else "MCPError: the tool returned an error"
    return out


def _synth_value(name: str, schema: Dict[str, Any], query: str, seed: int) -> Any:
    schema = schema if isinstance(schema, dict) else {}
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]
    jtype = schema.get("type")
    if jtype in ("integer", "number"):
        value = (seed % 7) + 2
        return int(value) if jtype == "integer" else float(value)
    if jtype == "boolean":
        return True
    if jtype == "array":
        return []
    if jtype == "object":
        return {}
    words = " ".join((query or "").split()[:8])
    return words or name


def _synth_args(spec: ToolSpec, query: str, seed: int) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    for arg in spec.required_args():
        args[arg] = _synth_value(arg, spec.arg_schema(arg), query, seed)
    return args


class MCPEnvironment(Environment):
    name = "mcp"

    def __init__(
        self,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        url: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        transport: Optional[str] = None,
        timeout: float = 30.0,
        connect: bool = True,
    ) -> None:
        self._command = command
        self._args = list(args or [])
        self._url = url
        self._env = env
        self._cwd = cwd
        self._transport = transport or ("stdio" if command else "sse")
        self.timeout = timeout
        self._loop: Any = None
        self._thread: Optional[threading.Thread] = None
        self._session: Any = None
        self._stop: Any = None
        self._tools_cache: Optional[List[ToolSpec]] = None
        if connect:
            self.connect()

    # -- connection lifecycle ------------------------------------------------

    def connect(self) -> None:
        if self._thread is not None:
            return
        self._loop = asyncio.new_event_loop()
        ready = threading.Event()
        error: Dict[str, BaseException] = {}

        def run() -> None:
            asyncio.set_event_loop(self._loop)
            self._stop = asyncio.Event()
            try:
                self._loop.run_until_complete(self._serve(ready))
            except BaseException as exc:  # surfaced to connect() below
                error["e"] = exc
                ready.set()
            finally:
                self._loop.close()

        self._thread = threading.Thread(target=run, name="mcp-env", daemon=True)
        self._thread.start()
        if not ready.wait(self.timeout):
            self._thread = None
            raise TimeoutError("MCP server did not become ready in time")
        if "e" in error:
            self._thread = None
            raise RuntimeError(f"failed to connect to MCP server: {error['e']}")

    async def _serve(self, ready: threading.Event) -> None:
        from mcp import ClientSession

        async with self._make_transport() as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                ready.set()
                await self._stop.wait()

    def _make_transport(self) -> Any:
        if self._transport == "stdio":
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=self._command or "", args=self._args, env=self._env, cwd=self._cwd
            )
            return stdio_client(params)
        if self._transport == "sse":
            from mcp.client.sse import sse_client

            return sse_client(self._url)
        if self._transport in ("http", "streamable-http", "streamable_http"):
            from mcp.client.streamable_http import streamablehttp_client

            return streamablehttp_client(self._url)
        raise ValueError(f"unknown MCP transport {self._transport!r}")

    def _run(self, coro: Any) -> Any:
        if self._session is None or self._loop is None:
            raise RuntimeError("MCP environment is not connected")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(self.timeout)

    # -- Environment API -----------------------------------------------------

    def tools(self) -> List[ToolSpec]:
        if self._tools_cache is None:
            result = self._run(self._session.list_tools())
            self._tools_cache = [_tool_to_spec(t) for t in result.tools]
        return self._tools_cache

    def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        result = self._run(self._session.call_tool(tool_name, args or {}))
        return _extract_text(result)

    def sample_args(self, tool_name: str, query: str, seed: int) -> Dict[str, Any]:
        for spec in self.tools():
            if spec.name == tool_name:
                return _synth_args(spec, query, seed)
        return {}

    def reset(self) -> None:
        self._tools_cache = None

    def close(self) -> None:
        if self._stop is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=self.timeout)
        self._thread = None
        self._session = None
        self._tools_cache = None


__all__ = ["MCPEnvironment"]
