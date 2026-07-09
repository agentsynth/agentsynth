"""Serve a scenario pack over MCP, so any MCP agent can be benched as-is.

    agentsynth serve-mcp --pack core_v1

speaks MCP over stdio — wire it into Claude Code, Claude Desktop, or any other
MCP client, and the agent works the pack's worlds through three kinds of tools:

* ``current_task`` — the scenario in play: task text, id, step budget, progress.
* the environment's own tools (``sql_query``, ...) — they execute for real
  against the live world.
* ``submit_answer`` — ends the episode. The checkers run against the world's
  end state, the result is recorded, and the next scenario loads. After the
  last one, the reply carries the pack report (pass rate per scenario).

One session is one bench run. The tool list is built from the first scenario,
so packs should keep a stable tool surface across scenarios (ours do).

Needs ``pip install "agentsynth-ai[mcp]"`` and Python 3.10+; the MCP SDK is
only imported when a server is actually built.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from .rl import AgentGym


class PackSession:
    """Walks a pack one scenario at a time, scoring each episode on submit."""

    def __init__(self, scenarios: Sequence[Any], seed: int = 7) -> None:
        if not scenarios:
            raise ValueError("empty pack")
        self.scenarios = list(scenarios)
        self.seed = seed
        self.results: List[Dict[str, Any]] = []
        self._index = 0
        self._gym: Optional[AgentGym] = None

    # -- state -------------------------------------------------------------------

    @property
    def done(self) -> bool:
        return self._index >= len(self.scenarios)

    def _current(self) -> AgentGym:
        if self.done:
            raise RuntimeError("pack finished — read the report in the last reply")
        if self._gym is None:
            self._gym = AgentGym.from_scenario(self.scenarios[self._index], seed=self.seed)
            self._gym.reset()
        return self._gym

    def task_info(self) -> Dict[str, Any]:
        if self.done:
            return {"done": True, "report": self.report()}
        gym = self._current()
        return {
            "scenario_id": self.scenarios[self._index].id,
            "task": gym.task,
            "steps_used": gym.step_count,
            "max_steps": gym.max_steps,
            "scenario": f"{self._index + 1}/{len(self.scenarios)}",
        }

    def report(self) -> Dict[str, Any]:
        passed = sum(1 for r in self.results if r["passed"])
        n = len(self.results) or 1
        return {
            "n": len(self.results),
            "passed": passed,
            "pass_rate": round(passed / n, 4),
            "results": self.results,
        }

    # -- episode flow --------------------------------------------------------------

    def _record_and_advance(self, outcome: Any) -> Dict[str, Any]:
        info = outcome.info.get("outcome", {})
        score = float(info.get("score", 0.0))
        entry = {
            "id": self.scenarios[self._index].id,
            "passed": score >= 1.0,
            "outcome_score": score,
            "checks": info.get("checks", []),
        }
        self.results.append(entry)
        if self._gym is not None:
            self._gym.close()
            self._gym = None
        self._index += 1
        reply = dict(entry)
        if self.done:
            reply["pack_report"] = self.report()
        else:
            reply["next"] = self.task_info()
        return reply

    def call_env_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        gym = self._current()
        outcome = gym.step({"tool": name, "args": arguments})
        if outcome.done:  # the step budget ran out and the episode self-scored
            reply = self._record_and_advance(outcome)
            reply["note"] = "step budget exhausted; the scenario was scored as-is"
            return reply
        return {"observation": outcome.observation, "steps_used": gym.step_count}

    def submit_answer(self, answer: str) -> Dict[str, Any]:
        gym = self._current()
        outcome = gym.step({"answer": answer})
        return self._record_and_advance(outcome)


def build_server(scenarios: Sequence[Any], seed: int = 7) -> Any:
    """A low-level MCP Server wrapping one PackSession."""
    import mcp.types as types
    from mcp.server import Server

    session = PackSession(scenarios, seed=seed)
    env = scenarios[0].build_environment()
    try:
        env_tools = list(env.tools())
    finally:
        env.close()
    env_tool_names = {t.name for t in env_tools}

    server: Any = Server("agentsynth")
    server._agentsynth_session = session  # reachable from tests and the CLI

    @server.list_tools()
    async def _list_tools() -> List[types.Tool]:
        tools = [
            types.Tool(
                name="current_task",
                description="The scenario being worked: task text, id, step budget.",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="submit_answer",
                description="Finish the scenario: the checkers run against the "
                "world's end state and the next scenario loads.",
                inputSchema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            ),
        ]
        for spec in env_tools:
            tools.append(
                types.Tool(
                    name=spec.name,
                    description=spec.description,
                    inputSchema=spec.parameters or {"type": "object", "properties": {}},
                )
            )
        return tools

    @server.call_tool()
    async def _call_tool(name: str, arguments: Optional[Dict[str, Any]]) -> List[Any]:
        arguments = arguments or {}
        try:
            if name == "current_task":
                payload = session.task_info()
            elif name == "submit_answer":
                payload = session.submit_answer(str(arguments.get("answer", "")))
            elif name in env_tool_names:
                payload = session.call_env_tool(name, arguments)
            else:
                payload = {"error": f"unknown tool {name!r}"}
        except RuntimeError as exc:
            payload = {"error": str(exc)}
        return [types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    return server


def serve_stdio(scenarios: Sequence[Any], seed: int = 7) -> None:
    """Block serving the pack over stdio until the client disconnects."""
    import asyncio

    from mcp.server.stdio import stdio_server

    server = build_server(scenarios, seed=seed)

    async def _run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(_run())


__all__ = ["PackSession", "build_server", "serve_stdio"]
