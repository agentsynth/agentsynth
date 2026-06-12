"""Bridges for bring-your-own agent loops.

`to_openai_tools` turns a world's tools into OpenAI function-calling schemas,
and `action_from_openai_tool_call` turns the model's tool call back into a gym
action. Together they let any framework that speaks that convention — the
OpenAI SDK, LangGraph, CrewAI — drive `AgentGym.reset()` / `step()` directly
instead of rewriting its loop as a policy function:

    gym = AgentGym.from_scenario(scenario)
    task = gym.reset()
    tools = to_openai_tools(gym)
    ... your loop calls the model with `tools`, then ...
    result = gym.step(action_from_openai_tool_call(call))
    ... and ends with gym.step({"answer": text}); the final step's
    `info["outcome"]` carries the world-state verdict.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def _tool_specs(source: Any) -> List[Any]:
    """Tool specs from an AgentGym, an Environment, or a plain list of specs."""
    env = getattr(source, "environment", source)
    tools = env.tools() if callable(getattr(env, "tools", None)) else env
    return list(tools)


def to_openai_tools(source: Any) -> List[Dict[str, Any]]:
    """The world's tools as OpenAI function-calling schemas."""
    out = []
    for spec in _tool_specs(source):
        parameters = spec.parameters or {"type": "object", "properties": {}}
        out.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description or "",
                    "parameters": parameters,
                },
            }
        )
    return out


def action_from_openai_tool_call(tool_call: Any) -> Dict[str, Any]:
    """An OpenAI-style tool call (dict or SDK object) as a gym action.

    Malformed argument JSON becomes `{}` rather than an exception — the gym
    turns that into a recoverable error observation, the way a real run should.
    """
    if isinstance(tool_call, dict):
        fn = tool_call.get("function", tool_call)
        name = fn.get("name", "")
        raw = fn.get("arguments", {})
    else:
        fn = getattr(tool_call, "function", tool_call)
        name = getattr(fn, "name", "") or ""
        raw = getattr(fn, "arguments", {})

    if isinstance(raw, str):
        try:
            args = json.loads(raw or "{}")
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
    else:
        args = dict(raw or {})
    return {"tool_name": name, "arguments": args}


__all__ = ["to_openai_tools", "action_from_openai_tool_call"]
