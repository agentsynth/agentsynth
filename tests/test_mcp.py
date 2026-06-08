"""MCP environment tests.

The adapter helpers are tested with fakes so they run everywhere (including the
3.9 dev interpreter where the `mcp` SDK can't be installed). The live tests spin
up the example MCP server over stdio and are skipped unless `mcp` is importable.
"""

import os
import sys

import pytest

from agentsynth.environments.mcp_env import (
    _extract_text,
    _synth_args,
    _synth_value,
    _tool_to_spec,
)
from agentsynth.schemas import ToolSpec

SERVER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "mcp_server.py")


# --- adapter helpers (no mcp needed) ---------------------------------------


class _FakeTool:
    def __init__(self, name, description, input_schema):
        self.name = name
        self.description = description
        self.inputSchema = input_schema


class _FakeText:
    def __init__(self, text):
        self.text = text


class _FakeResult:
    def __init__(self, content, is_error=False):
        self.content = content
        self.isError = is_error


def test_tool_to_spec():
    tool = _FakeTool(
        "add",
        "Add two integers",
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    )
    spec = _tool_to_spec(tool)
    assert spec.name == "add"
    assert set(spec.required_args()) == {"a", "b"}


def test_extract_text_joins_blocks():
    assert _extract_text(_FakeResult([_FakeText("5")])) == "5"
    assert _extract_text(_FakeResult([_FakeText("a"), _FakeText("b")])) == "a\nb"


def test_extract_text_flags_errors():
    assert _extract_text(_FakeResult([_FakeText("boom")], is_error=True)).startswith("MCPError")


def test_synth_value_handles_types():
    assert isinstance(_synth_value("a", {"type": "integer"}, "q", 1), int)
    assert _synth_value("c", {"type": "boolean"}, "q", 1) is True
    assert _synth_value("e", {"enum": ["red", "green"]}, "q", 1) == "red"
    assert isinstance(_synth_value("s", {"type": "string"}, "hello there", 1), str)


def test_synth_args_fills_required():
    spec = ToolSpec(
        name="t",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "txt": {"type": "string"}},
            "required": ["a", "txt"],
        },
    )
    args = _synth_args(spec, "some query words", seed=1)
    assert set(args) == {"a", "txt"}
    assert isinstance(args["a"], int) and isinstance(args["txt"], str)


# --- live server (skipped unless mcp is installed) -------------------------


@pytest.fixture
def mcp_env():
    pytest.importorskip("mcp")
    from agentsynth.environments import MCPEnvironment

    env = MCPEnvironment(command=sys.executable, args=[SERVER])
    yield env
    env.close()


def test_mcp_lists_server_tools(mcp_env):
    assert {"add", "word_count", "reverse", "uppercase"} <= set(mcp_env.tool_names())


def test_mcp_executes_tools(mcp_env):
    assert "5" in mcp_env.execute("add", {"a": 2, "b": 3})
    assert "cba" in mcp_env.execute("reverse", {"text": "abc"})


def test_mcp_sample_args_then_execute(mcp_env):
    args = mcp_env.sample_args("word_count", "the quick brown fox jumps over", seed=1)
    out = mcp_env.execute("word_count", args)
    assert any(ch.isdigit() for ch in out)


def test_generator_uses_mcp_tools(mcp_env):
    from agentsynth import AgentTrajectoryGenerator

    gen = AgentTrajectoryGenerator(use_mock=True, environment=mcp_env)
    traj = gen.generate("reverse this text and count the words", mode="single_agent")
    assert traj.tool_names_used()
    assert all(n in {"add", "word_count", "reverse", "uppercase"} for n in traj.tool_names_used())
