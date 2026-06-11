"""Trace-importer tests: OpenAI and Anthropic logs become verifiable Trajectories."""

import json

import pytest

from agentsynth import (
    AgentTrajectoryGenerator,
    TrajectoryEvaluator,
    import_traces,
    load_traces_jsonl,
    to_jsonl,
    trajectory_from_messages,
    verify_trajectory,
)
from agentsynth.importers import trajectory_from_anthropic

OPENAI_TRACE = [
    {"role": "user", "content": "What's the weather in Hanoi?"},
    {
        "role": "assistant",
        "content": "I'll check the weather.",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Hanoi"}'},
            }
        ],
    },
    {"role": "tool", "content": "Hanoi: 31C, humid"},
    {"role": "assistant", "content": "It's 31C and humid in Hanoi."},
]

ANTHROPIC_TRACE = [
    {"role": "user", "content": "What's the weather in Hanoi?"},
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me look that up."},
            {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Hanoi"}},
        ],
    },
    {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "Hanoi: 31C"}],
    },
    {"role": "assistant", "content": [{"type": "text", "text": "31C in Hanoi right now."}]},
]


# --- OpenAI format -------------------------------------------------------------


def test_openai_trace_maps_to_trajectory():
    traj = trajectory_from_messages(OPENAI_TRACE)
    assert traj.query == "What's the weather in Hanoi?"
    assert traj.tool_names_used() == ["get_weather"]
    call = traj.tool_calls()[0]
    assert call.tool_args == {"city": "Hanoi"}  # JSON-string arguments parsed
    assert [s.step_type for s in traj.steps] == [
        "tool_call",
        "thought",
        "observation",
        "final_answer",
    ]
    assert traj.final_answer == "It's 31C and humid in Hanoi."
    assert traj.metadata["source"] == "trace:openai"
    assert traj.tools and traj.tools[0].name == "get_weather"  # inferred spec


def test_openai_import_is_deterministic():
    assert trajectory_from_messages(OPENAI_TRACE).id == trajectory_from_messages(OPENAI_TRACE).id


def test_malformed_arguments_fall_back_to_empty():
    trace = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "t", "arguments": "{not json"}}],
        },
        {"role": "assistant", "content": "done"},
    ]
    traj = trajectory_from_messages(trace)
    assert traj.tool_calls()[0].tool_args == {}


# --- Anthropic format ------------------------------------------------------------


def test_anthropic_trace_maps_to_trajectory():
    traj = trajectory_from_anthropic(ANTHROPIC_TRACE)
    assert traj.query == "What's the weather in Hanoi?"
    assert traj.tool_names_used() == ["get_weather"]
    assert traj.tool_calls()[0].tool_args == {"city": "Hanoi"}
    assert any(s.step_type == "observation" and "31C" in (s.observation or "") for s in traj.steps)
    assert traj.final_answer == "31C in Hanoi right now."
    assert traj.metadata["source"] == "trace:anthropic"


# --- batch + auto-detect ----------------------------------------------------------


def test_import_traces_autodetects_formats():
    trajs = import_traces([OPENAI_TRACE, ANTHROPIC_TRACE, {"messages": OPENAI_TRACE}])
    assert len(trajs) == 3
    assert trajs[0].metadata["source"] == "trace:openai"
    assert trajs[1].metadata["source"] == "trace:anthropic"


def test_load_traces_jsonl(tmp_path):
    path = tmp_path / "traces.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"messages": OPENAI_TRACE}) + "\n")
        fh.write(json.dumps(ANTHROPIC_TRACE) + "\n")
        fh.write("\n")  # blank lines tolerated
    trajs = load_traces_jsonl(str(path))
    assert len(trajs) == 2


# --- the point: the whole stack applies to imported traces -------------------------


def test_imported_traces_flow_through_judge_verify_export(tmp_path):
    traj = trajectory_from_messages(OPENAI_TRACE)
    result = TrajectoryEvaluator(use_mock=True).evaluate(traj)
    assert 0.0 <= result.overall <= 1.0
    verification = verify_trajectory(traj)
    assert verification.checks  # ran with the inferred tool specs
    path = str(tmp_path / "imported.jsonl")
    to_jsonl([traj], path)
    assert json.loads(open(path, encoding="utf-8").readline())["query"] == traj.query


def test_round_trip_from_generated_trajectory():
    # to_messages -> import is (loosely) inverse: query, tools used, and answer survive
    source = AgentTrajectoryGenerator(use_mock=True).generate(
        "weather in Tokyo and an 18% tip on $54"
    )
    back = trajectory_from_messages(source.to_messages(), tools=source.tools)
    assert back.query == source.query
    assert back.tool_names_used() == source.tool_names_used()
    assert back.final_answer == source.final_answer


def test_explicit_format_and_empty_records():
    assert import_traces([], format="openai") == []
    assert import_traces([{"messages": []}, {"nope": 1}]) == []
    with pytest.raises(FileNotFoundError):
        load_traces_jsonl("/nonexistent/file.jsonl")
