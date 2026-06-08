"""Offline smoke tests: full public surface in mock mode, no network or keys.

Mock is forced both via AGENTSYNTH_FORCE_MOCK (set before the import below) and
via use_mock=True at each call site. Assertions check contracts, not the exact
mock wording, so they survive tweaks to the deterministic text.
"""

from __future__ import annotations

import json
import os

# Set before importing agentsynth so nothing reaches for a provider key that
# happens to be in the env.
os.environ["AGENTSYNTH_FORCE_MOCK"] = "1"

import agentsynth
from agentsynth import (
    RUBRIC_DIMENSIONS,
    AgentTrajectoryGenerator,
    EvalResult,
    Trajectory,
    TrajectoryEvaluator,
    compute_dataset_metrics,
    load_jsonl,
    to_adp,
    to_jsonl,
    to_sharegpt,
)

# Spans two tool domains (weather + numbers) so the mock has something to ground
# tool calls and code on.
QUERY = "What is the weather in Paris and what is 12 + 30?"


def test_generate_single_agent():
    """single_agent: at least one tool call, a final answer, catalog attached."""
    gen = AgentTrajectoryGenerator(use_mock=True)
    traj = gen.generate(QUERY, mode="single_agent")

    assert isinstance(traj, Trajectory)
    assert traj.mode == "single_agent"
    assert traj.query == QUERY

    tool_calls = traj.tool_calls()
    assert len(tool_calls) >= 1
    for step in tool_calls:
        assert step.tool_name
        assert isinstance(step.tool_args, dict)

    assert len(traj.tools) >= 1
    tool_names = {t.name for t in traj.tools}
    # Every called tool has to exist in the attached catalog.
    for name in traj.tool_names_used():
        assert name in tool_names

    assert traj.final_answer.strip()
    assert any(s.step_type == "final_answer" for s in traj.steps)


def test_generate_code_execution():
    """code_execution: a code step with both code and real (non-error) output."""
    gen = AgentTrajectoryGenerator(use_mock=True)
    traj = gen.generate(QUERY, mode="code_execution")

    assert traj.mode == "code_execution"

    code_steps = [s for s in traj.steps if s.step_type == "code_execution"]
    assert len(code_steps) >= 1

    step = code_steps[0]
    assert step.code and step.code.strip(), "code_execution step must carry code"
    assert step.code_output and step.code_output.strip(), (
        "code_execution step must carry non-empty (grounded) output"
    )
    # Output should be genuine REPL output, not an error or blocked message.
    assert "REPLError" not in step.code_output
    assert "Traceback" not in step.code_output

    assert traj.final_answer.strip()


def test_generate_multi_agent():
    """multi_agent: steps carry agent tags, including a planner and a critic."""
    gen = AgentTrajectoryGenerator(use_mock=True)
    traj = gen.generate(QUERY, mode="multi_agent")

    assert traj.mode == "multi_agent"

    agents = {s.agent for s in traj.steps if s.agent}
    assert agents, "multi_agent steps must carry an 'agent' tag"
    assert "planner" in agents
    assert "critic" in agents

    # Planner drives the plan; critic produces the critique.
    assert any(s.step_type == "plan" and s.agent == "planner" for s in traj.steps)
    assert any(s.step_type == "critique" and s.agent == "critic" for s in traj.steps)

    assert traj.final_answer.strip()


def test_generate_batch():
    """vary_modes batch returns the requested count and more than one mode."""
    gen = AgentTrajectoryGenerator(use_mock=True)
    trajs = gen.generate_batch(QUERY, num_trajectories=5, vary_modes=True)

    assert isinstance(trajs, list)
    assert len(trajs) == 5
    assert all(isinstance(t, Trajectory) for t in trajs)

    modes = {t.mode for t in trajs}
    assert len(modes) > 1
    assert modes.issubset(set(agentsynth.schemas.TRAJECTORY_MODES))

    for t in trajs:
        assert t.num_steps() >= 1
        assert t.final_answer.strip()


def test_evaluate():
    """Every rubric dimension scores as a float in [0, 1]; verdict is a bool."""
    gen = AgentTrajectoryGenerator(use_mock=True)
    traj = gen.generate(QUERY, mode="single_agent")

    ev = TrajectoryEvaluator(use_mock=True).evaluate(traj)

    assert isinstance(ev, EvalResult)
    assert ev.trajectory_id == traj.id
    assert ev.judge_model == "mock"

    scores = ev.scores.as_dict()
    assert set(scores.keys()) == set(RUBRIC_DIMENSIONS)
    for dim in RUBRIC_DIMENSIONS:
        val = scores[dim]
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0, f"{dim} out of range: {val}"

    assert isinstance(ev.overall, float)
    assert 0.0 <= ev.overall <= 1.0
    assert isinstance(ev.passed, bool)


def test_evaluate_batch_and_metrics():
    """evaluate_batch + compute_dataset_metrics expose the documented keys."""
    gen = AgentTrajectoryGenerator(use_mock=True)
    trajs = gen.generate_batch(QUERY, num_trajectories=5, vary_modes=True)

    evaluator = TrajectoryEvaluator(use_mock=True)
    results = evaluator.evaluate_batch(trajs)

    assert len(results) == len(trajs)
    assert all(isinstance(r, EvalResult) for r in results)

    metrics = compute_dataset_metrics(trajs, results)

    for key in ("num_trajectories", "pass_rate", "diversity_score", "tool_usage"):
        assert key in metrics, f"missing metrics key: {key}"

    assert metrics["num_trajectories"] == len(trajs)

    pass_rate = metrics["pass_rate"]
    assert pass_rate is not None
    assert 0.0 <= pass_rate <= 1.0

    # diversity_score in [0, 1]; tool_usage maps name -> count.
    assert 0.0 <= metrics["diversity_score"] <= 1.0
    assert isinstance(metrics["tool_usage"], dict)
    for name, count in metrics["tool_usage"].items():
        assert isinstance(name, str)
        assert isinstance(count, int) and count >= 1


def test_jsonl_roundtrip(tmp_path):
    """JSONL round trip preserves ids, queries, answers, modes, step counts."""
    gen = AgentTrajectoryGenerator(use_mock=True)
    trajs = gen.generate_batch(QUERY, num_trajectories=4, vary_modes=True)

    path = tmp_path / "dataset.jsonl"
    text = to_jsonl(trajs, str(path))

    # to_jsonl both writes the file and returns the JSONL string.
    assert path.exists()
    assert text.strip()
    nonempty_lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(nonempty_lines) == len(trajs)
    for ln in nonempty_lines:
        json.loads(ln)

    loaded = load_jsonl(str(path))

    assert len(loaded) == len(trajs)

    assert [t.id for t in loaded] == [t.id for t in trajs]
    assert [t.query for t in loaded] == [t.query for t in trajs]
    assert [t.final_answer for t in loaded] == [t.final_answer for t in trajs]
    assert [t.num_steps() for t in loaded] == [t.num_steps() for t in trajs]
    assert [t.mode for t in loaded] == [t.mode for t in trajs]


def test_sharegpt_and_adp(tmp_path):
    """ShareGPT and ADP exports are non-empty and carry the expected keys."""
    gen = AgentTrajectoryGenerator(use_mock=True)
    trajs = gen.generate_batch(QUERY, num_trajectories=3, vary_modes=True)

    sharegpt = to_sharegpt(trajs, str(tmp_path / "sharegpt.json"))
    assert isinstance(sharegpt, list)
    assert len(sharegpt) == len(trajs)
    for record in sharegpt:
        assert "conversations" in record
        assert "tools" in record
        convs = record["conversations"]
        assert isinstance(convs, list) and convs
        for turn in convs:
            assert "from" in turn and "value" in turn
        # First turn is always the human query.
        assert convs[0]["from"] == "human"

    adp = to_adp(trajs, str(tmp_path / "adp.json"))
    assert isinstance(adp, list)
    assert len(adp) == len(trajs)
    for record in adp:
        for key in ("schema", "trajectory_id", "instruction", "tools", "steps", "output"):
            assert key in record, f"ADP record missing key: {key}"
        assert isinstance(record["steps"], list) and record["steps"]
        for step in record["steps"]:
            assert "type" in step and "content" in step

    assert [r["trajectory_id"] for r in adp] == [t.id for t in trajs]


def test_repl_grounding():
    """REPL runs whitelisted code and blocks dangerous ops via error strings."""
    repl = agentsynth.utils.PythonREPL()

    assert repl.run("print(2 + 2)") == "4"
    # Bare expression echoes its value, like an interactive REPL.
    assert repl.run("3 * 7") == "21"

    # Blocked op comes back as an error string rather than raising.
    blocked = repl.run("open('/etc/passwd')")
    assert isinstance(blocked, str)
    assert blocked
    lowered = blocked.lower()
    assert ("blocked" in lowered) or ("error" in lowered)

    # Same for disallowed imports, and the call must not have run.
    bad_import = repl.run("import os\nprint(os.getcwd())")
    assert isinstance(bad_import, str)
    assert "os.getcwd()" not in bad_import


def test_determinism():
    """Same seed -> identical tool signatures, step renders and final answers."""
    gen_a = AgentTrajectoryGenerator(use_mock=True, seed=123)
    gen_b = AgentTrajectoryGenerator(use_mock=True, seed=123)

    for index in range(3):
        traj_a = gen_a.generate(QUERY, mode="single_agent", index=index)
        traj_b = gen_b.generate(QUERY, mode="single_agent", index=index)

        assert traj_a.tool_signature() == traj_b.tool_signature()
        assert traj_a.tool_signature()
        assert [s.short() for s in traj_a.steps] == [s.short() for s in traj_b.steps]
        assert traj_a.final_answer == traj_b.final_answer

    # A different seed is still self-consistent.
    gen_c = AgentTrajectoryGenerator(use_mock=True, seed=999)
    sig_c1 = gen_c.generate(QUERY, mode="single_agent", index=0).tool_signature()
    sig_c2 = (
        AgentTrajectoryGenerator(use_mock=True, seed=999)
        .generate(QUERY, mode="single_agent", index=0)
        .tool_signature()
    )
    assert sig_c1 == sig_c2
