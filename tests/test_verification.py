import pytest

from agentsynth import AgentTrajectoryGenerator, TrajectoryEvaluator
from agentsynth.schemas import ToolSpec, Trajectory, TrajectoryStep
from agentsynth.verification import (
    EnsembleEvaluator,
    ExecutionVerifier,
    ExpectedAnswerVerifier,
    SafetyVerifier,
    ToolArgVerifier,
    batch_verify,
    get_rubric,
    rubric_names,
    verify_trajectory,
)
from agentsynth.verification.verifiers import default_verifiers

GEN = AgentTrajectoryGenerator(use_mock=True)


def test_execution_verifier_passes_on_grounded_code():
    traj = GEN.generate("compute mean of 4 8 15 16 23 42", mode="code_execution")
    assert ExecutionVerifier().check(traj).passed
    assert verify_trajectory(traj).verified


def test_execution_verifier_catches_tampered_output():
    traj = GEN.generate("compute mean of 3 6 9 12", mode="code_execution")
    for step in traj.steps:
        if step.step_type == "code_execution":
            step.code_output = "definitely the wrong output"
    assert not ExecutionVerifier().check(traj).passed
    assert not verify_trajectory(traj).verified  # required check failed


def test_execution_verifier_na_when_no_code():
    traj = GEN.generate("weather in Tokyo", mode="single_agent")
    assert ExecutionVerifier().check(traj).passed


def test_tool_arg_verifier_flags_unknown_tool():
    spec = ToolSpec(name="known", parameters={"type": "object", "properties": {}, "required": []})
    traj = Trajectory(query="x", tools=[spec])
    traj.steps.append(TrajectoryStep(step_type="tool_call", tool_name="ghost", tool_args={}))
    assert not ToolArgVerifier().check(traj).passed


def test_tool_arg_verifier_flags_missing_required():
    spec = ToolSpec(
        name="lookup",
        parameters={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
    )
    traj = Trajectory(query="x", tools=[spec])
    traj.steps.append(TrajectoryStep(step_type="tool_call", tool_name="lookup", tool_args={}))
    assert not ToolArgVerifier().check(traj).passed


def test_tool_arg_verifier_passes_on_valid_calls():
    traj = GEN.generate("weather in Tokyo", mode="single_agent")
    assert ToolArgVerifier().check(traj).passed


def test_safety_verifier_flags_dangerous_content():
    traj = Trajectory(query="x", final_answer="then run rm -rf / to clean up")
    assert not SafetyVerifier().check(traj).passed
    assert not verify_trajectory(traj, [SafetyVerifier()]).verified


def test_safety_verifier_clean():
    traj = GEN.generate("weather in Tokyo", mode="single_agent")
    assert SafetyVerifier().check(traj).passed


def test_expected_answer_verifier():
    traj = Trajectory(query="x", final_answer="The capital of France is Paris.")
    assert ExpectedAnswerVerifier("Paris").check(traj).passed
    assert not ExpectedAnswerVerifier("London").check(traj).passed


def test_non_required_failure_does_not_block_verified():
    traj = GEN.generate("weather in Tokyo", mode="single_agent")
    verifier = ExpectedAnswerVerifier("zzz-not-in-answer", required=False)
    result = verify_trajectory(traj, [verifier])
    assert not result.checks[0].passed
    assert result.verified  # the failing check wasn't required


def test_verify_trajectory_shape():
    traj = GEN.generate("weather in Tokyo", mode="single_agent")
    result = verify_trajectory(traj)
    assert result.trajectory_id == traj.id
    assert 0.0 <= result.score <= 1.0
    assert len(result.checks) == len(default_verifiers())


def test_batch_verify():
    trajs = GEN.generate_batch("weather in Tokyo", num_trajectories=5, vary_modes=True)
    results = batch_verify(trajs)
    assert len(results) == 5
    assert all(isinstance(r.verified, bool) for r in results)


def test_rubric_presets():
    assert set(rubric_names()) == {"balanced", "strict", "lenient", "safety_first"}
    assert get_rubric("strict")["pass_threshold"] == 0.75
    assert get_rubric("lenient")["pass_threshold"] == 0.45
    for name in rubric_names():
        assert abs(sum(get_rubric(name)["weights"].values()) - 1.0) < 1e-9
    with pytest.raises(ValueError):
        get_rubric("does-not-exist")


def test_ensemble_aggregates_and_votes():
    traj = GEN.generate("weather in Tokyo", mode="single_agent")
    evaluators = [
        TrajectoryEvaluator(use_mock=True, **get_rubric(name))
        for name in ("balanced", "strict", "lenient")
    ]
    ensemble = EnsembleEvaluator(evaluators).evaluate(traj)
    singles = [e.evaluate(traj).overall for e in evaluators]
    assert ensemble.overall == round(sum(singles) / len(singles), 4)
    assert ensemble.judge_model == "ensemble:3"
    assert "agreement" in ensemble.explanation


def test_ensemble_needs_at_least_one_evaluator():
    with pytest.raises(ValueError):
        EnsembleEvaluator([])
