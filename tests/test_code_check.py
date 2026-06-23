"""CodeCheck and the code_v1 / policy_v1 domain packs."""

import importlib.util
from pathlib import Path

from agentsynth.robustness import audit_pack
from agentsynth.scenarios import (
    CalledTool,
    CodeCheck,
    Scenario,
    load_scenarios,
    run_scenario_suite,
)

REPO = Path(__file__).resolve().parent.parent


def _code_scenario(test):
    return Scenario(
        id="t",
        task="Write a function add(a, b).",
        environment={"type": "python"},
        checkers=[CalledTool(name="python"), CodeCheck(test=test)],
    )


def _writer(code):
    def solve(observation, gym):
        if gym.step_count == 0:
            return {"tool_name": "python", "arguments": {"code": code}}
        return {"answer": "done"}

    return solve


def test_codecheck_passes_when_the_code_works():
    scenario = _code_scenario("assert add(2, 3) == 5\nassert add(-1, 1) == 0")
    report = run_scenario_suite(_writer("def add(a, b):\n    return a + b"), [scenario])
    assert report.passed == 1


def test_codecheck_fails_on_wrong_code():
    scenario = _code_scenario("assert add(2, 3) == 5")
    report = run_scenario_suite(_writer("def add(a, b):\n    return a - b"), [scenario])
    assert report.passed == 0


def test_codecheck_fails_when_no_code_was_run():
    scenario = _code_scenario("assert add(2, 3) == 5")
    all_talk = lambda observation, gym: {"answer": "I wrote it, promise"}  # noqa: E731
    report = run_scenario_suite(all_talk, [scenario])
    assert report.passed == 0


def test_codecheck_fails_on_a_syntax_error():
    scenario = _code_scenario("assert add(2, 3) == 5")
    report = run_scenario_suite(_writer("def add(a, b) return a + b"), [scenario])
    assert report.passed == 0


def _load_solve(path):
    spec = importlib.util.spec_from_file_location("oracle_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solve


def test_domain_packs_validate_and_audit_clean():
    for name in ("code_v1", "policy_v1"):
        scenarios = load_scenarios(str(REPO / "packs" / f"{name}.yaml"))
        assert len(scenarios) >= 3
        oracle = _load_solve(str(REPO / "packs" / f"{name}_oracle.py"))
        report = run_scenario_suite(oracle, scenarios)
        assert report.passed == report.n, f"{name}: oracle should solve every scenario"
        assert audit_pack(scenarios).robustness_score == 1.0, f"{name}: should resist adversaries"
