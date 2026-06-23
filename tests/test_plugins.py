"""The plugin registry for custom environments."""

import pytest

from agentsynth import PythonSandbox
from agentsynth.plugins import (
    _REGISTRY,
    available_environments,
    get_environment_factory,
    register_environment,
)
from agentsynth.scenarios import CalledTool, CodeCheck, Scenario, run_scenario_suite


@pytest.fixture(autouse=True)
def _clean_registry():
    before = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(before)


def test_register_and_resolve():
    register_environment("custom_py", lambda **cfg: PythonSandbox())
    assert "custom_py" in available_environments()
    factory = get_environment_factory("custom_py")
    assert factory is not None
    assert factory().__class__.__name__ == "PythonSandbox"


def test_scenario_uses_a_registered_environment():
    register_environment("custom_py", lambda **cfg: PythonSandbox())
    scenario = Scenario(
        id="plugin-scenario",
        task="Define add(a, b).",
        environment={"type": "custom_py"},
        checkers=[CalledTool(name="python"), CodeCheck(test="assert add(2, 3) == 5")],
    )
    env = scenario.build_environment()
    assert env.__class__.__name__ == "PythonSandbox"

    add_code = "def add(a, b):\n    return a + b"

    def policy(observation, gym):
        if gym.step_count == 0:
            return {"tool_name": "python", "arguments": {"code": add_code}}
        return {"answer": "done"}

    report = run_scenario_suite(policy, [scenario])
    assert report.passed == 1  # the custom env ran the code and the test passed


def test_unknown_environment_type_still_raises():
    scenario = Scenario(id="x", task="t", environment={"type": "nope_not_registered"})
    with pytest.raises(ValueError, match="unknown scenario environment"):
        scenario.build_environment()
