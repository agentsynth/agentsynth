"""Scenarios: an environment with seed state, a task, and end-state checks.

Checkers run after the episode, against the final environment and the trajectory,
so the score reflects whether the goal state was reached rather than how the
transcript reads:

    scenario = Scenario(
        id="refund-7",
        task="Refund order 7",
        environment={"type": "sql", "schema": ..., "rows": [...], "read_only": False},
        checkers=[
            SqlCheck(query="SELECT status FROM orders WHERE id=7", equals=[["refunded"]]),
            CalledTool(name="sql_query"),
        ],
    )
    gym = AgentGym.from_scenario(scenario)   # outcome becomes the dominant reward

Scenarios serialize to YAML/JSON (`save_scenarios` / `load_scenarios`), and
`run_scenario_suite` scores a policy over a pack.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field

try:  # 3.9 ships Literal/Annotated in typing
    from typing import Annotated, Literal
except ImportError:  # pragma: no cover
    from typing_extensions import Annotated, Literal  # type: ignore

from .environments import Environment, PythonSandbox, RestEnvironment, SQLEnvironment
from .schemas import Trajectory


class CheckOutcome(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class SqlCheck(BaseModel):
    """Assert over the database's final state (needs a SQL environment)."""

    kind: Literal["sql"] = "sql"
    query: str
    equals: Optional[List[List[Any]]] = None  # exact rows, list-of-lists for YAML
    contains: Optional[str] = None  # substring of the stringified rows
    non_empty: bool = False

    def check(self, environment: Environment, trajectory: Trajectory) -> CheckOutcome:
        rows_fn = getattr(environment, "rows", None)
        if rows_fn is None:
            return CheckOutcome(name="sql", passed=False, detail="environment has no .rows()")
        try:
            rows = [list(r) for r in rows_fn(self.query)]
        except Exception as exc:
            return CheckOutcome(name="sql", passed=False, detail=f"query failed: {exc}")
        if self.equals is not None and rows != self.equals:
            return CheckOutcome(name="sql", passed=False, detail=f"got {rows!r}")
        if self.contains is not None and self.contains not in str(rows):
            return CheckOutcome(name="sql", passed=False, detail=f"{self.contains!r} not in rows")
        if self.non_empty and not rows:
            return CheckOutcome(name="sql", passed=False, detail="no rows came back")
        return CheckOutcome(name="sql", passed=True, detail=self.query)


class HttpCheck(BaseModel):
    """GET a path on the environment's API and assert on the body (REST scenarios)."""

    kind: Literal["http"] = "http"
    path: str
    contains: str

    def check(self, environment: Environment, trajectory: Trajectory) -> CheckOutcome:
        base = getattr(environment, "base_url", None)
        if not base:
            return CheckOutcome(name="http", passed=False, detail="environment has no base_url")
        from urllib import request

        try:
            with request.urlopen(base + self.path, timeout=10) as resp:
                body = resp.read().decode("utf-8", "replace")
        except Exception as exc:
            return CheckOutcome(name="http", passed=False, detail=f"GET {self.path}: {exc}")
        passed = self.contains in body
        detail = f"GET {self.path}" if passed else f"{self.contains!r} not in GET {self.path}"
        return CheckOutcome(name="http", passed=passed, detail=detail)


class CalledTool(BaseModel):
    """Assert the trajectory actually used a tool (optionally with given args)."""

    kind: Literal["called_tool"] = "called_tool"
    name: str
    args_contain: Optional[Dict[str, Any]] = None
    min_times: int = 1

    def check(self, environment: Environment, trajectory: Trajectory) -> CheckOutcome:
        hits = 0
        for call in trajectory.tool_calls():
            if call.tool_name != self.name:
                continue
            args = call.tool_args or {}
            if self.args_contain and not all(
                args.get(k) == v for k, v in self.args_contain.items()
            ):
                continue
            hits += 1
        passed = hits >= self.min_times
        return CheckOutcome(
            name=f"called:{self.name}",
            passed=passed,
            detail=f"{hits} matching call(s), needed {self.min_times}",
        )


class AnswerContains(BaseModel):
    """Assert the final answer mentions at least one of the expected strings."""

    kind: Literal["answer"] = "answer"
    any_of: List[str]
    case_sensitive: bool = False

    def check(self, environment: Environment, trajectory: Trajectory) -> CheckOutcome:
        answer = trajectory.final_answer or ""
        haystack = answer if self.case_sensitive else answer.lower()
        needles = self.any_of if self.case_sensitive else [s.lower() for s in self.any_of]
        passed = any(n in haystack for n in needles)
        detail = "answer mentions it" if passed else f"none of {self.any_of!r} in the answer"
        return CheckOutcome(name="answer", passed=passed, detail=detail)


_CODE_SENTINEL = "__agentsynth_tests_ok__"


class CodeCheck(BaseModel):
    """Run the agent's Python against hidden tests (needs a python environment).

    Gathers the code from every `python` tool call, appends the test, and runs the lot
    in the sandbox. Passes only when the tests run clean — the outcome is whether the
    code works, not whether the transcript claims it does.
    """

    kind: Literal["code"] = "code"
    test: str  # appended after the agent's code; assert what a correct solution must do

    def check(self, environment: Environment, trajectory: Trajectory) -> CheckOutcome:
        codes = [
            str((call.tool_args or {}).get("code", ""))
            for call in trajectory.tool_calls()
            if call.tool_name == "python"
        ]
        if not any(code.strip() for code in codes):
            return CheckOutcome(name="code", passed=False, detail="no python code was run")
        program = "\n".join(codes) + "\n" + self.test + f'\nprint("{_CODE_SENTINEL}")'
        try:
            if "python" in environment.tool_names():
                result = environment.execute("python", {"code": program})
            else:
                result = PythonSandbox().execute("python", {"code": program})
        except Exception as exc:  # the sandbox should not take the whole run down
            return CheckOutcome(name="code", passed=False, detail=f"{type(exc).__name__}: {exc}")
        passed = _CODE_SENTINEL in (result or "")
        detail = "tests passed" if passed else (result or "no output")[-200:]
        return CheckOutcome(name="code", passed=passed, detail=detail)


Checker = Annotated[
    Union[SqlCheck, HttpCheck, CalledTool, AnswerContains, CodeCheck],
    Field(discriminator="kind"),
]


class Scenario(BaseModel):
    """A serializable bundle: environment config, task, and outcome checkers."""

    id: str
    task: str
    environment: Dict[str, Any] = Field(default_factory=lambda: {"type": "sql"})
    checkers: List[Checker] = Field(default_factory=list)
    max_steps: int = 8
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def build_environment(self) -> Environment:
        """A fresh environment carrying this scenario's seed state."""
        config = dict(self.environment)
        env_type = config.pop("type", "sql")
        if env_type == "sql":
            if "rows" in config:
                config["rows"] = [tuple(r) for r in config["rows"]]
            config.setdefault("read_only", False)  # scenario databases are meant to be written
            return SQLEnvironment(**config)
        if env_type == "rest":
            return RestEnvironment(**config)
        if env_type == "python":
            return PythonSandbox(**config)
        raise ValueError(f"unknown scenario environment type {env_type!r}")

    def run_checks(
        self, environment: Environment, trajectory: Trajectory
    ) -> Tuple[float, List[CheckOutcome]]:
        """Outcome score in [0, 1] = fraction of checkers that pass on the end state."""
        if not self.checkers:
            return 1.0, []
        outcomes = [c.check(environment, trajectory) for c in self.checkers]
        score = sum(1 for o in outcomes if o.passed) / len(outcomes)
        return round(score, 6), outcomes


class ScenarioReport(BaseModel):
    n: int
    passed: int
    pass_rate: float
    results: List[Dict[str, Any]] = Field(default_factory=list)


def run_scenario_suite(
    policy: Any,
    scenarios: Sequence[Scenario],
    seed: int = 7,
    **gym_kwargs: Any,
) -> ScenarioReport:
    """Run a policy through every scenario. A scenario passes when every checker does.

    `policy(observation, gym) -> action`, the same shape `AgentGym.rollout` takes.
    """
    from .rl import AgentGym

    results: List[Dict[str, Any]] = []
    passed = 0
    for scenario in scenarios:
        gym = AgentGym.from_scenario(scenario, seed=seed, **gym_kwargs)
        try:
            episode = gym.rollout(policy)
        finally:
            gym.close()
        outcome = episode.info.get("outcome", {})
        ok = outcome.get("score", 0.0) >= 1.0
        passed += 1 if ok else 0
        results.append(
            {
                "id": scenario.id,
                "passed": ok,
                "outcome_score": outcome.get("score", 0.0),
                "reward": episode.total_reward,
                "checks": outcome.get("checks", []),
            }
        )
    n = len(results) or 1
    return ScenarioReport(
        n=len(results), passed=passed, pass_rate=round(passed / n, 4), results=results
    )


def save_scenarios(scenarios: Sequence[Scenario], path: str) -> str:
    """Write a scenario pack — YAML for .yaml/.yml, JSON otherwise."""
    payload = [s.model_dump() for s in scenarios]
    if path.endswith((".yaml", ".yml")):
        import yaml

        text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    else:
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def load_scenarios(path: str) -> List[Scenario]:
    """Read a scenario pack (YAML or JSON)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    if path.endswith((".yaml", ".yml")):
        import yaml

        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError("a scenario pack is a list of scenarios")
    return [Scenario(**item) for item in payload]


__all__ = [
    "CheckOutcome",
    "SqlCheck",
    "HttpCheck",
    "CalledTool",
    "AnswerContains",
    "CodeCheck",
    "Scenario",
    "ScenarioReport",
    "run_scenario_suite",
    "save_scenarios",
    "load_scenarios",
]
