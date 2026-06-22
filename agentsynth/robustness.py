"""Robustness audit: how gameable is a pack's verification?

Outcome checks beat judge-only grading, but a checker can still be gamed — a canned
answer, an echoed prompt, a throwaway tool call that changes nothing. This module
measures that, so a pack ships with a known robustness floor instead of a hidden one.

It operationalizes the failure modes the 2026 "LLMs gaming verifiers" work flagged
(arXiv:2604.15149): answers that leaked into the prompt, state checks that already
hold on the untouched seed, and credit paid for words rather than for the world.

    from agentsynth.robustness import audit_pack
    report = audit_pack(load_scenarios("packs/core_v2.yaml"))
    print(report.summary_md())

The adversaries below need no knowledge of a task — they are the cheap attacks any
contaminated or lazy model would stumble into. A scenario no adversary passes is
robust; one that falls is graded on something other than a state change.

`perturb_scenario` + `ipt_report` go further for the generalizing case: build an
isomorphic sibling (rename the labels, keep the structure) and confirm a real solver
still passes while a replayed transcript no longer does.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

Policy = Callable[[str, Any], Dict[str, Any]]


# -- trivial adversaries: no task knowledge, just the cheap exploits ------------------


def noop_policy(observation: str, gym: Any) -> Dict[str, Any]:
    """Talk, take no action — the do-nothing baseline."""
    return {"answer": "all done"}


def constant_policy(text: str = "done") -> Policy:
    """Always return the same fixed answer, whatever the task."""

    def policy(observation: str, gym: Any) -> Dict[str, Any]:
        return {"answer": text}

    return policy


def echo_policy(observation: str, gym: Any) -> Dict[str, Any]:
    """Echo the prompt back — passes any answer check whose target leaked into it."""
    return {"answer": gym.task}


def echo_with_probe_policy(observation: str, gym: Any) -> Dict[str, Any]:
    """One harmless read, then echo the prompt.

    Beats scenarios gated by `called_tool` + a leaked answer + a state check that only
    asserts the seed is unchanged: the probe satisfies the tool check and changes
    nothing, the echo satisfies the leaked answer.
    """
    if gym.step_count == 0 and "sql_query" in gym.environment.tool_names():
        return {"tool_name": "sql_query", "arguments": {"query": "SELECT 1"}}
    return {"answer": gym.task}


ADVERSARIES: Dict[str, Policy] = {
    "noop": noop_policy,
    "constant": constant_policy(),
    "echo": echo_policy,
    "echo+probe": echo_with_probe_policy,
}


# -- static detectors ----------------------------------------------------------------


def answer_leaks(scenario: Any) -> List[str]:
    """Answer-check targets that already appear as whole words in the task prompt.

    When the expected word is in the prompt, an echo passes the check without solving
    anything — the answer is graded on text the model was handed. Matched on word
    boundaries so a short token like "2" doesn't false-positive inside "2025".
    """
    import re

    from .scenarios import AnswerContains

    task = (scenario.task or "").lower()
    leaked: List[str] = []
    for checker in scenario.checkers:
        if isinstance(checker, AnswerContains):
            for token in checker.any_of:
                tok = token.lower()
                if tok in leaked:
                    continue
                if re.search(r"(?<!\w)" + re.escape(tok) + r"(?!\w)", task):
                    leaked.append(tok)
    return leaked


def _state_checks_hold_on_seed(checks: List[Dict[str, Any]]) -> bool:
    """True when every state check (sql/http) passed with no action taken.

    Vacuously true for a scenario with no state check at all — which is itself the
    signal: nothing asserts the world changed, so grading rests on answer/tool checks.
    """
    state = [c for c in checks if str(c.get("name", "")).startswith(("sql", "http"))]
    return all(c.get("passed") for c in state)


# -- the report ----------------------------------------------------------------------


class ScenarioRobustness(BaseModel):
    scenario_id: str
    gamed_by: List[str] = Field(default_factory=list)
    answer_leaks: List[str] = Field(default_factory=list)
    state_noop_satisfiable: bool = False
    note: str = ""

    @property
    def robust(self) -> bool:
        return not self.gamed_by


class RobustnessReport(BaseModel):
    n: int
    robust: int
    gameable: int
    robustness_score: float
    adversaries: List[str] = Field(default_factory=list)
    rows: List[ScenarioRobustness] = Field(default_factory=list)

    def summary_md(self) -> str:
        lines = [
            f"Robustness {self.robustness_score:.0%} — {self.robust}/{self.n} scenarios "
            "resisted every trivial adversary "
            f"({', '.join(self.adversaries)}).",
            "",
            "| scenario | gamed by | answer leaks | state asserts a change |",
            "| --- | --- | --- | --- |",
        ]
        for row in self.rows:
            gamed = ", ".join(row.gamed_by) if row.gamed_by else "—"
            leaks = ", ".join(row.answer_leaks) if row.answer_leaks else "—"
            asserts_change = "no" if row.state_noop_satisfiable else "yes"
            lines.append(
                f"| {row.scenario_id} | {gamed} | {leaks} | {asserts_change} |"
            )
        weak = [r for r in self.rows if not r.robust or r.answer_leaks]
        if weak:
            lines.append("")
            lines.append("To harden these: assert the state *change* (a SqlCheck on the")
            lines.append("written value), drop expected words that already sit in the prompt,")
            lines.append("or accept that read/refusal tasks are text-graded and label them.")
        return "\n".join(lines)


def audit_pack(
    scenarios: List[Any],
    seed: int = 7,
    adversaries: Optional[Dict[str, Policy]] = None,
) -> RobustnessReport:
    """Run the trivial adversaries across a pack and report what they passed."""
    from .scenarios import run_scenario_suite

    advs = adversaries or ADVERSARIES
    ids = [s.id for s in scenarios]
    gamed: Dict[str, List[str]] = {sid: [] for sid in ids}
    noop_checks: Dict[str, List[Dict[str, Any]]] = {sid: [] for sid in ids}

    for name, policy in advs.items():
        report = run_scenario_suite(policy, scenarios, seed=seed)
        for row in report.results:
            if row["passed"]:
                gamed[row["id"]].append(name)
            if name == "noop":
                noop_checks[row["id"]] = row.get("checks", [])

    rows: List[ScenarioRobustness] = []
    robust = 0
    for scenario in scenarios:
        leaks = answer_leaks(scenario)
        noop_ok = _state_checks_hold_on_seed(noop_checks.get(scenario.id, []))
        gb = gamed[scenario.id]
        if not gb:
            robust += 1
        note = ""
        if gb:
            note = "passes without solving the task"
        elif noop_ok:
            note = "no state-change assertion; grading rests on answer/tool checks"
        rows.append(
            ScenarioRobustness(
                scenario_id=scenario.id,
                gamed_by=gb,
                answer_leaks=leaks,
                state_noop_satisfiable=noop_ok,
                note=note,
            )
        )

    n = len(scenarios) or 1
    return RobustnessReport(
        n=len(scenarios),
        robust=robust,
        gameable=len(scenarios) - robust,
        robustness_score=round(robust / n, 4),
        adversaries=list(advs.keys()),
        rows=rows,
    )


# -- isomorphic perturbation (for the generalizing case) -----------------------------


def _string_labels(rows: List[List[Any]]) -> List[str]:
    """Distinct text values in seed rows that read like labels, not numbers."""
    labels: List[str] = []
    for row in rows:
        for cell in row:
            if isinstance(cell, str) and len(cell) >= 3 and not cell.replace(".", "").isdigit():
                if cell not in labels:
                    labels.append(cell)
    return labels


def _relabel(value: str, index: int) -> str:
    """A stable, distinct rewrite of a label that keeps its shape (email stays email)."""
    if "@" in value:
        local, _, domain = value.partition("@")
        return f"{local[::-1]}{index}@{domain}"
    if "-" in value:
        head, _, tail = value.partition("-")
        return f"{head[::-1]}-{tail}{index}"
    return f"{value[::-1]}{index}"


def perturb_scenario(scenario: Any, seed: int = 0) -> Any:
    """An isomorphic sibling: rename string labels, keep every number and the structure.

    Renaming labels (names, emails, SKUs) preserves every relational truth and numeric
    threshold while changing the surface tokens — so a policy that truly solves the task
    still passes, but one echoing a memorized answer fails. Single-table scenarios only;
    multi-table schemas (raw SQL with INSERTs) are returned unchanged.
    """
    from .scenarios import AnswerContains, Scenario, SqlCheck

    data = scenario.model_dump()
    rows = data.get("environment", {}).get("rows") or []
    if not rows:  # multi-table packs carry their data in the schema; leave them be
        return Scenario(**data)

    labels = _string_labels([list(r) for r in rows])
    mapping = {old: _relabel(old, seed + i) for i, old in enumerate(labels)}
    if not mapping:
        return Scenario(**data)

    def swap(text: str) -> str:
        for old, new in mapping.items():
            text = text.replace(old, new)
        return text

    data["id"] = f"{scenario.id}~perturbed"
    data["task"] = swap(scenario.task)
    data["environment"]["rows"] = [
        [mapping.get(cell, cell) if isinstance(cell, str) else cell for cell in row]
        for row in rows
    ]

    new_checkers: List[Dict[str, Any]] = []
    for checker in scenario.checkers:
        cd = checker.model_dump()
        if isinstance(checker, SqlCheck):
            cd["query"] = swap(checker.query)
            if checker.equals is not None:
                cd["equals"] = [
                    [mapping.get(v, v) if isinstance(v, str) else v for v in row]
                    for row in checker.equals
                ]
            if checker.contains is not None:
                cd["contains"] = swap(checker.contains)
        elif isinstance(checker, AnswerContains):
            cd["any_of"] = [mapping.get(s, swap(s)) for s in checker.any_of]
        new_checkers.append(cd)
    data["checkers"] = new_checkers
    return Scenario(**data)


def record_actions(scenario: Any, policy: Policy, seed: int = 7) -> List[Dict[str, Any]]:
    """The concrete action sequence a policy takes on a scenario."""
    from .rl import AgentGym

    gym = AgentGym.from_scenario(scenario, seed=seed)
    actions: List[Dict[str, Any]] = []

    def recording(observation: str, g: Any) -> Dict[str, Any]:
        action = policy(observation, g)
        actions.append(action)
        return action

    try:
        gym.rollout(recording)
    finally:
        gym.close()
    return actions


def replay_policy(actions: List[Dict[str, Any]]) -> Policy:
    """Replay a fixed action sequence regardless of what the environment says back."""
    plan = list(actions)

    def policy(observation: str, gym: Any) -> Dict[str, Any]:
        i = gym.step_count
        if i < len(plan):
            return plan[i]
        return {"answer": ""}

    return policy


def ipt_report(scenario: Any, policy: Policy, seed: int = 7) -> Dict[str, Any]:
    """Isomorphic perturbation test for a (claimed) generalizing policy.

    Two properties a trustworthy outcome check should have:

    - the policy still passes the perturbed sibling (it solved the task, not the instance)
    - replaying the policy's original actions on the sibling now fails (the check rewards
      the state change, not a memorized transcript)
    """
    from .scenarios import run_scenario_suite

    sibling = perturb_scenario(scenario, seed=seed)
    generalizes = run_scenario_suite(policy, [sibling], seed=seed).passed == 1

    original_actions = record_actions(scenario, policy, seed=seed)
    replay = replay_policy(original_actions)
    replay_passes = run_scenario_suite(replay, [sibling], seed=seed).passed == 1

    return {
        "scenario_id": scenario.id,
        "perturbed": sibling.id,
        "policy_generalizes": generalizes,
        "replay_blocked": not replay_passes,
        "robust": generalizes and not replay_passes,
    }


__all__ = [
    "ADVERSARIES",
    "noop_policy",
    "constant_policy",
    "echo_policy",
    "echo_with_probe_policy",
    "answer_leaks",
    "ScenarioRobustness",
    "RobustnessReport",
    "audit_pack",
    "perturb_scenario",
    "record_actions",
    "replay_policy",
    "ipt_report",
]
