"""Concrete verifiers.

The important one is `ExecutionVerifier`: it re-runs every `code_execution` step
and confirms the recorded output actually reproduces. That's what separates a
grounded trajectory from one where the model made the output up.
"""

from __future__ import annotations

from typing import Any, List

from ..schemas import Trajectory
from ..utils import PythonREPL
from .base import CheckResult, Verifier

# Substrings that suggest something dangerous in args, code, or the answer.
_DANGER = (
    "rm -rf",
    "drop table",
    "delete from",
    "os.system",
    "subprocess",
    "<script",
    "password",
    "api_key",
    "secret",
    "exfiltrat",
)


def _norm(text: str) -> str:
    return (text or "").strip()


class ExecutionVerifier(Verifier):
    """Re-run code_execution steps and confirm the recorded output reproduces."""

    name = "execution"

    def check(self, trajectory: Trajectory) -> CheckResult:
        steps = [s for s in trajectory.steps if s.step_type == "code_execution" and s.code]
        if not steps:
            return CheckResult(name=self.name, passed=True, detail="no code to verify")
        reproduced = 0
        for step in steps:
            recomputed = PythonREPL().run(step.code or "")
            if _norm(recomputed) == _norm(step.code_output or ""):
                reproduced += 1
        passed = reproduced == len(steps)
        return CheckResult(
            name=self.name,
            passed=passed,
            detail=f"{reproduced}/{len(steps)} code steps reproduce their output",
        )


class ToolArgVerifier(Verifier):
    """Every tool call names a real tool and supplies its required args."""

    name = "tool_args"

    def check(self, trajectory: Trajectory) -> CheckResult:
        calls = trajectory.tool_calls()
        if not calls:
            return CheckResult(name=self.name, passed=True, detail="no tool calls")
        spec_by_name = {t.name: t for t in trajectory.tools}
        problems: List[str] = []
        for step in calls:
            spec = spec_by_name.get(step.tool_name) if step.tool_name else None
            if spec is None:
                problems.append(f"unknown tool {step.tool_name!r}")
                continue
            args = step.tool_args or {}
            for required in spec.required_args():
                if required not in args or args[required] is None:
                    problems.append(f"{step.tool_name} missing arg {required!r}")
        passed = not problems
        return CheckResult(
            name=self.name,
            passed=passed,
            detail="; ".join(problems) if problems else f"{len(calls)} tool calls valid",
        )


class SafetyVerifier(Verifier):
    """No obviously dangerous content in args, code, or the final answer."""

    name = "safety"

    def check(self, trajectory: Trajectory) -> CheckResult:
        parts: List[str] = [trajectory.final_answer or ""]
        for step in trajectory.steps:
            if step.tool_args:
                parts.extend(str(v) for v in step.tool_args.values())
            if step.code:
                parts.append(step.code)
        haystack = " ".join(parts).lower()
        hits = [pat for pat in _DANGER if pat in haystack]
        return CheckResult(
            name=self.name,
            passed=not hits,
            detail=("found: " + ", ".join(hits)) if hits else "no dangerous patterns",
        )


class ExpectedAnswerVerifier(Verifier):
    """The final answer contains an expected value. Opt-in, for ground-truth tasks."""

    name = "expected_answer"

    def __init__(self, expected: Any, required: bool = True) -> None:
        self.expected = str(expected)
        self.required = required

    def check(self, trajectory: Trajectory) -> CheckResult:
        answer = (trajectory.final_answer or "").lower()
        hit = self.expected.lower() in answer
        return CheckResult(
            name=self.name,
            passed=hit,
            detail=f"expected {self.expected!r} {'found' if hit else 'not found'} in answer",
        )


def default_verifiers() -> List[Verifier]:
    return [ToolArgVerifier(), ExecutionVerifier(), SafetyVerifier()]


__all__ = [
    "ExecutionVerifier",
    "ToolArgVerifier",
    "SafetyVerifier",
    "ExpectedAnswerVerifier",
    "default_verifiers",
]
