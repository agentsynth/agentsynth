"""Trajectory verification: confirm a trajectory is actually sound, not just
judged sound by an LLM.

A Verifier runs one check and returns a CheckResult. `verify_trajectory` runs a
set of them and combines the outcome. A trajectory is "verified" when every
required check passes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from pydantic import BaseModel, Field

from ..schemas import Trajectory


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class VerificationResult(BaseModel):
    trajectory_id: str
    verified: bool
    score: float = 0.0  # fraction of checks that passed
    checks: List[CheckResult] = Field(default_factory=list)
    detail: str = ""


class Verifier(ABC):
    name: str = "verifier"
    required: bool = True  # a failed required check means the trajectory isn't verified

    @abstractmethod
    def check(self, trajectory: Trajectory) -> CheckResult:
        """Run the check against one trajectory."""


def verify_trajectory(
    trajectory: Trajectory, verifiers: Optional[List[Verifier]] = None
) -> VerificationResult:
    """Run a set of verifiers and combine their results.

    Defaults to the standard verifiers (tool args, execution grounding, safety).
    `verified` is True only if every *required* verifier passes.
    """
    if verifiers is None:
        from .verifiers import default_verifiers

        verifiers = default_verifiers()

    checks: List[CheckResult] = []
    for verifier in verifiers:
        try:
            result = verifier.check(trajectory)
        except Exception as exc:
            result = CheckResult(name=verifier.name, passed=False, detail=f"verifier error: {exc}")
        checks.append(result)

    passed = sum(1 for c in checks if c.passed)
    score = round(passed / len(checks), 3) if checks else 1.0
    verified = all(check.passed for check, verifier in zip(checks, verifiers) if verifier.required)
    detail = "; ".join(f"{c.name}: {'ok' if c.passed else 'FAIL'}" for c in checks)
    return VerificationResult(
        trajectory_id=trajectory.id,
        verified=verified,
        score=score,
        checks=checks,
        detail=detail,
    )


def batch_verify(
    trajectories: List[Trajectory],
    verifiers: Optional[List[Verifier]] = None,
    progress: Optional[Callable[..., object]] = None,
) -> List[VerificationResult]:
    results: List[VerificationResult] = []
    total = len(trajectories) or 1
    for i, traj in enumerate(trajectories):
        if callable(progress):
            try:
                progress(i / total, desc=f"Verifying {i + 1}/{total}")
            except Exception:
                pass
        results.append(verify_trajectory(traj, verifiers))
    return results


__all__ = [
    "CheckResult",
    "VerificationResult",
    "Verifier",
    "verify_trajectory",
    "batch_verify",
]
