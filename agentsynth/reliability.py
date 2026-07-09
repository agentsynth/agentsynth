"""Reliability statistics for a benchmark run — beyond a single pass@1 number.

A model that solves a task 6 times out of 10 isn't 60% reliable for an agent you ship;
what matters is how often it succeeds *every* time you ask. This follows the 2026
reliability-science framing (arXiv:2603.29231) and tau-bench's pass^k: report the whole
decay curve from pass^1 to pass^k, a confidence interval on each, and which scenarios are
flaky rather than cleanly passing or failing.

    rel = reliability_report({"refund-7": [True, True, False, True]}, trials=4)
    print(rel.summary_md())

`per_scenario_passes` maps a scenario id to its per-trial pass booleans (length = trials).
"""

from __future__ import annotations

from math import comb, sqrt
from typing import Dict, List, Tuple

from pydantic import BaseModel, Field


def wilson_interval(passes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial pass rate — sane near 0%, 100%, and small n.

    The plain `p ± z·sqrt(p(1-p)/n)` interval collapses to zero width at 0/n and n/n,
    which is exactly where a benchmark lands; Wilson doesn't.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = passes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4))


def pass_hat_k(c: int, n: int, k: int) -> float:
    """Unbiased estimate that all k of k sampled attempts pass, given c/n passed.

    `comb(c, k) / comb(n, k)` — the probability that k attempts drawn from n observed
    ones are all passes. k=1 gives the pass rate c/n; k=n gives 1.0 only if c==n. This
    is the all-must-pass analogue of the HumanEval pass@k estimator.
    """
    if k <= 0 or k > n or c < k:
        return 0.0 if c < k else 1.0
    return comb(c, k) / comb(n, k)


class ScenarioReliability(BaseModel):
    id: str
    passes: int
    trials: int
    rate: float

    @property
    def flaky(self) -> bool:
        return 0 < self.passes < self.trials


class ReliabilityReport(BaseModel):
    trials: int
    n_scenarios: int
    pass1: float  # mean per-attempt success rate
    pass1_ci: Tuple[float, float]  # Wilson CI over every scenario x trial attempt
    passk: float  # fraction of scenarios that passed every trial
    passk_ci: Tuple[float, float]  # Wilson CI over scenarios
    curve: List[float] = Field(default_factory=list)  # pass^1 .. pass^k, the decay
    scenarios: List[ScenarioReliability] = Field(default_factory=list)

    @property
    def flaky(self) -> List[ScenarioReliability]:
        return [s for s in self.scenarios if s.flaky]

    def summary_md(self) -> str:
        lo1, hi1 = self.pass1_ci
        lok, hik = self.passk_ci
        decay = " → ".join(f"{v:.0%}" for v in self.curve)
        lines = [
            f"Reliability over {self.trials} trials, {self.n_scenarios} scenarios:",
            f"- pass^1 (any single attempt): {self.pass1:.0%}  (95% CI {lo1:.0%}–{hi1:.0%})",
            f"- pass^{self.trials} (every attempt): {self.passk:.0%}  (95% CI {lok:.0%}–{hik:.0%})",
            f"- decay pass^1..pass^{self.trials}: {decay}",
        ]
        flaky = self.flaky
        if flaky:
            worst = ", ".join(f"{s.id} ({s.passes}/{s.trials})" for s in flaky)
            lines.append(f"- flaky (sometimes pass, sometimes fail): {worst}")
        return "\n".join(lines)


def reliability_report(
    per_scenario_passes: Dict[str, List[bool]], trials: int
) -> ReliabilityReport:
    """Turn per-trial pass booleans into the full reliability picture."""
    ids = list(per_scenario_passes)
    n = len(ids)
    counts = {sid: sum(1 for p in per_scenario_passes[sid] if p) for sid in ids}

    total_attempts = n * trials
    total_passes = sum(counts.values())
    pass1 = round(total_passes / total_attempts, 4) if total_attempts else 0.0

    all_pass = sum(1 for sid in ids if counts[sid] == trials)
    passk = round(all_pass / n, 4) if n else 0.0

    curve = []
    for k in range(1, trials + 1):
        if not n:
            curve.append(0.0)
            continue
        curve.append(round(sum(pass_hat_k(counts[sid], trials, k) for sid in ids) / n, 4))

    scenarios = [
        ScenarioReliability(
            id=sid,
            passes=counts[sid],
            trials=trials,
            rate=round(counts[sid] / trials, 4) if trials else 0.0,
        )
        for sid in ids
    ]

    return ReliabilityReport(
        trials=trials,
        n_scenarios=n,
        pass1=pass1,
        pass1_ci=wilson_interval(total_passes, total_attempts),
        passk=passk,
        passk_ci=wilson_interval(all_pass, n),
        curve=curve,
        scenarios=scenarios,
    )


__all__ = [
    "wilson_interval",
    "pass_hat_k",
    "ScenarioReliability",
    "ReliabilityReport",
    "reliability_report",
]
