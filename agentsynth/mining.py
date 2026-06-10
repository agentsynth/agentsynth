"""Close the flywheel: mine failures, then aim the next generation run at them.

    generate -> verify -> train -> evaluate -> mine failures -> generate ...

A benchmark run tells you *that* a model fails; this module turns it into *what to
generate next*. `mine_failures` categorizes every miss (didn't emit a call, picked
the wrong tool, fumbled the arguments), `mine_judge_failures` does the same for
judge scores below a threshold, and `recipe_from_failures` converts the report into
a ready-to-run Recipe whose queries are variations of exactly the tasks the model
got wrong.

The variations are deterministic templates — deliberately mock-first, like the rest
of the engine. Swap in an LLM paraphraser when you want richer expansions; the loop
structure stays the same.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from .benchmarks.tool_calling import BenchmarkCase, BenchmarkReport
from .pipelines import Recipe
from .schemas import RUBRIC_DIMENSIONS, EvalResult, Trajectory
from .utils import stable_seed

_TEMPLATES = (
    "{query}",
    "{query} Pick the right tool for this.",
    "Here's the task: {query}",
    "{query} Be precise with the arguments.",
    "A user asks: {query}",
    "{query} Double-check the values you pass.",
)


class Failure(BaseModel):
    kind: str  # invalid_call | wrong_tool | bad_args | judge:<dimension>
    case_id: str
    query: str
    expected_tool: Optional[str] = None
    predicted_tool: Optional[str] = None
    detail: str = ""


class FailureReport(BaseModel):
    n_cases: int
    failures: List[Failure] = Field(default_factory=list)
    by_kind: Dict[str, int] = Field(default_factory=dict)
    weak_dimensions: List[str] = Field(default_factory=list)

    def summary_md(self) -> str:
        lines = [f"Failure report — {len(self.failures)} failures over {self.n_cases} cases", ""]
        for kind, count in sorted(self.by_kind.items(), key=lambda kv: -kv[1]):
            lines.append(f"- **{kind}**: {count}")
        if self.weak_dimensions:
            lines.append(f"- weakest judge dimensions: {', '.join(self.weak_dimensions)}")
        missed = [f.expected_tool for f in self.failures if f.expected_tool]
        if missed:
            top = sorted(set(missed), key=missed.count, reverse=True)[:5]
            lines.append(f"- most-missed tools: {', '.join(top)}")
        return "\n".join(lines)


def mine_failures(report: BenchmarkReport, cases: Sequence[BenchmarkCase]) -> FailureReport:
    """Categorize every benchmark miss so the next run can target it."""
    case_by_id = {c.id: c for c in cases}
    failures: List[Failure] = []
    for result in report.results:
        if result.tool_ok and result.args_ok:
            continue
        if result.predicted_tool is None:
            kind, detail = "invalid_call", "no parsable tool call came back"
        elif not result.tool_ok:
            kind = "wrong_tool"
            detail = f"picked {result.predicted_tool!r}, expected {result.expected_tool!r}"
        else:
            kind, detail = "bad_args", "right tool, wrong or missing arguments"
        case = case_by_id.get(result.id)
        failures.append(
            Failure(
                kind=kind,
                case_id=result.id,
                query=case.query if case else "",
                expected_tool=result.expected_tool,
                predicted_tool=result.predicted_tool,
                detail=detail,
            )
        )
    return FailureReport(n_cases=report.n, failures=failures, by_kind=_count_kinds(failures))


def mine_judge_failures(
    trajectories: Sequence[Trajectory],
    eval_results: Sequence[EvalResult],
    threshold: float = 0.7,
) -> FailureReport:
    """Flag every rubric dimension scoring below `threshold`, per trajectory."""
    traj_by_id = {t.id: t for t in trajectories}
    failures: List[Failure] = []
    sums = {dim: 0.0 for dim in RUBRIC_DIMENSIONS}
    for result in eval_results:
        flat = result.flat()
        traj = traj_by_id.get(result.trajectory_id)
        for dim in RUBRIC_DIMENSIONS:
            score = float(flat.get(dim, 0.0))
            sums[dim] += score
            if score < threshold:
                failures.append(
                    Failure(
                        kind=f"judge:{dim}",
                        case_id=result.trajectory_id,
                        query=traj.query if traj else "",
                        detail=f"{dim}={score:.2f} < {threshold:.2f}",
                    )
                )
    n = len(eval_results) or 1
    weak = sorted(
        (dim for dim in RUBRIC_DIMENSIONS if sums[dim] / n < threshold),
        key=lambda dim: sums[dim],
    )
    return FailureReport(
        n_cases=len(eval_results),
        failures=failures,
        by_kind=_count_kinds(failures),
        weak_dimensions=weak,
    )


def _count_kinds(failures: Sequence[Failure]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for failure in failures:
        counts[failure.kind] = counts.get(failure.kind, 0) + 1
    return counts


def targeted_queries(report: FailureReport, k: int = 20, seed: int = 7) -> List[str]:
    """`k` query variations aimed at the failed tasks, deterministic for a seed.

    Failures are visited round-robin so every weak spot gets coverage before any
    gets a second variant."""
    sources = [f for f in report.failures if f.query]
    queries: List[str] = []
    seen = set()
    round_no = 0
    while sources and len(queries) < k:
        for failure in sources:
            template = _TEMPLATES[
                stable_seed(seed, f"{failure.case_id}#{round_no}") % len(_TEMPLATES)
            ]
            query = template.format(query=failure.query)
            if query not in seen:
                seen.add(query)
                queries.append(query)
            if len(queries) >= k:
                break
        round_no += 1
        if round_no > k:  # template space exhausted — stop rather than spin
            break
    return queries


def recipe_from_failures(
    report: FailureReport, k: int = 20, seed: int = 7, **recipe_kwargs
) -> Recipe:
    """A ready-to-run Recipe whose queries chase the report's failures.

    Defaults to `verify=True` (the whole point is trustworthy patches); any Recipe
    field can be overridden through `recipe_kwargs`."""
    queries = targeted_queries(report, k=k, seed=seed)
    if not queries:
        raise ValueError("the failure report has no failures with queries to target")
    recipe_kwargs.setdefault("verify", True)
    return Recipe(queries=queries, **recipe_kwargs)


__all__ = [
    "Failure",
    "FailureReport",
    "mine_failures",
    "mine_judge_failures",
    "targeted_queries",
    "recipe_from_failures",
]
