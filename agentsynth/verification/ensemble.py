"""Judge ensembles.

Run several evaluators over a trajectory and combine them: mean per-dimension
scores, mean overall, a majority pass vote, and an agreement number (how close the
judges were). Useful with several LLM judges, or several rubric presets, to get a
calibrated verdict instead of a single opinion.
"""

from __future__ import annotations

import statistics
from typing import Any, Callable, List, Optional

from ..evaluator import TrajectoryEvaluator
from ..schemas import RUBRIC_DIMENSIONS, EvalResult, RubricScores, Trajectory


class EnsembleEvaluator:
    def __init__(self, evaluators: List[TrajectoryEvaluator]) -> None:
        if not evaluators:
            raise ValueError("EnsembleEvaluator needs at least one evaluator")
        self.evaluators = list(evaluators)

    def evaluate(self, trajectory: Trajectory) -> EvalResult:
        results = [e.evaluate(trajectory) for e in self.evaluators]
        n = len(results)

        dim_means = {
            dim: round(statistics.fmean(float(getattr(r.scores, dim)) for r in results), 3)
            for dim in RUBRIC_DIMENSIONS
        }
        overall = round(statistics.fmean(r.overall for r in results), 4)
        passes = sum(1 for r in results if r.passed)
        passed = passes * 2 >= n  # majority, ties pass

        overalls = [r.overall for r in results]
        spread = (max(overalls) - min(overalls)) if n > 1 else 0.0
        agreement = round(1.0 - spread, 3)

        explanation = (
            f"Ensemble of {n} judges: {passes}/{n} voted pass; "
            f"agreement {agreement} (overall spread {round(spread, 3)})."
        )
        return EvalResult(
            trajectory_id=trajectory.id,
            scores=RubricScores(**dim_means),
            overall=overall,
            passed=passed,
            explanation=explanation,
            judge_model=f"ensemble:{n}",
        )

    def evaluate_batch(
        self,
        trajectories: List[Trajectory],
        progress: Optional[Callable[..., Any]] = None,
    ) -> List[EvalResult]:
        out: List[EvalResult] = []
        total = len(trajectories) or 1
        for i, traj in enumerate(trajectories):
            if callable(progress):
                try:
                    progress(i / total, desc=f"Judging {i + 1}/{total}")
                except Exception:
                    pass
            out.append(self.evaluate(traj))
        return out


__all__ = ["EnsembleEvaluator"]
