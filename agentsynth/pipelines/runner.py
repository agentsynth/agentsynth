"""Run a recipe: generate, optionally evaluate, compute metrics, export.

Generation can run concurrently (`max_workers > 1`), which helps when the backend
is a real LLM. Results are reassembled in index order, so the output is identical
to a sequential run.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..schemas import EvalResult, Trajectory
from ..verification.base import VerificationResult
from .recipe import Recipe, make_environment

_VARY_CYCLE = ("single_agent", "multi_agent", "code_execution")


@dataclass
class RunResult:
    trajectories: List[Trajectory]
    eval_results: List[EvalResult] = field(default_factory=list)
    verifications: List[VerificationResult] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    output_path: Optional[str] = None
    duplicates_removed: int = 0


def _plan(recipe: Recipe) -> List[Tuple[str, int, str]]:
    """Resolve a recipe into a list of (query, index, mode)."""
    explicit_n = "num_trajectories" in recipe.model_fields_set
    count = recipe.num_trajectories
    if recipe.queries:
        queries = list(recipe.queries)
        base: Optional[List[str]] = queries
        # An explicit query list means "one per query" unless a count was set on purpose.
        if not explicit_n:
            count = len(queries)
    elif recipe.query is not None:
        base = None  # single query -> N variations by index
    else:
        from ..tasks import sample_tasks

        base = [t.query for t in sample_tasks(recipe.num_trajectories, recipe.domains, recipe.seed)]
        base = base or None

    plan: List[Tuple[str, int, str]] = []
    for i in range(max(0, count)):
        if base is None:
            query = recipe.query if recipe.query is not None else "Complete the task."
        else:
            query = base[i % len(base)]
        mode = _VARY_CYCLE[i % len(_VARY_CYCLE)] if recipe.vary_modes else recipe.mode
        plan.append((query, i, mode))
    return plan


def _tick(progress: Optional[Callable[..., Any]], done: List[int], total: int) -> None:
    done[0] += 1
    if callable(progress):
        try:
            progress(done[0] / total, desc=f"Generating {done[0]}/{total}")
        except Exception:
            pass


def run_recipe(recipe: Recipe, progress: Optional[Callable[..., Any]] = None) -> RunResult:
    from ..generator import AgentTrajectoryGenerator

    env = make_environment(recipe.environment)
    gen = AgentTrajectoryGenerator(
        model=recipe.model,
        temperature=recipe.temperature,
        max_steps=recipe.max_steps,
        use_mock=recipe.use_mock,
        seed=recipe.seed,
        environment=env,
    )

    plan = _plan(recipe)
    total = len(plan) or 1
    done = [0]

    def work(item: Tuple[str, int, str]) -> Tuple[int, Trajectory]:
        query, idx, mode = item
        return idx, gen.generate(query, mode=mode, index=idx)

    pairs: List[Tuple[int, Trajectory]] = []
    if recipe.max_workers and recipe.max_workers > 1 and plan:
        with ThreadPoolExecutor(max_workers=recipe.max_workers) as pool:
            for idx, traj in pool.map(work, plan):
                pairs.append((idx, traj))
                _tick(progress, done, total)
    else:
        for item in plan:
            pairs.append(work(item))
            _tick(progress, done, total)

    pairs.sort(key=lambda p: p[0])
    trajectories = [t for _, t in pairs]

    duplicates_removed = 0
    if recipe.dedup:
        from ..dedup import dedup_trajectories

        deduped = dedup_trajectories(trajectories, threshold=recipe.dedup_threshold)
        duplicates_removed = len(deduped.removed)
        trajectories = deduped.kept

    eval_results: List[EvalResult] = []
    if recipe.evaluate:
        from ..evaluator import TrajectoryEvaluator

        judge_kwargs: Dict[str, Any] = {"model": recipe.model, "use_mock": recipe.use_mock}
        if recipe.rubric:
            from ..verification.rubrics import get_rubric

            judge_kwargs.update(get_rubric(recipe.rubric))
        evaluator = TrajectoryEvaluator(**judge_kwargs)
        eval_results = evaluator.evaluate_batch(trajectories)

    verifications: List[VerificationResult] = []
    if recipe.verify:
        from ..verification.base import batch_verify

        verifications = batch_verify(trajectories)

    from ..metrics import compute_dataset_metrics

    metrics = compute_dataset_metrics(trajectories, eval_results or None)
    if verifications:
        metrics["verified_rate"] = round(
            sum(1 for v in verifications if v.verified) / len(verifications), 4
        )
    if duplicates_removed:
        metrics["duplicates_removed"] = duplicates_removed

    output_path: Optional[str] = None
    if recipe.export_format and recipe.export_path:
        from ..exporters import save_dataset

        output_path = save_dataset(
            trajectories,
            recipe.export_path,
            fmt=recipe.export_format,
            eval_results=eval_results or None,
        )

    if env is not None:
        env.close()

    return RunResult(
        trajectories=trajectories,
        eval_results=eval_results,
        verifications=verifications,
        metrics=metrics,
        output_path=output_path,
        duplicates_removed=duplicates_removed,
    )


__all__ = ["RunResult", "run_recipe"]
