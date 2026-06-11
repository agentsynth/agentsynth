"""Failure-mining tests: the mine-failures leg of the flywheel."""

import pytest

from agentsynth import (
    AgentTrajectoryGenerator,
    TrajectoryEvaluator,
    mine_failures,
    mine_judge_failures,
    recipe_from_failures,
    run_recipe,
    targeted_queries,
)
from agentsynth.benchmarks import BUILTIN_CASES, run_benchmark


def _weather_only(query, tools):
    """A deliberately weak model: always calls get_weather with no args."""
    return "get_weather", {}


def _mute(query, tools):
    return None, {}


@pytest.fixture(scope="module")
def weak_report():
    report = run_benchmark(_weather_only, BUILTIN_CASES)
    return mine_failures(report, BUILTIN_CASES)


# --- benchmark mining ---------------------------------------------------------


def test_failures_are_categorized(weak_report):
    kinds = weak_report.by_kind
    assert kinds.get("wrong_tool")  # calculator/search/... cases got get_weather
    assert kinds.get("bad_args")  # weather cases: right tool, missing city
    assert sum(kinds.values()) == len(weak_report.failures)
    assert weak_report.n_cases == len(BUILTIN_CASES)
    assert all(f.query for f in weak_report.failures)  # queries resolved from cases


def test_mute_model_yields_invalid_calls():
    report = run_benchmark(_mute, BUILTIN_CASES)
    mined = mine_failures(report, BUILTIN_CASES)
    assert mined.by_kind == {"invalid_call": len(BUILTIN_CASES)}


def test_perfect_model_yields_no_failures():
    def perfect(query, tools):
        case = next(c for c in BUILTIN_CASES if c.query == query)
        args = {k: (v if v is not None else "x") for k, v in case.expected_args.items()}
        return case.expected_tool, args

    report = run_benchmark(perfect, BUILTIN_CASES)
    mined = mine_failures(report, BUILTIN_CASES)
    assert mined.failures == []
    with pytest.raises(ValueError):
        recipe_from_failures(mined)


def test_summary_md_reads_like_a_report(weak_report):
    text = weak_report.summary_md()
    assert "wrong_tool" in text and "most-missed tools" in text


# --- targeting ----------------------------------------------------------------


def test_targeted_queries_deterministic_and_bounded(weak_report):
    a = targeted_queries(weak_report, k=12, seed=3)
    b = targeted_queries(weak_report, k=12, seed=3)
    assert a == b
    assert len(a) == 12 == len(set(a))
    # every variant keeps the original task text somewhere inside
    originals = {f.query for f in weak_report.failures}
    assert all(any(orig in q for orig in originals) for q in a)


def test_round_robin_covers_every_failure_first(weak_report):
    n_failures = len(weak_report.failures)
    queries = targeted_queries(weak_report, k=n_failures, seed=3)
    # one variant per distinct failed case before any repeats
    assert len(queries) == n_failures


def test_recipe_from_failures_closes_the_loop(weak_report):
    recipe = recipe_from_failures(weak_report, k=6, evaluate=False)
    assert recipe.verify is True  # on by default
    result = run_recipe(recipe)
    assert len(result.trajectories) == 6
    assert result.metrics.get("verified_rate") is not None


# --- judge mining ---------------------------------------------------------------


@pytest.fixture(scope="module")
def judged_batch():
    gen = AgentTrajectoryGenerator(use_mock=True)
    trajs = gen.generate_batch(
        "analyze sales and email a report", num_trajectories=10, vary_modes=True
    )
    return trajs, TrajectoryEvaluator(use_mock=True).evaluate_batch(trajs)


def test_judge_failures_flag_low_dimensions(judged_batch):
    trajs, results = judged_batch
    mined = mine_judge_failures(trajs, results, threshold=0.99)
    assert mined.failures and all(f.kind.startswith("judge:") for f in mined.failures)
    assert mined.weak_dimensions  # nearly everything is below 0.99
    assert all(f.query for f in mined.failures)  # queries resolved from trajectories

    quiet = mine_judge_failures(trajs, results, threshold=0.0)
    assert quiet.failures == [] and quiet.weak_dimensions == []


def test_judge_failures_feed_the_same_recipe_path(judged_batch):
    trajs, results = judged_batch
    mined = mine_judge_failures(trajs, results, threshold=0.99)
    recipe = recipe_from_failures(mined, k=4, evaluate=False)
    assert len(recipe.queries) == 4
