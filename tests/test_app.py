"""Tests for the Gradio app's batch-explorer helpers.

The app imports gradio at module top, so these only run where the app extra is
installed — they're skipped on the 3.9 core interpreter.
"""

import pytest

pytest.importorskip("gradio")

import app  # noqa: E402
from agentsynth import AgentTrajectoryGenerator, TrajectoryEvaluator  # noqa: E402


@pytest.fixture(scope="module")
def batch():
    gen = AgentTrajectoryGenerator(use_mock=True)
    trajs = gen.generate_batch(
        "weather in Paris and an 18% tip", num_trajectories=6, vary_modes=True
    )
    results = TrajectoryEvaluator(use_mock=True).evaluate_batch(trajs)
    return trajs, results


def test_overview_rows_shape_and_idx(batch):
    trajs, _ = batch
    rows = app.traj_overview_rows(trajs)
    assert len(rows) == len(trajs)
    assert [r[0] for r in rows] == list(range(len(trajs)))  # idx lives in column 0
    assert all(len(r) == len(app._OVERVIEW_HEADERS) for r in rows)
    assert all(r[4] == "—" for r in rows)  # score column, nothing judged yet


def test_overview_rows_fill_score_when_judged(batch):
    trajs, results = batch
    rows = app.traj_overview_rows(trajs, results)
    assert all(isinstance(r[4], float) for r in rows)  # score is numeric once judged


def test_filter_by_mode_preserves_true_idx(batch):
    trajs, results = batch
    rows = app.filter_overview_rows(trajs, results, "code_execution", 0.0)
    assert rows  # vary_modes guarantees some code_execution
    assert all(r[1] == "code_execution" for r in rows)
    assert all(trajs[r[0]].mode == "code_execution" for r in rows)  # col 0 maps back


def test_filter_by_min_score_is_a_subset(batch):
    trajs, results = batch
    everything = app.filter_overview_rows(trajs, results, "all", 0.0)
    high = app.filter_overview_rows(trajs, results, "all", 0.99)
    assert len(everything) == len(trajs)
    assert len(high) <= len(everything)


def test_select_handler_maps_a_filtered_row_to_the_right_trajectory(batch):
    import types

    import pandas as pd

    trajs, results = batch
    # filter to a subset so the visible row index no longer equals the true index
    rows = app.filter_overview_rows(trajs, results, "code_execution", 0.0)
    df = pd.DataFrame(rows, columns=app._OVERVIEW_HEADERS)  # the shape Gradio passes
    true_idx = int(df.iloc[0, 0])
    evt = types.SimpleNamespace(index=[0, 0])  # the user clicks the first visible row
    detail = app.do_select_trajectory(trajs, results, df, evt)
    assert trajs[true_idx].id in detail
    assert trajs[true_idx].mode == "code_execution"
    assert "Judge" in detail  # the verdict is rendered because results were passed


def test_render_detail_with_and_without_eval(batch):
    trajs, results = batch
    plain = app.render_trajectory_detail(trajs[0])
    assert trajs[0].id in plain
    judged = app.render_trajectory_detail(trajs[0], results)
    assert "Judge" in judged
    assert app.render_trajectory_detail(None).startswith("_")


def test_agent_run_renders_outcome_and_timeline():
    sid = next(iter(sorted(app._DEMO_SCENARIOS)))  # whatever the demo pack leads with
    out = app.do_agent_run(sid, "expert (inspect-act-verify)", "mock (offline)")
    assert "Outcome checks" in out and "PASS" in out
    assert 'class="traj"' in out  # the episode timeline renders below the card

    lazy = app.do_agent_run(sid, "lazy (just talks)", "mock (offline)")
    assert "FAIL" in lazy and "✗" in lazy


def test_agent_run_guards_the_llm_policy():
    sid = next(iter(sorted(app._DEMO_SCENARIOS)))
    out = app.do_agent_run(sid, app._LLM_POLICY_LABEL, "mock (offline)")
    assert "needs a model id" in out


def test_compare_tab_builds_the_pass_k_table():
    out = app.do_compare(list(app.DEMO_POLICIES), "mock (offline)", 2)
    assert "pass^2" in out and "pass^1 avg" in out
    assert "✓" in out and "✗" in out
    assert "100%" in out and "0%" in out  # expert clears it, lazy holds at zero


def test_compare_tab_needs_two_items():
    out = app.do_compare(["lazy (just talks)"], "mock (offline)", 1)
    assert "at least two" in out
