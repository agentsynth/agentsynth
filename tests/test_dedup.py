from agentsynth.dedup import decontaminate, dedup_trajectories, jaccard
from agentsynth.schemas import Trajectory


def _traj(query, answer="done"):
    return Trajectory(query=query, final_answer=answer)


def test_jaccard():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert 0.0 < jaccard({"a", "b"}, {"a", "c"}) < 1.0
    assert jaccard(set(), set()) == 1.0


def test_dedup_removes_identical_content():
    a = _traj("what is the total revenue by region this quarter", "EMEA leads at 41k")
    b = _traj("what is the total revenue by region this quarter", "EMEA leads at 41k")
    c = _traj("write a python function to compute fibonacci numbers quickly", "use memoization")
    result = dedup_trajectories([a, b, c])
    assert len(result.kept) == 2
    assert len(result.removed) == 1
    assert result.duplicates and result.duplicates[0][2] >= 0.85


def test_dedup_keeps_distinct_trajectories():
    trajs = [
        _traj("alpha beta gamma delta epsilon zeta"),
        _traj("one two three four five six seven"),
    ]
    result = dedup_trajectories(trajs)
    assert len(result.kept) == 2
    assert not result.removed


def test_dedup_keeps_same_query_with_different_content():
    # Same prompt, genuinely different trajectories -> not duplicates by default.
    a = _traj("summarize the sales data", "revenue rose 12% led by APAC widgets")
    b = _traj("summarize the sales data", "units fell in EMEA while gizmos grew in AMER")
    result = dedup_trajectories([a, b])
    assert len(result.kept) == 2


def test_dedup_query_key_collapses_same_prompt():
    a = _traj("summarize the sales data", "answer one")
    b = _traj("summarize the sales data", "answer two")
    result = dedup_trajectories([a, b], key=lambda t: t.query)
    assert len(result.kept) == 1
    assert len(result.removed) == 1


def test_dedup_removes_repeated_object():
    # The same object appearing twice (e.g. merged datasets) is a duplicate.
    traj = _traj("this exact trajectory shows up twice in a merged set", "answer")
    result = dedup_trajectories([traj, traj])
    assert len(result.kept) == 1
    assert len(result.removed) == 1


def test_decontaminate_flags_benchmark_overlap():
    benchmark = "explain the difference between sft and dpo for training agents"
    trajs = [_traj(benchmark), _traj("what is the weather in tokyo today please")]
    clean, flagged = decontaminate(trajs, [benchmark])
    assert len(flagged) == 1
    assert len(clean) == 1
    assert flagged[0].query == benchmark
