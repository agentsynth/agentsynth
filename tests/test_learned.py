"""Learned-verifier tests.

Feature extraction is pure Python and runs everywhere. The training tests need
scikit-learn (the `learned` extra) and are skipped where it isn't installed.
"""

import statistics

import pytest

from agentsynth import AgentTrajectoryGenerator, TrajectoryEvaluator
from agentsynth.verification.learned import FEATURE_NAMES, extract_features


@pytest.fixture(scope="module")
def judged_batch():
    gen = AgentTrajectoryGenerator(use_mock=True)
    trajs = gen.generate_batch(
        "analyze sales by region and email the report", num_trajectories=40, vary_modes=True
    )
    results = TrajectoryEvaluator(use_mock=True).evaluate_batch(trajs)
    return trajs, results


# --- features (no sklearn needed) -------------------------------------------


def test_extract_features_shape_and_determinism(judged_batch):
    trajs, _ = judged_batch
    vec = extract_features(trajs[0])
    assert len(vec) == len(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in vec)
    assert vec == extract_features(trajs[0])  # deterministic


def test_features_reflect_the_trajectory(judged_batch):
    trajs, _ = judged_batch
    by_name = dict(zip(FEATURE_NAMES, extract_features(trajs[0])))
    assert by_name["num_steps"] == float(trajs[0].num_steps())
    assert by_name["num_tool_calls"] == float(len(trajs[0].tool_calls()))
    mode_flags = [by_name["mode_single"], by_name["mode_multi"], by_name["mode_code"]]
    assert sum(mode_flags) == 1.0


# --- training (skipped without scikit-learn) --------------------------------


def test_train_and_agreement(judged_batch):
    pytest.importorskip("sklearn")
    from agentsynth import train_learned_verifier
    from agentsynth.verification import verify_trajectory

    trajs, results = judged_batch
    # mock scores cluster high, so split labels at the median to get two classes
    median = statistics.median(r.overall for r in results)
    verifier, report = train_learned_verifier(trajs, results, threshold=median)

    assert report["n"] == len(trajs)
    assert 0.0 <= report["agreement"] <= 1.0
    assert report["features"] == list(FEATURE_NAMES)

    check = verifier.check(trajs[0])
    assert check.name == "learned_judge"
    assert "p(pass)=" in check.detail
    assert 0.0 <= verifier.predict_proba(trajs[0]) <= 1.0

    # plugs into the standard verification entrypoint as an advisory check
    outcome = verify_trajectory(trajs[0], verifiers=[verifier])
    assert outcome.checks[0].name == "learned_judge"
    assert outcome.verified  # advisory: required=False can't hard-fail it


def test_single_class_labels_raise(judged_batch):
    pytest.importorskip("sklearn")
    from agentsynth import train_learned_verifier

    trajs, results = judged_batch
    with pytest.raises(ValueError):
        train_learned_verifier(trajs, results, threshold=0.0)  # everything passes
