"""Reliability statistics — beyond pass@1."""

from agentsynth.reliability import (
    pass_hat_k,
    reliability_report,
    wilson_interval,
)


def test_wilson_interval_brackets_the_rate_and_handles_edges():
    lo, hi = wilson_interval(8, 10)
    assert lo < 0.8 < hi
    assert 0.0 <= lo <= hi <= 1.0
    # the edges don't collapse to zero width the way a normal approximation would
    lo_all, hi_all = wilson_interval(10, 10)
    assert hi_all == 1.0 and lo_all < 1.0
    lo_none, hi_none = wilson_interval(0, 10)
    assert lo_none == 0.0 and hi_none > 0.0
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_pass_hat_k_is_the_all_pass_estimator():
    # a perfect scenario passes every k; a never-passing one passes no k
    assert pass_hat_k(10, 10, 1) == 1.0
    assert pass_hat_k(10, 10, 10) == 1.0
    assert pass_hat_k(0, 10, 1) == 0.0
    # k=1 is just the pass rate
    assert pass_hat_k(5, 10, 1) == 0.5
    # comb(5,2)/comb(10,2) = 10/45
    assert abs(pass_hat_k(5, 10, 2) - 10 / 45) < 1e-9
    # monotonically non-increasing in k
    vals = [pass_hat_k(6, 10, k) for k in range(1, 7)]
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


def test_reliability_report_curve_and_flakiness():
    passes = {
        "always": [True, True, True, True],
        "never": [False, False, False, False],
        "flaky": [True, False, True, False],
    }
    rel = reliability_report(passes, trials=4)

    assert rel.n_scenarios == 3
    # pass^1 = total passes / total attempts = (4 + 0 + 2) / 12
    assert rel.pass1 == round(6 / 12, 4)
    # only "always" passes every trial
    assert rel.passk == round(1 / 3, 4)
    # the curve starts at pass^1 and ends at pass^k, and never rises
    assert rel.curve[0] == rel.pass1
    assert rel.curve[-1] == rel.passk
    assert all(rel.curve[i] >= rel.curve[i + 1] for i in range(len(rel.curve) - 1))
    # confidence intervals bracket their point estimates
    assert rel.pass1_ci[0] <= rel.pass1 <= rel.pass1_ci[1]
    assert rel.passk_ci[0] <= rel.passk <= rel.passk_ci[1]
    # exactly one flaky scenario, surfaced by name
    flaky = rel.flaky
    assert [s.id for s in flaky] == ["flaky"]
    assert flaky[0].passes == 2 and flaky[0].trials == 4


def test_reliability_report_all_pass_is_full_reliability():
    rel = reliability_report({"a": [True, True], "b": [True, True]}, trials=2)
    assert rel.pass1 == 1.0 and rel.passk == 1.0
    assert rel.curve == [1.0, 1.0]
    assert rel.flaky == []


def test_summary_md_mentions_the_decay_and_flaky():
    rel = reliability_report({"a": [True, False, True], "b": [True, True, True]}, trials=3)
    text = rel.summary_md()
    assert "pass^1" in text and "pass^3" in text
    assert "decay" in text
    assert "flaky" in text and "a (2/3)" in text
