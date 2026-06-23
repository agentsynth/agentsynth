"""Contamination audit — canaries, corpus overlap, held-out siblings."""

from agentsynth.contamination import (
    canary_for,
    contamination_report,
    corpus_overlap,
    held_out_pack,
)
from agentsynth.scenarios import AnswerContains, Scenario, SqlCheck


def _scenario(sid, task):
    return Scenario(
        id=sid,
        task=task,
        environment={
            "type": "sql",
            "schema": "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer TEXT, status TEXT)",
            "table": "orders",
            "rows": [[7, "alice", "paid"]],
        },
        checkers=[SqlCheck(query="SELECT status FROM orders WHERE id=7", equals=[["refunded"]])],
    )


def test_canary_is_deterministic_and_unique():
    assert canary_for("refund-7") == canary_for("refund-7")  # stable
    assert canary_for("refund-7") != canary_for("ship-1")  # unique per scenario
    assert canary_for("refund-7").startswith("agentsynth-canary-")


def test_corpus_overlap_catches_a_seen_task():
    scenario = _scenario("refund-7", "Refund order 7 for the customer who complained.")
    seen = ["Refund order 7 for the customer who complained.", "totally unrelated text here"]
    assert corpus_overlap(scenario, seen) > 0.9
    novel = ["the quick brown fox", "lorem ipsum dolor sit amet"]
    assert corpus_overlap(scenario, novel) < 0.2
    assert corpus_overlap(scenario, []) == 0.0


def test_contamination_report_flags_overlap_and_mints_canaries():
    scenarios = [
        _scenario("seen", "Cancel order 3 and refund the customer immediately please."),
        _scenario("fresh", "Reconcile the quarterly ledger against the shipping manifest."),
    ]
    corpus = ["Cancel order 3 and refund the customer immediately please."]
    report = contamination_report(scenarios, corpus=corpus, threshold=0.8)

    assert report.has_corpus is True
    assert report.flagged == 1
    flagged = {r.id for r in report.rows if r.contaminated}
    assert flagged == {"seen"}
    # every scenario still gets a canary
    assert all(r.canary.startswith("agentsynth-canary-") for r in report.rows)


def test_report_without_corpus_still_gives_canaries():
    scenarios = [_scenario("a", "do a"), _scenario("b", "do b")]
    report = contamination_report(scenarios)
    assert report.has_corpus is False
    assert report.flagged == 0
    assert all(r.max_overlap is None for r in report.rows)
    text = report.summary_md()
    assert "canary" in text.lower()


def test_held_out_pack_relabels_single_table_scenarios():
    scenario = Scenario(
        id="spend",
        task="Did alice or bob spend more?",
        environment={
            "type": "sql",
            "schema": "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer TEXT, total REAL)",
            "table": "orders",
            "rows": [[1, "alice", 120.0], [2, "bob", 300.0]],
        },
        checkers=[AnswerContains(any_of=["bob"])],
    )
    siblings = held_out_pack([scenario])
    assert len(siblings) == 1
    names = {r[1] for r in siblings[0].environment["rows"]}
    assert "alice" not in names and "bob" not in names  # relabelled, memorizing model misses
