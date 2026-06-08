from agentsynth import AgentTrajectoryGenerator, TrajectoryEvaluator
from agentsynth.preferences import build_preference_pairs, load_dpo_jsonl, to_dpo_jsonl

GEN = AgentTrajectoryGenerator(use_mock=True)
EV = TrajectoryEvaluator(use_mock=True)


def test_build_pairs_chosen_outscores_rejected():
    pairs = build_preference_pairs(GEN, EV, "analyze sales by region and email a summary", k=8)
    assert pairs
    for pair in pairs:
        assert pair.chosen_score >= pair.rejected_score
        assert pair.margin >= 0.0
        assert pair.chosen and pair.rejected
        assert pair.prompt == [{"role": "user", "content": pair.query}]


def test_min_margin_filters_pairs():
    pairs = build_preference_pairs(GEN, EV, "weather in Tokyo", k=6, min_margin=0.9)
    assert all(p.margin >= 0.9 for p in pairs)


def test_multiple_queries():
    queries = ["analyze the quarterly revenue data", "write code to sort a list"]
    pairs = build_preference_pairs(GEN, EV, queries, k=6)
    assert {p.query for p in pairs}.issubset(set(queries))


def test_dpo_export_roundtrip(tmp_path):
    pairs = build_preference_pairs(GEN, EV, "compute the mean of 4 8 15 16 23", k=6)
    assert pairs
    path = tmp_path / "dpo.jsonl"
    to_dpo_jsonl(pairs, str(path))
    rows = load_dpo_jsonl(str(path))
    assert len(rows) == len(pairs)
    for row in rows:
        assert {"prompt", "chosen", "rejected", "margin"} <= set(row)
        assert row["chosen"] and row["rejected"]


def test_determinism():
    a = build_preference_pairs(GEN, EV, "analyze sales by region", k=6)
    b = build_preference_pairs(GEN, EV, "analyze sales by region", k=6)
    assert [(p.chosen, p.rejected) for p in a] == [(p.chosen, p.rejected) for p in b]
