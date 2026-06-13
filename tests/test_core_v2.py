"""core_v2 is the flagship pack: it must validate, span tiers, and stay multi-table."""

from agentsynth.cli import main as cli_main
from agentsynth.scenarios import load_scenarios

PACK = "packs/core_v2.yaml"


def test_core_v2_passes_the_gate(capsys):
    code = cli_main(["pack", "validate", PACK])
    out = capsys.readouterr().out
    assert code == 0
    assert "oracle passes 14/14" in out
    assert "do-nothing policy passes 0/14" in out
    assert "PACK OK" in out


def test_core_v2_spans_difficulty_tiers():
    scenarios = load_scenarios(PACK)
    tiers = {s.metadata.get("tier") for s in scenarios}
    assert {"easy", "medium", "hard"} <= tiers


def test_core_v2_has_multi_table_scenarios():
    scenarios = load_scenarios(PACK)
    multi = [s for s in scenarios if s.metadata.get("multi_table")]
    assert len(multi) >= 3
    # a multi-table world declares more than one CREATE TABLE in its schema
    for s in multi:
        assert s.environment["schema"].upper().count("CREATE TABLE") >= 2


def test_core_v2_oracle_keeps_two_tables_in_agreement():
    # refund-and-restock must end with the order refunded AND inventory restocked
    scenarios = {s.id: s for s in load_scenarios(PACK)}
    from agentsynth.rl import AgentGym
    from packs.core_v2_oracle import solve

    gym = AgentGym.from_scenario(scenarios["refund-and-restock"], seed=7)
    try:
        episode = gym.rollout(solve)
        outcome = episode.info["outcome"]
        assert outcome["score"] == 1.0
        assert all(c["passed"] for c in outcome["checks"])
    finally:
        gym.close()
