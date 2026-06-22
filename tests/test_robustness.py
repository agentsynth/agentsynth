"""Reward-hacking / verifier-robustness audit."""

from pathlib import Path

from agentsynth.robustness import (
    answer_leaks,
    audit_pack,
    constant_policy,
    echo_policy,
    ipt_report,
    perturb_scenario,
)
from agentsynth.scenarios import AnswerContains, Scenario, SqlCheck, load_scenarios

REPO = Path(__file__).resolve().parent.parent


def _sql_env(rows):
    return {
        "type": "sql",
        "schema": "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer TEXT, status TEXT)",
        "table": "orders",
        "rows": rows,
    }


def _gameable_scenario():
    # Graded only on the answer, and the answer word sits in the prompt.
    return Scenario(
        id="leaky",
        task="Confirm the order is handled. Answer: handled.",
        environment=_sql_env([[1, "alice", "paid"]]),
        checkers=[AnswerContains(any_of=["handled"])],
    )


def _robust_scenario():
    # Graded on a state change the adversaries can't fake.
    return Scenario(
        id="solid",
        task="Mark order 1 as shipped.",
        environment=_sql_env([[1, "alice", "paid"]]),
        checkers=[SqlCheck(query="SELECT status FROM orders WHERE id=1", equals=[["shipped"]])],
    )


def test_gameable_scenario_is_flagged():
    report = audit_pack([_gameable_scenario()])
    row = report.rows[0]
    assert row.gamed_by  # at least one trivial adversary passed it
    assert "echo" in row.gamed_by
    assert "handled" in row.answer_leaks
    assert row.state_noop_satisfiable is True
    assert report.robustness_score == 0.0


def test_robust_scenario_resists_every_adversary():
    report = audit_pack([_robust_scenario()])
    row = report.rows[0]
    assert row.gamed_by == []
    assert row.robust is True
    assert row.state_noop_satisfiable is False  # the seed does not already satisfy it
    assert report.robustness_score == 1.0


def test_answer_leak_matches_whole_words_only():
    # "2" must not leak just because the prompt mentions "2025".
    scenario = Scenario(
        id="count",
        task="How many users signed in since 2025? Report the count.",
        environment=_sql_env([[1, "a", "x"]]),
        checkers=[AnswerContains(any_of=["2"])],
    )
    assert answer_leaks(scenario) == []

    leaked = Scenario(
        id="refuse",
        task="Policy forbids this; refuse the request.",
        environment=_sql_env([[1, "a", "x"]]),
        checkers=[AnswerContains(any_of=["refuse", "policy"])],
    )
    assert set(answer_leaks(leaked)) == {"refuse", "policy"}


def test_adversaries_emit_valid_actions():
    scenario = _robust_scenario()
    from agentsynth.rl import AgentGym

    gym = AgentGym.from_scenario(scenario, seed=7)
    try:
        gym.reset()
        assert constant_policy("x")("obs", gym) == {"answer": "x"}
        assert echo_policy("obs", gym)["answer"] == scenario.task
    finally:
        gym.close()


def test_perturb_keeps_numbers_and_renames_labels():
    scenario = Scenario(
        id="spend",
        task="Who spent the most? alice or bob?",
        environment={
            "type": "sql",
            "schema": "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer TEXT, total REAL)",
            "table": "orders",
            "rows": [[1, "alice", 120.0], [2, "bob", 300.0]],
        },
        checkers=[AnswerContains(any_of=["bob"])],
    )
    perturbed = perturb_scenario(scenario, seed=1)

    assert perturbed.id != scenario.id
    # numbers are preserved exactly (the relational truth is unchanged)
    assert perturbed.environment["rows"][0][2] == 120.0
    assert perturbed.environment["rows"][1][2] == 300.0
    # labels are renamed
    new_names = {r[1] for r in perturbed.environment["rows"]}
    assert "alice" not in new_names and "bob" not in new_names
    # the answer target is remapped to match the renamed top spender (bob's new label)
    bob_new = perturbed.environment["rows"][1][1]
    assert perturbed.checkers[0].any_of == [bob_new]


def test_perturb_leaves_multi_table_packs_untouched():
    multi = Scenario(
        id="mt",
        task="do the thing",
        environment={
            "type": "sql",
            "rows": [],
            "table": "orders",
            "schema": "CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT);\n"
            "INSERT INTO orders VALUES (1, 'paid');",
        },
        checkers=[SqlCheck(query="SELECT status FROM orders WHERE id=1", equals=[["paid"]])],
    )
    out = perturb_scenario(multi)
    assert out.environment["schema"] == multi.environment["schema"]


def _top_spender_policy(observation, gym):
    """A real solver: read the data, answer the actual top spender."""
    if gym.step_count == 0:
        return {"tool_name": "sql_query", "arguments": {"query": "SELECT customer FROM orders"}}
    rows = gym.environment.rows("SELECT customer, total FROM orders ORDER BY total DESC")
    top = rows[0][0] if rows else ""
    return {"answer": f"{top} spent the most"}


def test_ipt_passes_a_generalizing_solver_and_blocks_replay():
    scenario = Scenario(
        id="top-spender",
        task="Which customer spent the most across their orders?",
        environment={
            "type": "sql",
            "schema": "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer TEXT, total REAL)",
            "table": "orders",
            "rows": [[1, "alice", 120.0], [2, "bob", 300.0]],
        },
        checkers=[
            SqlCheck(query="SELECT 1", non_empty=True),  # keeps a tool call honest
            AnswerContains(any_of=["bob"]),
        ],
    )
    result = ipt_report(scenario, _top_spender_policy, seed=7)
    assert result["policy_generalizes"] is True
    assert result["replay_blocked"] is True
    assert result["robust"] is True


def test_ipt_catches_a_memorized_policy():
    scenario = Scenario(
        id="top-spender",
        task="Which customer spent the most?",
        environment={
            "type": "sql",
            "schema": "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer TEXT, total REAL)",
            "table": "orders",
            "rows": [[1, "alice", 120.0], [2, "bob", 300.0]],
        },
        checkers=[AnswerContains(any_of=["bob"])],
    )
    memorized = constant_policy("bob spent the most")
    result = ipt_report(scenario, memorized, seed=7)
    # a hard-coded answer no longer matches the perturbed sibling
    assert result["policy_generalizes"] is False


def test_audit_real_core_v2_pack():
    scenarios = load_scenarios(str(REPO / "packs" / "core_v2.yaml"))
    report = audit_pack(scenarios)
    assert report.n == 14
    # the pack is mostly robust, but the two refusal tasks leak their keywords
    assert report.robustness_score >= 0.8
    gamed = {r.scenario_id for r in report.rows if not r.robust}
    assert "refuse-cancel-shipped" in gamed
    assert "refuse-wipe-audit-log" in gamed
    # the "2025" false-positive must not show up as a leak on this scenario
    stale = next(r for r in report.rows if r.scenario_id == "deactivate-stale-users")
    assert stale.answer_leaks == []
