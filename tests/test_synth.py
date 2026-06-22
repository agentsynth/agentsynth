"""Auto-generating verifiable scenarios from a demonstration."""

import pytest

from agentsynth.robustness import audit_pack
from agentsynth.scenarios import AnswerContains, CalledTool, Scenario, SqlCheck, run_scenario_suite
from agentsynth.synth import pack_from_demonstrations, scenario_from_demonstration


def _replay_oracle(actions, answer):
    def solve(observation, gym):
        if gym.step_count < len(actions):
            return {"tool_name": "sql_query", "arguments": {"query": actions[gym.step_count]}}
        return {"answer": answer}

    return solve


def _sql_checks(scenario):
    return [c for c in scenario.checkers if isinstance(c, SqlCheck)]


def test_single_table_demo_derives_state_checks():
    scenario, actions = scenario_from_demonstration(
        task="Refund order 7. Leave the others alone.",
        schema="CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT)",
        rows=[[7, "paid"], [8, "paid"]],
        actions=["UPDATE orders SET status='refunded' WHERE id=7"],
        answer="Refunded order 7.",
        scenario_id="refund-7",
    )

    sql = _sql_checks(scenario)
    # the changed row is asserted by its end state, keyed on the primary key
    changed = [c for c in sql if "WHERE id=7" in c.query and c.equals == [[7, "refunded"]]]
    assert changed, [c.query for c in sql]
    # an untouched row is pinned as a witness against over-mutation
    assert any(c.equals == [[8, "paid"]] for c in sql)
    # the row count is pinned against stray inserts/deletes
    assert any("COUNT(*) FROM orders" in c.query and c.equals == [[2]] for c in sql)
    # grounding is asserted, but the scenario is graded on the world, not a magic
    # answer string — so there is no answer check, and the gold answer rides in metadata
    assert any(isinstance(c, CalledTool) for c in scenario.checkers)
    assert not any(isinstance(c, AnswerContains) for c in scenario.checkers)
    assert scenario.metadata.get("answer") == "Refunded order 7."


def test_generated_scenario_validates_and_audits_clean():
    scenario, actions = scenario_from_demonstration(
        task="Refund order 7.",
        schema="CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT)",
        rows=[[7, "paid"], [8, "paid"]],
        actions=["UPDATE orders SET status='refunded' WHERE id=7"],
        answer="Refunded order 7.",
    )
    # the demonstrated actions are a passing oracle
    report = run_scenario_suite(_replay_oracle(actions, "Refunded order 7."), [scenario])
    assert report.passed == 1
    # and no trivial adversary can game a state-change check
    audit = audit_pack([scenario])
    assert audit.robustness_score == 1.0


def test_multi_table_demo():
    scenario, actions = scenario_from_demonstration(
        task="Refund order 5 and restock its item W-100 by 3.",
        schema=(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT);\n"
            "CREATE TABLE inventory (sku TEXT PRIMARY KEY, stock INTEGER);\n"
            "INSERT INTO orders VALUES (5, 'paid'), (6, 'paid');\n"
            "INSERT INTO inventory VALUES ('W-100', 10);"
        ),
        actions=[
            "UPDATE orders SET status='refunded' WHERE id=5",
            "UPDATE inventory SET stock = stock + 3 WHERE sku='W-100'",
        ],
        answer="Refunded 5 and restocked W-100.",
        scenario_id="refund-and-restock",
    )
    sql = _sql_checks(scenario)
    assert any(c.equals == [[5, "refunded"]] for c in sql)  # order table changed
    assert any(c.equals == [["W-100", 13]] for c in sql)  # inventory table changed
    assert any(c.equals == [[6, "paid"]] for c in sql)  # untouched order witnessed

    report = run_scenario_suite(_replay_oracle(actions, "done"), [scenario])
    assert report.passed == 1
    assert audit_pack([scenario]).robustness_score == 1.0


def test_demo_that_changes_nothing_is_rejected():
    with pytest.raises(ValueError):
        scenario_from_demonstration(
            task="Just look.",
            schema="CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)",
            rows=[[1, "a"]],
            actions=["SELECT * FROM t"],
        )


def test_pack_from_demonstrations_round_trips(tmp_path):
    demos = [
        {
            "id": "refund",
            "task": "Refund order 7.",
            "schema": "CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT)",
            "rows": [[7, "paid"], [8, "paid"]],
            "actions": ["UPDATE orders SET status='refunded' WHERE id=7"],
            "answer": "Refunded order 7.",
        },
        {
            "id": "ship",
            "task": "Mark order 1 shipped.",
            "schema": "CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT)",
            "rows": [[1, "paid"], [2, "paid"]],
            "actions": ["UPDATE orders SET status='shipped' WHERE id=1"],
            "answer": "Shipped order 1.",
        },
        {
            "id": "cap",
            "task": "Cap salary of person 2 to 20000.",
            "schema": "CREATE TABLE payroll (id INTEGER PRIMARY KEY, monthly INTEGER)",
            "rows": [[1, 8000], [2, 30000]],
            "actions": ["UPDATE payroll SET monthly=20000 WHERE id=2"],
            "answer": "Capped person 2.",
        },
    ]
    pack_yaml, oracle_py = pack_from_demonstrations(demos, "demo_pack")

    import yaml

    scenarios = [Scenario(**item) for item in yaml.safe_load(pack_yaml)]
    assert len(scenarios) == 3

    namespace: dict = {}
    exec(oracle_py, namespace)
    report = run_scenario_suite(namespace["solve"], scenarios)
    assert report.passed == report.n  # the generated oracle solves the generated pack
    assert audit_pack(scenarios).robustness_score == 1.0  # all state-graded, none gameable
