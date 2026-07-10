"""Reproducible run manifests — verify a leaderboard score can be re-derived."""

from agentsynth.provenance import pack_fingerprint, run_manifest, verify_run
from agentsynth.scenarios import CalledTool, Scenario, SqlCheck, run_scenario_suite


def _refund_scenario(sid="refund-7", expected="refunded"):
    return Scenario(
        id=sid,
        task="Refund order 7.",
        environment={
            "type": "sql",
            "schema": "CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT)",
            "table": "orders",
            "rows": [[7, "paid"], [8, "paid"]],
        },
        checkers=[
            SqlCheck(query="SELECT status FROM orders WHERE id=7", equals=[[expected]]),
            CalledTool(name="sql_query"),
        ],
    )


_REFUND_SQL = "UPDATE orders SET status='refunded' WHERE id=7"


def _oracle(observation, gym):
    if gym.step_count == 0:
        return {"tool_name": "sql_query", "arguments": {"query": _REFUND_SQL}}
    return {"answer": "done"}


def _lazy(observation, gym):
    return {"answer": "all done"}


def _manifest_for(scenarios, policy, seed=7):
    report = run_scenario_suite(policy, scenarios, seed=seed)
    return run_manifest("p", scenarios, report, model="oracle", seed=seed, trials=1)


def test_pack_fingerprint_is_stable_and_content_sensitive():
    a = [_refund_scenario()]
    assert pack_fingerprint(a) == pack_fingerprint([_refund_scenario()])  # same content, same hash
    changed = [_refund_scenario(expected="cancelled")]  # a different checker
    assert pack_fingerprint(a) != pack_fingerprint(changed)


def test_manifest_carries_the_run_hash_and_results():
    scenarios = [_refund_scenario()]
    manifest = _manifest_for(scenarios, _oracle)
    assert manifest["pack_fingerprint"] == pack_fingerprint(scenarios)
    assert manifest["run_hash"] and len(manifest["run_hash"]) >= 8
    assert manifest["results"][0]["id"] == "refund-7"
    assert "client_version" in manifest


def test_same_policy_reproduces_exactly():
    scenarios = [_refund_scenario()]
    manifest = _manifest_for(scenarios, _oracle)
    result = verify_run(manifest, scenarios, _oracle)
    assert result["pack_intact"] is True
    assert result["reproduced"] is True
    assert result["pass_rate_delta"] == 0.0


def test_a_different_policy_does_not_reproduce():
    scenarios = [_refund_scenario()]
    manifest = _manifest_for(scenarios, _oracle)  # oracle passes
    result = verify_run(manifest, scenarios, _lazy)  # lazy fails
    assert result["pack_intact"] is True  # pack unchanged
    assert result["reproduced"] is False  # but the result can't be reproduced
    assert result["actual_hash"] != result["expected_hash"]


def test_a_tampered_pack_is_caught():
    scenarios = [_refund_scenario()]
    manifest = _manifest_for(scenarios, _oracle)
    tampered = [_refund_scenario(expected="cancelled")]  # checker edited after the run
    result = verify_run(manifest, tampered, _oracle)
    assert result["pack_intact"] is False
    assert result["reproduced"] is False


def test_tolerance_accepts_a_matching_pass_rate():
    scenarios = [_refund_scenario()]
    manifest = _manifest_for(scenarios, _oracle)
    # same pass rate, exact hash — within any tolerance
    result = verify_run(manifest, scenarios, _oracle, tolerance=0.1)
    assert result["within_tolerance"] is True


def test_cost_rides_along_without_touching_the_hash():
    scenarios = [_refund_scenario()]
    report = run_scenario_suite(_oracle, scenarios, seed=7)
    bare = run_manifest("p", scenarios, report, model="oracle", seed=7, trials=1)
    costed = run_manifest(
        "p",
        scenarios,
        report,
        model="oracle",
        seed=7,
        trials=1,
        cost={"usd": 0.0123, "total_tokens": 456, "calls": 3},
    )
    assert "cost" not in bare
    assert costed["cost"] == {"usd": 0.0123, "total_tokens": 456, "calls": 3}
    # same outcomes -> same run_hash, spend isn't part of what "reproduced" means
    assert costed["run_hash"] == bare["run_hash"]


def test_zero_cost_is_treated_as_no_cost():
    scenarios = [_refund_scenario()]
    manifest = _manifest_for(scenarios, _oracle)
    with_empty = run_manifest(
        "p",
        scenarios,
        run_scenario_suite(_oracle, scenarios, seed=7),
        model="oracle",
        seed=7,
        trials=1,
        cost={},
    )
    assert "cost" not in with_empty
    assert "cost" not in manifest
