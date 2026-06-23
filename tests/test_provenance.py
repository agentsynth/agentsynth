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
