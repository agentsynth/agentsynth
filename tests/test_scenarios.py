"""Scenario tests: seedable worlds, outcome checkers, and outcome-dominant rewards."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agentsynth import (
    AgentGym,
    AnswerContains,
    CalledTool,
    HttpCheck,
    Scenario,
    SqlCheck,
    load_scenarios,
    run_scenario_suite,
    save_scenarios,
)
from agentsynth.environments import SQLEnvironment

ORDERS_ENV = {
    "type": "sql",
    "schema": "CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT)",
    "rows": [[7, "paid"], [8, "paid"]],
    "table": "orders",
}

REFUND_SCENARIO = Scenario(
    id="refund-7",
    task="Refund order 7 in the orders database, then confirm.",
    environment=ORDERS_ENV,
    checkers=[
        SqlCheck(query="SELECT status FROM orders WHERE id=7", equals=[["refunded"]]),
        SqlCheck(query="SELECT status FROM orders WHERE id=8", equals=[["paid"]]),
        CalledTool(name="sql_query"),
        AnswerContains(any_of=["refund"]),
    ],
)


# --- writable SQL environment ---------------------------------------------------


def test_sql_env_is_read_only_by_default():
    env = SQLEnvironment()
    assert env.execute("sql_query", {"query": "DELETE FROM sales"}).startswith("SQLError")
    env.close()


def test_writable_sql_env_mutates_and_reseeds():
    env = Scenario(id="x", task="t", environment=ORDERS_ENV).build_environment()
    out = env.execute("sql_query", {"query": "UPDATE orders SET status='refunded' WHERE id=7"})
    assert out == "OK: 1 row(s) affected"
    assert env.rows("SELECT status FROM orders WHERE id=7") == [("refunded",)]
    env.reset()  # the seed state comes back — episodes start clean
    assert env.rows("SELECT status FROM orders WHERE id=7") == [("paid",)]
    env.close()


# --- checkers --------------------------------------------------------------------


def test_sql_check_modes():
    env = REFUND_SCENARIO.build_environment()
    ok = SqlCheck(query="SELECT status FROM orders WHERE id=7", equals=[["paid"]])
    bad = SqlCheck(query="SELECT status FROM orders WHERE id=7", equals=[["refunded"]])
    sub = SqlCheck(query="SELECT * FROM orders", contains="paid")
    some = SqlCheck(query="SELECT * FROM orders", non_empty=True)
    broken = SqlCheck(query="SELECT * FROM nope", non_empty=True)
    assert ok.check(env, None).passed
    assert not bad.check(env, None).passed
    assert sub.check(env, None).passed
    assert some.check(env, None).passed
    assert not broken.check(env, None).passed
    env.close()


def test_called_tool_and_answer_checkers():
    from agentsynth import AgentTrajectoryGenerator

    traj = AgentTrajectoryGenerator(use_mock=True).generate("weather in Hanoi please")
    used = traj.tool_names_used()[0]
    assert CalledTool(name=used).check(None, traj).passed
    assert not CalledTool(name=used, min_times=99).check(None, traj).passed
    assert not CalledTool(name="never_called").check(None, traj).passed
    assert AnswerContains(any_of=[traj.final_answer.split()[0]]).check(None, traj).passed
    assert not AnswerContains(any_of=["zzz-not-there"]).check(None, traj).passed


# --- outcome-dominant rewards through the gym -------------------------------------


def _good_policy():
    script = iter(
        [
            {
                "tool_name": "sql_query",
                "arguments": {"query": "UPDATE orders SET status='refunded' WHERE id=7"},
            },
            {"answer": "Order 7 has been refunded."},
        ]
    )
    return lambda obs, gym: next(script)


def test_scenario_gym_rewards_the_actual_outcome():
    gym = AgentGym.from_scenario(REFUND_SCENARIO, seed=3)
    episode = gym.rollout(_good_policy())
    outcome = episode.info["outcome"]
    assert outcome["score"] == 1.0
    assert all(c["passed"] for c in outcome["checks"])
    assert episode.total_reward > 0.6  # outcome credit dominates
    gym.close()


def test_scenario_gym_pays_nothing_for_talk():
    gym = AgentGym.from_scenario(REFUND_SCENARIO, seed=3)
    gym.reset()
    out = gym.step({"answer": "Done! Order 7 refunded, all good."})  # it wasn't
    outcome = out.info["outcome"]
    assert outcome["score"] < 1.0  # state checks fail: world unchanged
    failed = [c for c in outcome["checks"] if not c["passed"]]
    assert any(c["name"] == "sql" for c in failed)
    assert out.reward < 0.5
    gym.close()


def test_outcome_state_resets_between_episodes():
    gym = AgentGym.from_scenario(REFUND_SCENARIO, seed=3)
    first = gym.rollout(_good_policy())
    assert first.info["outcome"]["score"] == 1.0
    gym.reset()  # fresh world: order 7 is 'paid' again
    out = gym.step({"answer": "already refunded earlier"})
    assert out.info["outcome"]["score"] < 1.0
    gym.close()


# --- REST scenarios ----------------------------------------------------------------


class _OrdersApi(BaseHTTPRequestHandler):
    state = {}

    def log_message(self, *args):
        pass

    def _send(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        order_id = self.path.rsplit("/", 1)[-1]
        self._send(self.state.get(order_id, {"error": "not found"}))

    def do_POST(self):
        if self.path.startswith("/refund/"):
            order_id = self.path.rsplit("/", 1)[-1]
            if order_id in self.state:
                self.state[order_id]["status"] = "refunded"
            self._send(self.state.get(order_id, {}))
        else:
            self._send({"error": "bad route"})


REST_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Orders", "version": "1"},
    "paths": {
        "/refund/{order_id}": {
            "post": {
                "operationId": "refund_order",
                "summary": "Refund one order.",
                "parameters": [
                    {
                        "name": "order_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"},
                    }
                ],
            }
        }
    },
}


def test_rest_scenario_checks_the_api_state():
    _OrdersApi.state = {"7": {"id": 7, "status": "paid"}}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OrdersApi)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address

    scenario = Scenario(
        id="rest-refund-7",
        task="Refund order 7 via the API.",
        environment={"type": "rest", "spec": REST_SPEC, "base_url": f"http://{host}:{port}"},
        checkers=[HttpCheck(path="/orders/7", contains='"refunded"')],
    )
    gym = AgentGym.from_scenario(scenario, seed=5)
    script = iter(
        [
            {"tool_name": "refund_order", "arguments": {"order_id": 7}},
            {"answer": "Refunded order 7 via the API."},
        ]
    )
    episode = gym.rollout(lambda obs, g: next(script))
    assert episode.info["outcome"]["score"] == 1.0
    gym.close()
    server.shutdown()


# --- packs + suites ------------------------------------------------------------------


def test_scenario_pack_round_trips_yaml_and_json(tmp_path):
    pytest.importorskip("yaml")
    for name in ("pack.yaml", "pack.json"):
        path = str(tmp_path / name)
        save_scenarios([REFUND_SCENARIO], path)
        loaded = load_scenarios(path)
        assert len(loaded) == 1
        back = loaded[0]
        assert back.id == REFUND_SCENARIO.id
        assert [type(c) for c in back.checkers] == [type(c) for c in REFUND_SCENARIO.checkers]
        gym = AgentGym.from_scenario(back, seed=3)  # and it still runs
        assert gym.rollout(_good_policy()).info["outcome"]["score"] == 1.0
        gym.close()


def test_run_scenario_suite_reports_pass_rate():
    impossible = REFUND_SCENARIO.model_copy(
        update={"id": "impossible", "checkers": [AnswerContains(any_of=["zzz-never"])]}
    )

    def lazy_policy(obs, gym):
        return {"answer": "refund handled"}

    report = run_scenario_suite(lazy_policy, [REFUND_SCENARIO, impossible], seed=3)
    assert report.n == 2
    assert report.passed == 0  # lazy talk passes neither world-state nor zzz check
    assert report.pass_rate == 0.0
    assert {r["id"] for r in report.results} == {"refund-7", "impossible"}
