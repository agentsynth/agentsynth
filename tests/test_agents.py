"""External agents over stdio/HTTP, and the run-diff that gates CI on them."""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agentsynth.agents import agent_policy, http_policy, step_payload, subprocess_policy
from agentsynth.cli import main as cli_main
from agentsynth.provenance import diff_runs, run_manifest
from agentsynth.scenarios import load_scenarios, run_scenario_suite

REFUND = load_scenarios("packs/core_v1.yaml")[:1]  # refund-order: update row 7, say "refund"

# An agent that actually solves refund-order, driven only by the step number.
SOLVER = """\
import json, sys
for line in sys.stdin:
    req = json.loads(line)
    if req["step"] == 1:
        act = {"tool": "sql_query",
               "args": {"query": "UPDATE orders SET status='refunded' WHERE id=7"}}
    else:
        act = {"answer": "refund issued for order 7"}
    print(json.dumps(act), flush=True)
"""


def _write_agent(tmp_path, body, name="agent.py"):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return f"{sys.executable} {path}"


def test_step_payload_carries_the_whole_picture():
    from agentsynth.rl import AgentGym

    gym = AgentGym.from_scenario(REFUND[0])
    gym.reset()
    payload = step_payload("the task", gym)
    assert payload["task"] == REFUND[0].task
    assert payload["step"] == 1 and payload["max_steps"] == REFUND[0].max_steps
    assert any(t["name"] == "sql_query" for t in payload["tools"])
    assert "parameters" in payload["tools"][0]


def test_subprocess_agent_passes_the_scenario(tmp_path):
    policy = subprocess_policy(_write_agent(tmp_path, SOLVER))
    try:
        report = run_scenario_suite(policy, REFUND, seed=7)
    finally:
        policy.close()
    assert report.pass_rate == 1.0


def test_one_shot_agent_is_respawned_each_step(tmp_path):
    one_shot = SOLVER.replace("for line in sys.stdin:", "for line in [sys.stdin.readline()]:")
    policy = subprocess_policy(_write_agent(tmp_path, one_shot))
    try:
        report = run_scenario_suite(policy, REFUND, seed=7)
    finally:
        policy.close()
    assert report.pass_rate == 1.0  # the harness restarted it between steps


def test_agent_that_dies_raises_a_clear_error(tmp_path):
    policy = subprocess_policy(_write_agent(tmp_path, "import sys; sys.exit(3)\n"))
    try:
        with pytest.raises(RuntimeError, match="without replying|closed stdin"):
            run_scenario_suite(policy, REFUND, seed=7)
    finally:
        policy.close()


@pytest.mark.skipif(sys.platform == "win32", reason="reply timeout needs select() on pipes")
def test_silent_agent_times_out(tmp_path):
    policy = subprocess_policy(_write_agent(tmp_path, "import time\ntime.sleep(30)\n"), timeout=0.5)
    try:
        with pytest.raises(RuntimeError, match="sent nothing"):
            run_scenario_suite(policy, REFUND, seed=7)
    finally:
        policy.close()


def test_unparsable_reply_ends_the_episode_as_an_answer(tmp_path):
    chatty = 'import sys\nsys.stdin.readline()\nprint("no json here", flush=True)\n'
    policy = subprocess_policy(_write_agent(tmp_path, chatty))
    try:
        report = run_scenario_suite(policy, REFUND, seed=7)
    finally:
        policy.close()
    assert report.pass_rate == 0.0  # graded, not crashed


def _serve(handler_logic):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length))
            body = json.dumps(handler_logic(request)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_http_agent_passes_the_scenario():
    def logic(req):
        if req["step"] == 1:
            return {
                "tool": "sql_query",
                "args": {"query": "UPDATE orders SET status='refunded' WHERE id=7"},
            }
        return {"answer": "refund issued"}

    server = _serve(logic)
    try:
        policy = http_policy(f"http://127.0.0.1:{server.server_port}/act")
        report = run_scenario_suite(policy, REFUND, seed=7)
    finally:
        server.shutdown()
    assert report.pass_rate == 1.0


def test_unreachable_endpoint_raises():
    policy = http_policy("http://127.0.0.1:1/act", timeout=1.0)
    with pytest.raises(RuntimeError, match="unreachable"):
        run_scenario_suite(policy, REFUND, seed=7)


def test_agent_policy_picks_the_transport():
    from agentsynth.agents import _HTTPAgent, _SubprocessAgent

    assert isinstance(agent_policy("http://x/act"), _HTTPAgent)
    assert isinstance(agent_policy("python agent.py"), _SubprocessAgent)


def test_bench_agent_end_to_end(tmp_path, capsys):
    out = tmp_path / "run.json"
    code = cli_main(
        [
            "bench",
            "--pack",
            "packs/core_v1.yaml",
            "--agent",
            _write_agent(tmp_path, SOLVER),
            "--json",
            str(out),
        ]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "[pass] refund-order" in text
    blob = json.loads(out.read_text())
    assert blob["manifest"]["policy"].endswith("agent.py")


# -- diff -----------------------------------------------------------------------


def _manifest(results):
    class Report:
        n = len(results)
        passed = sum(1 for r in results if r["passed"])
        pass_rate = round(passed / max(1, len(results)), 4)

    Report.results = results  # type: ignore[attr-defined]
    return run_manifest("demo", REFUND, Report, model="m", seed=7)


def test_diff_runs_finds_regressions_and_fixes():
    before = _manifest(
        [
            {"id": "a", "passed": True, "outcome_score": 1.0},
            {"id": "b", "passed": False, "outcome_score": 0.0},
            {"id": "gone", "passed": True, "outcome_score": 1.0},
        ]
    )
    after = _manifest(
        [
            {"id": "a", "passed": False, "outcome_score": 0.5},
            {"id": "b", "passed": True, "outcome_score": 1.0},
            {"id": "new", "passed": True, "outcome_score": 1.0},
        ]
    )
    delta = diff_runs(before, after)
    assert delta["regressed"] == ["a"] and delta["fixed"] == ["b"]
    assert delta["added"] == ["new"] and delta["removed"] == ["gone"]
    assert delta["same_pack"] is True


def test_diff_cli_gates_on_regressions(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(_manifest([{"id": "s1", "passed": True, "outcome_score": 1.0}])))
    b.write_text(json.dumps(_manifest([{"id": "s1", "passed": False, "outcome_score": 0.0}])))

    assert cli_main(["diff", str(a), str(b)]) == 1  # regression -> CI fails
    assert "- s1" in capsys.readouterr().out
    assert cli_main(["diff", str(a), str(b), "--ok"]) == 0  # report-only mode
    assert cli_main(["diff", str(b), str(a)]) == 0  # the fix direction is clean
