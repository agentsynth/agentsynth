"""Query evolution, run reports, the Docker sandbox, and the new CLI commands."""

import json

import pytest

from agentsynth import Recipe, evolve_queries, run_recipe, run_report_md
from agentsynth.cli import main as cli_main
from agentsynth.environments import DockerSandbox

QUERIES = ["total revenue by region", "refund order 7"]


# --- evolve_queries -------------------------------------------------------------


def test_evolve_is_deterministic_and_bounded():
    a = evolve_queries(QUERIES, k=8, seed=3)
    b = evolve_queries(QUERIES, k=8, seed=3)
    assert a == b
    assert len(a) == 8 == len(set(a))
    assert all(any(q in v for q in QUERIES) for v in a)


def test_evolve_uses_an_llm_when_given_one():
    class FakeClient:
        available = True

        def complete(self, messages):
            return "Paraphrased: " + messages[0]["content"].rsplit("Task: ", 1)[-1]

    out = evolve_queries(QUERIES, k=6, seed=3, llm_client=FakeClient())
    assert out[:2] == QUERIES  # round 0 keeps the originals
    assert any(v.startswith("Paraphrased:") for v in out[2:])


def test_evolve_falls_back_when_the_llm_returns_nothing():
    class DeadClient:
        available = True

        def complete(self, messages):
            return ""

    out = evolve_queries(QUERIES, k=6, seed=3, llm_client=DeadClient())
    assert len(out) == 6  # templates filled the gap


# --- run_report_md ----------------------------------------------------------------


def test_run_report_reads_like_a_summary():
    result = run_recipe(Recipe(queries=QUERIES, vary_modes=True, verify=True))
    report = run_report_md(result)
    assert "2 trajectories" in report
    assert "verified_rate" in report
    assert "modes:" in report

    class FakeMeter:
        def report(self):
            return {"calls": 3, "total_tokens": 360, "usd": 0.004}

    assert "$0.004" in run_report_md(result, meter=FakeMeter())


# --- DockerSandbox -----------------------------------------------------------------


def test_docker_sandbox_surface():
    box = DockerSandbox()
    assert box.tool_names() == ["python"]
    assert "code" in box.sample_args("python", "compute 2 + 2", seed=1)
    assert box.execute("python", {"code": ""}) == ""
    with pytest.raises(KeyError):
        box.execute("nope", {})


def test_docker_missing_binary_is_a_clean_observation(monkeypatch):
    import subprocess

    def boom(*args, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(subprocess, "run", boom)
    out = DockerSandbox().execute("python", {"code": "print(1)"})
    assert out.startswith("DockerError")


@pytest.mark.skipif(not DockerSandbox.available(), reason="docker not installed")
def test_docker_runs_code_for_real():
    out = DockerSandbox(timeout=120).execute("python", {"code": "print(40 + 2)"})
    assert out.strip() == "42" or out.startswith("DockerError")


# --- CLI ---------------------------------------------------------------------------


def test_cli_import_command(tmp_path):
    trace = [
        {"role": "user", "content": "weather in Hanoi?"},
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city": "Hanoi"}'}}],
        },
        {"role": "tool", "content": "31C"},
        {"role": "assistant", "content": "31C in Hanoi."},
    ]
    traces = tmp_path / "traces.jsonl"
    traces.write_text(json.dumps({"messages": trace}) + "\n", encoding="utf-8")
    out = tmp_path / "imported.jsonl"

    assert cli_main(["import", "--in", str(traces), "--out", str(out)]) == 0
    row = json.loads(out.read_text().splitlines()[0])
    assert row["query"] == "weather in Hanoi?"


def test_cli_flywheel_command(tmp_path):
    from agentsynth import AgentTrajectoryGenerator, to_jsonl

    trajs = AgentTrajectoryGenerator(use_mock=True).generate_batch(
        "analyze sales by region", num_trajectories=4, vary_modes=True
    )
    data = tmp_path / "data.jsonl"
    to_jsonl(trajs, str(data))
    out = tmp_path / "patch.jsonl"

    code = cli_main(
        ["flywheel", "--in", str(data), "--out", str(out), "--k", "4", "--threshold", "0.99"]
    )
    assert code == 0
    assert len(out.read_text().splitlines()) == 4


def test_cli_flywheel_with_nothing_to_fix(tmp_path):
    from agentsynth import AgentTrajectoryGenerator, to_jsonl

    trajs = AgentTrajectoryGenerator(use_mock=True).generate_batch(
        "analyze sales", num_trajectories=2
    )
    data = tmp_path / "data.jsonl"
    to_jsonl(trajs, str(data))
    code = cli_main(["flywheel", "--in", str(data), "--threshold", "0.0"])
    assert code == 0  # nothing under the threshold, exits cleanly


def test_cli_import_missing_file():
    with pytest.raises(SystemExit):
        cli_main(["import", "--in", "/nonexistent.jsonl"])


def test_cli_bench_with_custom_policy(capsys):
    code = cli_main(
        ["bench", "--pack", "packs/core_v1.yaml", "--policy", "tests.bench_policy:lazy"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "scenarios passed" in out
    assert "FAIL" in out  # talk alone fails the state checks


def test_cli_bench_requires_model_or_policy():
    with pytest.raises(SystemExit):
        cli_main(["bench", "--pack", "packs/core_v1.yaml"])
