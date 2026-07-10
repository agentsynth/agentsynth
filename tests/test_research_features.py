"""pass^k trials, failure detail, and pack teach."""

import json

import pytest

from agentsynth.cli import main as cli_main
from agentsynth.exporters import load_jsonl


@pytest.fixture()
def scaffold(tmp_path):
    assert cli_main(["pack", "new", "demo_v1", "--dir", str(tmp_path)]) == 0
    return tmp_path / "demo_v1.yaml", tmp_path / "demo_v1_oracle.py"


def test_trials_flag_scores_pass_k(scaffold, tmp_path, capsys):
    pack, oracle = scaffold
    # solves the pack on even seeds, talks on odd ones — flaky on purpose
    flaky = tmp_path / "flaky.py"
    flaky.write_text(
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('demo_oracle', r'{oracle}')\n"
        "demo = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(demo)\n"
        "\n"
        "def solve(observation, gym):\n"
        "    if gym.seed % 2 == 0:\n"
        "        return demo.solve(observation, gym)\n"
        "    return {'answer': 'all done'}\n",
        encoding="utf-8",
    )

    code = cli_main(
        [
            "bench",
            "--pack",
            str(pack),
            "--policy",
            f"{flaky}:solve",
            "--trials",
            "2",
            "--seed",
            "7",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "FLAKY" in out  # passes the even seed, fails the odd one
    assert "pass^2" in out and "pass^1" in out
    assert "0/3 (0%)" in out  # nothing survives both trials


def test_trials_all_pass_for_a_deterministic_oracle(scaffold, capsys):
    pack, oracle = scaffold
    code = cli_main(["bench", "--pack", str(pack), "--policy", f"{oracle}:solve", "--trials", "3"])
    out = capsys.readouterr().out
    assert code == 0
    assert "pass^3 (all trials must pass): 3/3" in out
    assert "FLAKY" not in out


def test_single_trial_names_the_failed_checkers(scaffold, capsys):
    pack, _ = scaffold
    code = cli_main(["bench", "--pack", str(pack), "--policy", "tests.bench_policy:lazy"])
    out = capsys.readouterr().out
    assert code == 0
    assert "failed:" in out
    assert "sql" in out


def test_pack_teach_exports_gold_trajectories(scaffold, tmp_path, capsys):
    pack, _ = scaffold
    out_path = tmp_path / "gold.jsonl"
    code = cli_main(["pack", "teach", str(pack), "--out", str(out_path)])
    stdout = capsys.readouterr().out
    assert code == 0
    assert "3 gold trajectories" in stdout

    rows = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 3
    assert all(row.get("verification") for row in rows)

    trajs = load_jsonl(str(out_path))  # round-trips into Trajectory objects
    assert {t.query[:10] for t in trajs} == {r["query"][:10] for r in rows}


def test_bench_json_report(scaffold, tmp_path, capsys):
    pack, oracle = scaffold
    out = tmp_path / "report.json"
    code = cli_main(
        [
            "bench",
            "--pack",
            str(pack),
            "--policy",
            f"{oracle}:solve",
            "--trials",
            "2",
            "--json",
            str(out),
        ]
    )
    assert code == 0
    report = json.loads(out.read_text())
    assert report["trials"] == 2
    assert report["pass1_avg"] == 1.0
    assert report["pass_rate"] == 1.0
    assert len(report["results"]) == 3
    assert report["pack_id"] == "demo_v1"


def test_bench_folds_a_metered_policys_spend_into_the_manifest(scaffold, tmp_path, capsys):
    # a policy carrying a pre-populated CostMeter, the shape _policy_from_model
    # attaches for a real LLM run — exercised here without any network call.
    pack, oracle = scaffold
    metered = tmp_path / "metered.py"
    metered.write_text(
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('demo_oracle', r'{oracle}')\n"
        "demo = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(demo)\n"
        "from agentsynth.scale import CostMeter\n"
        "\n"
        "def solve(observation, gym):\n"
        "    return demo.solve(observation, gym)\n"
        "\n"
        "solve.meter = CostMeter()\n"
        "solve.meter.add(120, 40, 0.0123)\n",
        encoding="utf-8",
    )
    out = tmp_path / "costed.json"
    code = cli_main(
        ["bench", "--pack", str(pack), "--policy", f"{metered}:solve", "--json", str(out)]
    )
    printed = capsys.readouterr().out
    assert code == 0
    assert "cost: $0.0123 (1 calls, 160 tokens)" in printed
    report = json.loads(out.read_text())
    assert report["cost"] == {
        "calls": 1,
        "prompt_tokens": 120,
        "completion_tokens": 40,
        "total_tokens": 160,
        "usd": 0.0123,
    }
    assert report["manifest"]["cost"] == report["cost"]


def test_bench_omits_cost_for_an_unmetered_policy(scaffold, capsys):
    pack, oracle = scaffold
    code = cli_main(["bench", "--pack", str(pack), "--policy", f"{oracle}:solve"])
    printed = capsys.readouterr().out
    assert code == 0
    assert "cost:" not in printed


def test_compare_runs_items_side_by_side(scaffold, tmp_path, capsys):
    pack, oracle = scaffold
    out = tmp_path / "cmp.json"
    code = cli_main(
        [
            "bench",
            "--pack",
            str(pack),
            "--compare",
            f"{oracle}:solve,tests.bench_policy:lazy",
            "--trials",
            "2",
            "--json",
            str(out),
        ]
    )
    stdout = capsys.readouterr().out
    assert code == 0
    assert "✓" in stdout and "✗" in stdout
    assert "pass^2" in stdout and "100%" in stdout and "0%" in stdout

    blob = json.loads(out.read_text())
    assert [run["pass_rate"] for run in blob["compare"]] == [1.0, 0.0]
    assert blob["compare"][1]["name"] == "tests.bench_policy:lazy"


def test_compare_submits_each_item_under_its_name(scaffold, monkeypatch):
    from agentsynth import cli as climod

    pack, oracle = scaffold
    posted = []
    monkeypatch.setattr(
        climod, "_post_json", lambda url, payload: posted.append(payload) or '{"id": 1}'
    )
    code = cli_main(
        [
            "bench",
            "--pack",
            str(pack),
            "--compare",
            f"{oracle}:solve,tests.bench_policy:lazy",
            "--submit",
            "--hub",
            "https://hub.example",
        ]
    )
    assert code == 0
    assert [p["model"] for p in posted] == [f"{oracle}:solve", "tests.bench_policy:lazy"]
    assert posted[0]["report"]["pass_rate"] == 1.0


def test_compare_guards(scaffold):
    pack, oracle = scaffold
    with pytest.raises(SystemExit):
        cli_main(["bench", "--pack", str(pack), "--compare", f"{oracle}:solve"])
    with pytest.raises(SystemExit):
        cli_main(
            [
                "bench",
                "--pack",
                str(pack),
                "--compare",
                f"{oracle}:solve,tests.bench_policy:lazy",
                "--policy",
                "tests.bench_policy:lazy",
            ]
        )


def test_pack_teach_rejects_an_imperfect_oracle(scaffold, tmp_path, capsys):
    pack, oracle = scaffold
    oracle.write_text(
        "def solve(observation, gym):\n    return {'answer': 'no idea'}\n", encoding="utf-8"
    )
    code = cli_main(["pack", "teach", str(pack), "--out", str(tmp_path / "g.jsonl")])
    out = capsys.readouterr().out
    assert code == 1
    assert "gold data has to pass every checker" in out
