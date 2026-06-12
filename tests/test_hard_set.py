"""The hard-set script: hub breakdown in, targeted verified dataset out."""

import json
import os
import subprocess
import sys

import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "hard_set.py")

pytestmark = pytest.mark.skipif(
    not os.path.exists(SCRIPT), reason="scripts/ not present in this checkout"
)


def _run(args, cwd):
    env = dict(os.environ, AGENTSYNTH_FORCE_MOCK="1", PYTHONPATH=os.path.dirname(SCRIPT) + "/..")
    return subprocess.run(
        [sys.executable, SCRIPT, *args], cwd=cwd, env=env, capture_output=True, text=True
    )


def _breakdown(tmp_path, scenarios):
    path = tmp_path / "breakdown.json"
    path.write_text(json.dumps({"pack": "core_v1", "models": 3, "scenarios": scenarios}))
    return path


def test_builds_a_dataset_from_the_hard_scenarios(tmp_path):
    bd = _breakdown(
        tmp_path,
        [
            {"id": "merge-duplicate-contacts", "attempts": 3, "passes": 0, "pass_rate": 0.0},
            {"id": "payroll-sanity-check", "attempts": 3, "passes": 1, "pass_rate": 0.3333},
            {"id": "refund-order", "attempts": 3, "passes": 3, "pass_rate": 1.0},
        ],
    )
    out = tmp_path / "hard.jsonl"
    pack = os.path.abspath("packs/core_v1.yaml")

    proc = _run(
        ["--breakdown", str(bd), "--pack-file", pack, "--k", "8", "--out", str(out)],
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert "merge-duplicate-contacts" in proc.stdout
    assert "refund-order" not in proc.stdout.splitlines()[0]  # passing scenario not targeted

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == 8
    assert any("m.jones" in row["query"] for row in rows)  # variants stay anchored to the task


def test_exits_nonzero_when_nothing_is_hard(tmp_path):
    bd = _breakdown(
        tmp_path, [{"id": "refund-order", "attempts": 2, "passes": 2, "pass_rate": 1.0}]
    )
    proc = _run(
        ["--breakdown", str(bd), "--pack-file", os.path.abspath("packs/core_v1.yaml")],
        cwd=tmp_path,
    )
    assert proc.returncode == 1
    assert "nothing under the threshold" in proc.stdout
