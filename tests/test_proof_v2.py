"""The Proof v2 dataset runner (offline / mock path)."""

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_proof():
    path = REPO / "scripts" / "proof_v2.py"
    spec = importlib.util.spec_from_file_location("proof_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_proof_builds_a_verified_dataset(tmp_path):
    proof = _load_proof()
    summary = proof.run_proof(n=8, out_dir=str(tmp_path), seed=0)

    assert summary["mock"] is True
    assert summary["kept"] > 0
    # the four artifacts are written
    for name in ("dataset.jsonl", "dataset.sharegpt.json", "README.md", "manifest.json"):
        assert (tmp_path / name).exists(), name
    # the manifest round-trips and carries the metrics
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["kept"] == summary["kept"]
    assert "pass_rate" in manifest["metrics"]
    # the card documents how to reproduce
    assert "Reproduce" in (tmp_path / "README.md").read_text()


def test_budget_caps_volume_on_a_real_run():
    proof = _load_proof()
    # a real run is gated by --yes, but the cost estimate + cap are pure functions
    est = proof.estimate_cost_usd(1000, "claude-haiku-4-5-20251001")
    assert est > 0
    # $1 budget at ~$0.004/traj should cap to a few hundred, well under 10k
    per = proof.estimate_cost_usd(1, "claude-haiku-4-5-20251001")
    assert int(1.0 / per) < 10000


def test_main_mock_run_exits_zero(tmp_path):
    proof = _load_proof()
    code = proof.main(["--n", "5", "--out", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "dataset.jsonl").exists()
