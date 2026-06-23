"""Hub API tests — SQLite-backed, no network."""

import os
import sys
from pathlib import Path

os.environ["DATABASE_URL"] = "sqlite:///./test_hub.db"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import PACKS, app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app)
PACK_ID = "core_v1"
N = len(PACKS[PACK_ID])


def _report(pass_rate=0.5, passed=None, n=N):
    passed = int(round(pass_rate * n)) if passed is None else passed
    return {
        "n": n,
        "passed": passed,
        "pass_rate": pass_rate,
        "results": [{"id": f"s{i}", "passed": i < passed} for i in range(n)],
    }


def test_healthz_lists_packs():
    data = client.get("/healthz").json()
    assert data["status"] == "ok"
    assert PACK_ID in data["packs"]


def test_packs_endpoints():
    listing = client.get("/v1/packs").json()
    assert listing[PACK_ID]["scenarios"] == N
    pack = client.get(f"/v1/packs/{PACK_ID}").json()
    assert len(pack) == N and pack[0]["id"]
    assert client.get("/v1/packs/nope").status_code == 404


def test_submit_and_leaderboard():
    for model, rate in (("model-a", 0.8), ("model-b", 0.5), ("model-a", 0.6)):
        resp = client.post(
            "/v1/submissions",
            json={"pack_id": PACK_ID, "model": model, "report": _report(rate)},
        )
        assert resp.status_code == 201, resp.text

    board = client.get("/v1/leaderboard", params={"pack": PACK_ID}).json()
    models = [e["model"] for e in board["entries"]]
    assert models.index("model-a") < models.index("model-b")
    top = board["entries"][0]
    assert top["model"] == "model-a" and top["pass_rate"] == 0.8  # best run wins, not latest

    page = client.get("/leaderboard", params={"pack": PACK_ID}).text
    assert "model-a" in page and "80%" in page


def test_breakdown_ranks_hardest_scenarios_first():
    # two models, best run each: s0 fails for both, s1 fails for one, the rest pass
    for model, fail_ids in (("bd-model-a", {"s0", "s1"}), ("bd-model-b", {"s0"})):
        results = [{"id": f"s{i}", "passed": f"s{i}" not in fail_ids} for i in range(N)]
        passed = sum(1 for r in results if r["passed"])
        report = {"n": N, "passed": passed, "pass_rate": passed / N, "results": results}
        resp = client.post(
            "/v1/submissions", json={"pack_id": PACK_ID, "model": model, "report": report}
        )
        assert resp.status_code == 201, resp.text

    data = client.get(f"/v1/packs/{PACK_ID}/breakdown").json()
    assert data["models"] >= 2
    by_id = {s["id"]: s for s in data["scenarios"]}
    assert by_id["s0"]["pass_rate"] < by_id["s1"]["pass_rate"] < by_id["s2"]["pass_rate"]
    assert data["scenarios"][0]["pass_rate"] <= data["scenarios"][-1]["pass_rate"]

    assert client.get("/v1/packs/nope/breakdown").status_code == 404

    page = client.get("/leaderboard", params={"pack": PACK_ID}).text
    assert "Hardest scenarios" in page
    assert "s0" in page


def test_home_serves_the_landing():
    page = client.get("/").text
    assert "Agent training data" in page
    assert "--pack core_v1" in page  # the quickstart shows the current funnel


def test_og_image_and_leaderboard_chrome():
    img = client.get("/og.png")
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/png"

    page = client.get("/leaderboard").text
    assert "background:#fff" in page  # dark-mode browsers must not bleed through
    assert 'name="viewport"' in page


def test_submission_validation():
    bad_pack = client.post(
        "/v1/submissions", json={"pack_id": "nope", "model": "m", "report": _report()}
    )
    assert bad_pack.status_code == 404

    wrong_size = client.post(
        "/v1/submissions", json={"pack_id": PACK_ID, "model": "m", "report": _report(n=3)}
    )
    assert wrong_size.status_code == 422

    fake_numbers = client.post(
        "/v1/submissions",
        json={"pack_id": PACK_ID, "model": "m", "report": dict(_report(), pass_rate=3.0)},
    )
    assert fake_numbers.status_code == 422


def test_manifest_marks_a_submission_reproducible():
    rate = 0.6
    resp = client.post(
        "/v1/submissions",
        json={
            "pack_id": PACK_ID,
            "model": "verifiable-model",
            "report": _report(rate),
            "manifest": {"run_hash": "abc123def456", "pack_fingerprint": "fp00", "pass_rate": rate},
        },
    )
    assert resp.status_code == 201, resp.text
    board = client.get("/v1/leaderboard", params={"pack": PACK_ID}).json()
    entry = next(e for e in board["entries"] if e["model"] == "verifiable-model")
    assert entry["reproducible"] is True
    assert entry["run_hash"] == "abc123def456"


def test_manifest_disagreeing_with_its_report_is_rejected():
    resp = client.post(
        "/v1/submissions",
        json={
            "pack_id": PACK_ID,
            "model": "liar",
            "report": _report(0.9),
            "manifest": {"run_hash": "x", "pass_rate": 0.1},  # contradicts the report
        },
    )
    assert resp.status_code == 422
