"""Reproducible run manifests — make a leaderboard score something you can re-run.

A number on a leaderboard is only worth what you can independently reproduce. 2026's
contamination findings (frontier models reciting benchmark answers from the task id
alone) make that the whole game: the score that matters is the one anyone can re-derive.

A run manifest pins everything a bench run depended on — a content hash of the pack (so
the pack can't be quietly changed underneath the score), the policy, the seed, the trial
count, the library version, and the per-scenario outcomes — and folds them into a single
`run_hash`. Re-run the same pack with the same policy and seed and you get the same hash;
`verify_run` does exactly that and reports whether it reproduced.

    manifest = run_manifest("core_v2", scenarios, report, model="gpt-4o-mini", seed=7, trials=4)
    check = verify_run(manifest, scenarios, my_policy)   # {reproduced, pack_intact, ...}
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional


def content_hash(value: Any, length: int = 16) -> str:
    """A stable short hex digest of any JSON-able value (sorted, so order doesn't matter)."""
    blob = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


def pack_fingerprint(scenarios: List[Any]) -> str:
    """A content hash of the pack itself — its tasks, worlds, and checkers.

    Two packs with the same fingerprint are the same benchmark; a changed checker or seed
    row changes it, so a score can't be claimed against a pack that was edited afterwards.
    """
    # hash each scenario, then sort the digests — order-independent, and never compares
    # raw dicts (which aren't orderable).
    digests = sorted(content_hash(scenario.model_dump()) for scenario in scenarios)
    return content_hash(digests)


def _result_rows(report: Any) -> List[Dict[str, Any]]:
    rows = []
    for row in report.results:
        rows.append(
            {
                "id": row["id"],
                "passed": bool(row["passed"]),
                "outcome_score": round(float(row.get("outcome_score", 0.0)), 6),
            }
        )
    return sorted(rows, key=lambda r: r["id"])


def run_hash(pack_fp: str, model: str, seed: int, trials: int, rows: List[Dict[str, Any]]) -> str:
    """The verifiable fingerprint of a result: same inputs + same outcomes -> same hash."""
    return content_hash(
        {
            "pack": pack_fp,
            "policy": model,
            "seed": seed,
            "trials": trials,
            "rows": [(r["id"], r["passed"], r["outcome_score"]) for r in rows],
        }
    )


def run_manifest(
    pack_id: str,
    scenarios: List[Any],
    report: Any,
    model: str,
    seed: int,
    trials: int = 1,
    version: Optional[str] = None,
    cost: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Everything needed to reproduce and check a bench run.

    `cost` (calls/tokens/usd from a `CostMeter`, when the policy is a metered LLM
    client) rides along as telemetry but is deliberately NOT a `run_hash` input —
    reproducing a run means matching its outcomes, not its exact spend.
    """
    if version is None:
        from . import __version__ as version
    pack_fp = pack_fingerprint(scenarios)
    rows = _result_rows(report)
    manifest = {
        "pack_id": pack_id,
        "pack_fingerprint": pack_fp,
        "policy": model,
        "seed": seed,
        "trials": trials,
        "client_version": version,
        "n": report.n,
        "passed": report.passed,
        "pass_rate": report.pass_rate,
        "results": rows,
        "run_hash": run_hash(pack_fp, model, seed, trials, rows),
    }
    if cost:
        manifest["cost"] = cost
    return manifest


def verify_run(
    manifest: Dict[str, Any],
    scenarios: List[Any],
    policy: Any,
    tolerance: float = 0.0,
) -> Dict[str, Any]:
    """Re-run a manifest's bench and report whether it reproduced.

    `pack_intact` catches a pack edited after the fact (fingerprint mismatch). `reproduced`
    is an exact run_hash match — what you get from a deterministic policy. For a stochastic
    model, allow a `tolerance` on the pass-rate and read `pass_rate_delta` instead.
    """
    from .scenarios import run_scenario_suite

    pack_fp = pack_fingerprint(scenarios)
    pack_intact = pack_fp == manifest.get("pack_fingerprint")

    trials = int(manifest.get("trials", 1))
    seed = int(manifest.get("seed", 7))
    if trials > 1:
        from .cli import _run_trials

        report, _ = _run_trials(policy, scenarios, seed, trials)
    else:
        report = run_scenario_suite(policy, scenarios, seed=seed)

    rows = _result_rows(report)
    actual_hash = run_hash(pack_fp, manifest.get("policy", ""), seed, trials, rows)
    delta = round(abs(report.pass_rate - float(manifest.get("pass_rate", 0.0))), 6)
    return {
        "pack_intact": pack_intact,
        "reproduced": pack_intact and actual_hash == manifest.get("run_hash"),
        "within_tolerance": pack_intact and delta <= tolerance,
        "expected_hash": manifest.get("run_hash"),
        "actual_hash": actual_hash,
        "expected_pass_rate": manifest.get("pass_rate"),
        "actual_pass_rate": report.pass_rate,
        "pass_rate_delta": delta,
    }


def diff_runs(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """What changed between two bench manifests of the same pack.

    `regressed` is the list that should fail a CI gate: scenarios that passed
    before and fail now. Scenarios only present on one side land in `added` /
    `removed` rather than silently skewing the counts.
    """
    rows_a = {r["id"]: r for r in before.get("results", [])}
    rows_b = {r["id"]: r for r in after.get("results", [])}
    shared = sorted(set(rows_a) & set(rows_b))
    regressed = [i for i in shared if rows_a[i]["passed"] and not rows_b[i]["passed"]]
    fixed = [i for i in shared if not rows_a[i]["passed"] and rows_b[i]["passed"]]
    return {
        "same_pack": before.get("pack_fingerprint") == after.get("pack_fingerprint"),
        "pass_rate_before": before.get("pass_rate"),
        "pass_rate_after": after.get("pass_rate"),
        "regressed": regressed,
        "fixed": fixed,
        "still_failing": [i for i in shared if not rows_a[i]["passed"] and not rows_b[i]["passed"]],
        "added": sorted(set(rows_b) - set(rows_a)),
        "removed": sorted(set(rows_a) - set(rows_b)),
    }


__all__ = [
    "content_hash",
    "diff_runs",
    "pack_fingerprint",
    "run_hash",
    "run_manifest",
    "verify_run",
]
