"""Every number in the technical report, regenerated from the installed package.

    python paper/experiments.py

Runs offline (scripted policies, seeded worlds, no API keys), writes
paper/numbers.json, and prints the tables it was asked for. If a number in the
paper disagrees with this script, the script wins.
"""

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentsynth.contamination import canary_for, contamination_report, held_out_pack
from agentsynth.demo import DEMO_POLICIES
from agentsynth.provenance import pack_fingerprint, run_manifest, verify_run
from agentsynth.reliability import reliability_report
from agentsynth.robustness import audit_pack
from agentsynth.scenarios import load_scenarios, run_scenario_suite

PACKS = {
    "core_v1": ("packs/core_v1.yaml", "examples.core_v1_oracle:solve"),
    "core_v2": ("packs/core_v2.yaml", "packs.core_v2_oracle:solve"),
    "policy_v1": ("packs/policy_v1.yaml", "packs.policy_v1_oracle:solve"),
    "code_v1": ("packs/code_v1.yaml", "packs.code_v1_oracle:solve"),
}

SEED = 7


def _load_policy(ref):
    module_name, _, attr = ref.partition(":")
    module = __import__(module_name, fromlist=[attr])
    return getattr(module, attr)


def exp_audit():
    """Table 1 — knowledge-free adversaries against every pack."""
    out = {}
    for name, (path, _) in PACKS.items():
        report = audit_pack(load_scenarios(path), seed=SEED)
        out[name] = {
            "n": report.n,
            "robust": report.robust,
            "robustness": report.robustness_score,
            "adversaries": report.adversaries,
            "gamed": {r.scenario_id: r.gamed_by for r in report.rows if r.gamed_by},
            "leaks": {r.scenario_id: r.answer_leaks for r in report.rows if r.answer_leaks},
        }
    return out


def exp_bench():
    """Table 2 — oracle / generic expert / read-only / do-nothing, per pack."""
    out = {}
    for name, (path, oracle_ref) in PACKS.items():
        scenarios = load_scenarios(path)
        rows = {}
        rows["oracle"] = run_scenario_suite(_load_policy(oracle_ref), scenarios, seed=SEED)
        for label, policy in DEMO_POLICIES.items():
            key = label.split(" ")[0]  # expert / read-only / lazy
            rows[key] = run_scenario_suite(policy, scenarios, seed=SEED)
        out[name] = {k: {"pass_rate": r.pass_rate, "n": r.n} for k, r in rows.items()}
    return out


def exp_reliability():
    """Table 3 — pass^1 vs pass^k for a deliberately flaky solver on core_v1."""
    scenarios = load_scenarios(PACKS["core_v1"][0])
    oracle = _load_policy(PACKS["core_v1"][1])

    def flaky_factory(episode_seed):
        # Solves like the oracle, except a seeded coin makes it quit early
        # ~25% of the time — the shape of a stochastic model, without one.
        import random

        rng = random.Random(episode_seed)

        def policy(observation, gym):
            if gym.step_count == 0 and rng.random() < 0.25:
                return {"answer": "done (not really)"}
            return oracle(observation, gym)

        return policy

    trials = 4
    order = [s.id for s in scenarios]
    wins = {sid: [] for sid in order}
    for t in range(trials):
        report = run_scenario_suite(flaky_factory(1000 + t), scenarios, seed=SEED + t)
        for row in report.results:
            wins[row["id"]].append(bool(row["passed"]))
    rel = reliability_report(wins, trials)
    return {
        "trials": trials,
        "pass1": rel.pass1,
        "pass1_ci": list(rel.pass1_ci),
        "passk": rel.passk,
        "passk_ci": list(rel.passk_ci),
        "flaky": [f"{s.id} ({s.passes}/{s.trials})" for s in rel.flaky],
        "curve": [round(v, 4) for v in rel.curve],
    }


def exp_provenance():
    """Table 4 — a manifest reproduces; a tampered pack is caught."""
    scenarios = load_scenarios(PACKS["core_v1"][0])
    oracle = _load_policy(PACKS["core_v1"][1])
    report = run_scenario_suite(oracle, scenarios, seed=SEED)
    manifest = run_manifest("core_v1", scenarios, report, model="oracle", seed=SEED)

    honest = verify_run(manifest, scenarios, oracle)

    tampered = [s.model_copy(deep=True) for s in scenarios]
    tampered[0].task = tampered[0].task + " (edited after the run)"
    forged = verify_run(manifest, tampered, oracle)

    return {
        "run_hash": manifest["run_hash"],
        "pack_fingerprint": manifest["pack_fingerprint"],
        "honest": {k: honest[k] for k in ("pack_intact", "reproduced", "pass_rate_delta")},
        "tampered": {k: forged[k] for k in ("pack_intact", "reproduced")},
        "tampered_fingerprint": pack_fingerprint(tampered),
    }


def _adaptive(observation, gym):
    """Reads the world and works from the task's meaning — no memorized labels.

    Covers the first two core_v1 tasks (a refund by id, a read-only refusal).
    The point: it scores the same on a scenario and on its held-out sibling,
    where an answer-key replay collapses.
    """
    import re

    task = gym.task.lower()
    if "refund" in task:
        match = re.search(r"order (\d+)", task)
        oid = match.group(1) if match else "7"
        if gym.step_count == 0:
            return {
                "tool": "sql_query",
                "args": {"query": f"UPDATE orders SET status='refunded' WHERE id={oid}"},
            }
        return {"answer": f"refund issued: order {oid} is now refunded"}
    # refusal: read the row's actual status and echo the world back
    if gym.step_count == 0:
        return {"tool": "sql_query", "args": {"query": "SELECT status FROM orders WHERE id=9"}}
    lines = observation.splitlines()  # "status\n<value>\n(1 row)"
    status = lines[1].strip() if len(lines) > 1 else ""
    return {"answer": f"cannot cancel: order 9 is already {status}"}


def exp_contamination():
    """Table 5 — canaries, corpus overlap, and what held-out siblings catch."""
    scenarios = load_scenarios(PACKS["core_v1"][0])
    oracle = _load_policy(PACKS["core_v1"][1])
    leaked_corpus = [scenarios[0].task, "unrelated text about croissants"]
    report = contamination_report(scenarios, corpus=leaked_corpus, threshold=0.8)
    leaked_row = next(r for r in report.rows if r.id == scenarios[0].id)
    clean_row = next(r for r in report.rows if r.id != scenarios[0].id)

    siblings = held_out_pack(scenarios, seed=0)
    # An answer-key replay (the id-keyed oracle) against original vs sibling:
    replay_orig = run_scenario_suite(oracle, scenarios, seed=SEED).pass_rate
    replay_sib = run_scenario_suite(oracle, siblings, seed=SEED).pass_rate
    # An adaptive agent on the two tasks it covers, original vs sibling:
    subset, sib_subset = scenarios[:2], siblings[:2]
    adaptive_orig = run_scenario_suite(_adaptive, subset, seed=SEED).pass_rate
    adaptive_sib = run_scenario_suite(_adaptive, sib_subset, seed=SEED).pass_rate

    return {
        "canary_example": canary_for(scenarios[0].id),
        "leaked_scenario_overlap": leaked_row.max_overlap,
        "clean_scenario_overlap": clean_row.max_overlap,
        "flagged": report.flagged,
        "n": report.n,
        "siblings": {
            "world_relabelled": siblings[0].environment != scenarios[0].environment,
            "replay_pass_original": replay_orig,
            "replay_pass_sibling": replay_sib,
            "adaptive_pass_original": adaptive_orig,
            "adaptive_pass_sibling": adaptive_sib,
            "adaptive_n": len(subset),
        },
    }


def exp_transport_overhead():
    """Section 5 — what the stdio protocol costs per episode vs in-process."""
    from agentsynth.agents import subprocess_policy

    scenarios = load_scenarios(PACKS["core_v1"][0])[:5]

    def in_process(observation, gym):
        return {"answer": "done"}

    echo_agent = (
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    json.loads(line)\n"
        '    print(json.dumps({"answer": "done"}), flush=True)\n'
    )
    agent_path = os.path.join(os.path.dirname(__file__), "_echo_agent.py")
    with open(agent_path, "w", encoding="utf-8") as fh:
        fh.write(echo_agent)

    def timed(policy, reps=3):
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter()
            run_scenario_suite(policy, scenarios, seed=SEED)
            samples.append(time.perf_counter() - t0)
        return statistics.median(samples)

    try:
        base = timed(in_process)
        proto = timed(subprocess_policy(f"{sys.executable} {agent_path}"))
    finally:
        os.remove(agent_path)
    per_episode_ms = (proto - base) / len(scenarios) * 1000
    return {
        "episodes": len(scenarios),
        "in_process_s": round(base, 3),
        "stdio_s": round(proto, 3),
        "overhead_ms_per_episode": round(per_episode_ms, 1),
        "note": "median of 3; includes one process spawn per suite",
    }


def main():
    os.environ.setdefault("AGENTSYNTH_FORCE_MOCK", "1")
    # Everything in numbers.json is seeded and deterministic — CI regenerates it
    # and diffs against the committed copy. Wall-clock stays out of the file.
    numbers = {
        "audit": exp_audit(),
        "bench": exp_bench(),
        "reliability": exp_reliability(),
        "provenance": exp_provenance(),
        "contamination": exp_contamination(),
    }
    out = os.path.join(os.path.dirname(__file__), "numbers.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(numbers, fh, indent=2)
        fh.write("\n")
    print(json.dumps(numbers, indent=2))
    print(f"\n-> {out}")
    print("\ntransport overhead (machine-dependent, not committed):")
    print(json.dumps(exp_transport_overhead(), indent=2))


if __name__ == "__main__":
    main()
