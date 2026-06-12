"""Build a training set aimed at what breaks models on a pack.

Reads the hub's per-scenario breakdown, keeps the scenarios models fail most,
expands their tasks into query variants, and runs a verified generation pass:

    python scripts/hard_set.py --pack core_v1 --out hard_set.jsonl

Offline run (no hub): point --breakdown and --pack-file at local JSON/YAML.
The generator is the offline mock unless a provider key is set.
"""

from __future__ import annotations

import argparse
import json
import sys
from urllib import request

from agentsynth import Recipe, evolve_queries, run_recipe, to_jsonl
from agentsynth.scenarios import load_scenarios


def _fetch_json(url: str):
    req = request.Request(url, headers={"User-Agent": "agentsynth-hard-set"})
    with request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", default="core_v1", help="Pack id on the hub (default: core_v1).")
    ap.add_argument("--hub", default="https://api.agentsynth.tech", metavar="URL")
    ap.add_argument(
        "--breakdown", default=None, metavar="PATH", help="Local breakdown JSON instead of the hub."
    )
    ap.add_argument(
        "--pack-file", default=None, metavar="PATH", help="Local pack file instead of the hub."
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Keep scenarios with a cross-model pass rate at or under this (default: 0.5).",
    )
    ap.add_argument("--k", type=int, default=40, help="Queries after expansion (default: 40).")
    ap.add_argument("--out", default="hard_set.jsonl", metavar="PATH")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.breakdown:
        with open(args.breakdown, encoding="utf-8") as fh:
            breakdown = json.load(fh)
    else:
        breakdown = _fetch_json(f"{args.hub.rstrip('/')}/v1/packs/{args.pack}/breakdown")

    hard_ids = [
        s["id"]
        for s in breakdown.get("scenarios", [])
        if s.get("pass_rate") is not None and s["pass_rate"] <= args.threshold
    ]
    if not hard_ids:
        print("nothing under the threshold — models are passing everything; raise --threshold")
        return 1

    if args.pack_file:
        scenarios = load_scenarios(args.pack_file)
    else:
        payload = _fetch_json(f"{args.hub.rstrip('/')}/v1/packs/{args.pack}")
        from agentsynth.scenarios import Scenario

        scenarios = [Scenario(**item) for item in payload]

    by_id = {s.id: s for s in scenarios}
    seeds = [by_id[sid].task for sid in hard_ids if sid in by_id]
    if not seeds:
        print("breakdown ids don't match the pack — wrong pack file?")
        return 1

    print(f"{len(seeds)} hard scenario(s): {', '.join(hard_ids)}")
    queries = evolve_queries(seeds, k=args.k, seed=args.seed)
    result = run_recipe(Recipe(queries=queries, vary_modes=True, verify=True, seed=args.seed))

    to_jsonl(result.trajectories, args.out)
    rate = result.metrics.get("verified_rate")
    print(f"{len(result.trajectories)} trajectories (verified_rate={rate}) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
