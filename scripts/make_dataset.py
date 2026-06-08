#!/usr/bin/env python
"""Generate a verified AgentSynth dataset, and optionally push it to the Hub.

Runs on CPU. With no provider key it uses the deterministic mock generator.

    python scripts/make_dataset.py --n 500 --vary-modes --verify --dedup --out data
    python scripts/make_dataset.py --n 500 --push agentsynth/my-dataset --token $HF_TOKEN
"""

from __future__ import annotations

import argparse
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a verified AgentSynth dataset.")
    parser.add_argument("--n", type=int, default=200, help="Number of trajectories.")
    parser.add_argument("--query", default=None, help="A single query (else sample the taxonomy).")
    parser.add_argument("--domains", nargs="*", default=None, help="Restrict taxonomy domains.")
    parser.add_argument("--mode", default="single_agent")
    parser.add_argument("--vary-modes", action="store_true")
    parser.add_argument("--rubric", default="balanced", help="balanced/strict/lenient/safety_first")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--dedup", action="store_true")
    parser.add_argument("--model", default=None, help="LLM id (else mock).")
    parser.add_argument("--out", default="data", help="Output dataset directory.")
    parser.add_argument(
        "--push", default=None, metavar="REPO_ID", help="Push to this HF dataset repo."
    )
    parser.add_argument(
        "--token", default=None, help="HF write token (or use huggingface-cli login)."
    )
    args = parser.parse_args(argv)

    from agentsynth import Recipe, run_recipe
    from agentsynth.hub import prepare_dataset_dir

    recipe = Recipe(
        query=args.query,
        domains=args.domains,
        num_trajectories=args.n,
        mode=args.mode,
        vary_modes=args.vary_modes,
        rubric=args.rubric,
        verify=args.verify,
        dedup=args.dedup,
        model=args.model,
    )
    result = run_recipe(recipe)
    repo_id = args.push or "agentsynth/agentsynth-trajectories"
    out = prepare_dataset_dir(
        result.trajectories, args.out, eval_results=result.eval_results, repo_id=repo_id
    )
    print(
        f"Wrote {len(result.trajectories)} trajectories to {out}/ "
        f"(pass@1={result.metrics.get('pass_rate')}, "
        f"verified_rate={result.metrics.get('verified_rate')}, "
        f"duplicates_removed={result.duplicates_removed})"
    )

    if args.push:
        from agentsynth.hub import push_dataset

        url = push_dataset(
            result.trajectories, args.push, token=args.token, eval_results=result.eval_results
        )
        print(f"Pushed to {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
