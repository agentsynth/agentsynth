"""Proof v2: build a verified agent-trajectory dataset, reproducibly.

Runs offline against the deterministic mock by default — so it works anywhere, in CI, and
as a dry run — and switches to a real provider when you pass a model and set a key in the
environment. Every trajectory is verified (execution + tool-arg + safety) and near-
duplicates are dropped, so what lands on disk is the high-signal subset, not raw
generation. It writes JSONL + ShareGPT + a dataset card + a manifest you can publish.

    # offline dry run — a handful, proves the pipeline end to end
    python scripts/proof_v2.py --n 50 --out data/proof_v2

    # the real drop — set the key in the ENVIRONMENT, never on the command line
    export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY, GROQ_API_KEY, ...
    python scripts/proof_v2.py --n 10000 --model claude-haiku-4-5-20251001 \
        --budget-usd 30 --out data/proof_v2 --yes

`--budget-usd` is a volume cap: the run is trimmed so an estimate of the cost can't
exceed it. The estimate is rough — also set a spend limit in your provider console.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Rough blended $/1k tokens, and an assumed ~3k tokens per verified trajectory. These are
# deliberately conservative so the budget cap errs toward under-spending; tune for your run.
_PRICE_PER_1K = {
    "claude-haiku-4-5-20251001": 0.004,
    "gpt-4o-mini": 0.002,
    "groq": 0.001,
}
_TOKENS_PER_TRAJ = 3.0  # in thousands


def estimate_cost_usd(n: int, model) -> float:
    price = 0.004
    for key, value in _PRICE_PER_1K.items():
        if model and key in str(model):
            price = value
            break
    return round(n * _TOKENS_PER_TRAJ * price, 2)


def build_recipe(n: int, model, out_dir: str, seed: int):
    """A verified-dataset recipe: generate, verify, dedup, export JSONL."""
    from agentsynth import Recipe

    return Recipe(
        name="proof-v2",
        num_trajectories=n,
        vary_modes=True,
        model=model,
        use_mock=("auto" if model else True),
        seed=seed,
        evaluate=True,
        verify=True,
        dedup=True,
        export_format="jsonl",
        export_path=os.path.join(out_dir, "dataset.jsonl"),
        max_workers=8 if model else 1,
    )


def _write_card(out_dir: str, summary: dict) -> str:
    path = os.path.join(out_dir, "README.md")
    cmd = summary["command"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            f"# AgentSynth Proof v2 dataset\n\n"
            f"{summary['kept']} verified agent trajectories "
            f"({'mock' if summary['mock'] else summary['model']}), generated with "
            f"[AgentSynth](https://github.com/agentsynth/agentsynth).\n\n"
            f"- pass@1 (judge): {summary['metrics'].get('pass_rate', 'n/a')}\n"
            f"- verified rate: {summary['metrics'].get('verified_rate', 'n/a')}\n"
            f"- near-duplicates dropped: {summary['metrics'].get('duplicates_removed', 0)}\n"
            f"- seed: {summary['seed']}\n\n"
            f"Every trajectory was checked by execution / tool-arg / safety verifiers, and\n"
            f"near-duplicates were removed, so this is the high-signal subset.\n\n"
            f"## Reproduce\n\n```bash\n{cmd}\n```\n\n"
            f"Files: `dataset.jsonl` (round-trippable), `dataset.sharegpt.json` "
            f"(chat format), `manifest.json`.\n\nLicense: MIT.\n"
        )
    return path


def run_proof(n, model=None, out_dir="data/proof_v2", seed=0, budget_usd=None):
    """Build the dataset and write the JSONL, ShareGPT, card, and manifest."""
    from agentsynth import run_recipe, save_dataset

    capped = n
    if budget_usd is not None and model:
        per = estimate_cost_usd(1, model) or 1e-9
        capped = min(n, int(budget_usd / per))

    os.makedirs(out_dir, exist_ok=True)
    recipe = build_recipe(capped, model, out_dir, seed)
    result = run_recipe(recipe)

    sharegpt_path = os.path.join(out_dir, "dataset.sharegpt.json")
    save_dataset(result.trajectories, sharegpt_path, fmt="sharegpt")

    cmd = "python scripts/proof_v2.py --n {} {}--out {} --seed {}".format(
        capped, f"--model {model} " if model else "", out_dir, seed
    )
    summary = {
        "requested": n,
        "kept": len(result.trajectories),
        "model": model,
        "mock": model is None,
        "seed": seed,
        "budget_usd": budget_usd,
        "estimated_cost_usd": estimate_cost_usd(capped, model) if model else 0.0,
        "metrics": result.metrics,
        "jsonl": result.output_path,
        "sharegpt": sharegpt_path,
        "command": cmd,
    }
    summary["card"] = _write_card(out_dir, summary)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build a verified Proof v2 dataset.")
    parser.add_argument("--n", type=int, default=50, help="Target number of trajectories.")
    parser.add_argument("--model", default=None, help="Provider model id; omit for the mock.")
    parser.add_argument("--out", default="data/proof_v2", help="Output directory.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget-usd", dest="budget_usd", type=float, default=None)
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation on a real run.")
    args = parser.parse_args(argv)

    if args.model:
        est = estimate_cost_usd(args.n, args.model)
        print(f"Real run: ~{args.n} trajectories on {args.model}, rough estimate ${est}.")
        if args.budget_usd:
            print(f"Budget ${args.budget_usd} → capping volume to stay under it.")
        if not args.yes:
            print("Set a spend limit in your provider console, then re-run with --yes.")
            return 0

    summary = run_proof(
        args.n, model=args.model, out_dir=args.out, seed=args.seed, budget_usd=args.budget_usd
    )
    print(
        f"kept {summary['kept']}/{summary['requested']} verified trajectories -> {summary['jsonl']}"
    )
    print(f"card: {summary['card']}  manifest: {os.path.join(args.out, 'manifest.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
