#!/usr/bin/env python
"""GRPO-train a model with AgentSynth's verified reward as the signal.

A real run needs a GPU; --dry-run wires everything up on CPU (environment, reward
function, prompts) and scores two canned completions so you can validate the setup
before spending compute.

    python scripts/train_grpo.py --dry-run
    python scripts/train_grpo.py --model unsloth/Llama-3.2-1B --steps 30

Install: pip install "agentsynth-ai[train]"  (plus `unsloth` for the 4-bit path).
"""

from __future__ import annotations

import argparse
import json
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="GRPO with a verified reward.")
    parser.add_argument("--model", default="unsloth/Llama-3.2-1B")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-completion", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--out", default="out/grpo")
    parser.add_argument("--prompts", default=None, help="A text file, one prompt per line.")
    parser.add_argument("--dry-run", action="store_true", help="Validate the wiring, no GPU.")
    args = parser.parse_args(argv)

    from agentsynth import make_reward_fn
    from agentsynth.environments import SQLEnvironment

    environment = SQLEnvironment()
    reward_fn = make_reward_fn(environment=environment)

    if args.prompts:
        with open(args.prompts, encoding="utf-8") as fh:
            prompts = [line.strip() for line in fh if line.strip()]
    else:
        tools_json = json.dumps(
            [
                {"name": t.name, "description": t.description, "parameters": t.parameters}
                for t in environment.tools()
            ]
        )
        prefix = f"You can call exactly one tool to help the user.\nTools (JSON): {tools_json}\n\n"
        suffix = '\nRespond with ONLY a JSON object: {"tool": "<tool name>", "args": {<arguments>}}'
        from agentsynth import sample_tasks

        prompts = [prefix + "User: " + t.query + suffix for t in sample_tasks(24, None, seed=7)]

    print(f"{len(prompts)} prompts | reward checks: parse, tool, args, real execution")

    if args.dry_run:
        good = json.dumps({"tool": "sql_query", "args": {"query": "SELECT COUNT(*) FROM sales"}})
        scores = reward_fn(prompts=prompts[:2], completions=[good, "not a tool call"])
        print(f"dry-run OK: reward_fn scored [{scores[0]}, {scores[1]}] — wiring is sound.")
        environment.close()
        return 0

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_fn,
        train_dataset=Dataset.from_list([{"prompt": p} for p in prompts]),
        args=GRPOConfig(
            output_dir=args.out,
            max_steps=args.steps,
            num_generations=args.num_generations,
            per_device_train_batch_size=args.batch_size,
            max_completion_length=args.max_completion,
            learning_rate=args.lr,
            logging_steps=5,
            report_to=[],
        ),
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"Saved to {args.out}")
    environment.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
