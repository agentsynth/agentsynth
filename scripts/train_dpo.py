#!/usr/bin/env python
"""DPO fine-tune on AgentSynth preference pairs (TRL).

Needs a GPU for a real run. Usually you run this after SFT, pointing --model at the
SFT output. Use --dry-run to validate the data on CPU.

    python scripts/train_dpo.py --data data/dpo.jsonl --model out/sft --out out/dpo
    python scripts/train_dpo.py --data data/dpo.jsonl --dry-run

Install: pip install "agentsynth-ai[train]".
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional


def _load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="DPO on AgentSynth preference pairs.")
    parser.add_argument("--data", required=True, help="DPO JSONL (prompt/chosen/rejected).")
    parser.add_argument("--model", required=True, help="Base model, often the SFT output dir.")
    parser.add_argument("--out", default="out/dpo")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true", help="Validate data, no training.")
    args = parser.parse_args(argv)

    records = _load_records(args.data)
    print(f"Loaded {len(records)} preference pairs from {args.data}")
    needed = {"prompt", "chosen", "rejected"}
    if not records or not needed <= set(records[0]):
        raise SystemExit("error: expected DPO records with prompt/chosen/rejected fields")
    if args.dry_run:
        print("dry-run OK: data looks valid; skipping training.")
        return 0

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    dataset = Dataset.from_list(records)
    config = DPOConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        beta=args.beta,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=10,
    )
    trainer = DPOTrainer(
        model=model, args=config, train_dataset=dataset, processing_class=tokenizer
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"Saved DPO model to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
