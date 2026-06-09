#!/usr/bin/env python
"""Supervised fine-tune on an AgentSynth SFT dataset (TRL, optionally Unsloth).

A real run needs a GPU. A free Colab T4 handles an 8B 4-bit model with LoRA.
Use --dry-run to validate the data and config on CPU without training.

    python scripts/train_sft.py --data data/train.jsonl --model unsloth/llama-3.1-8b-bnb-4bit
    python scripts/train_sft.py --data data/train.jsonl --dry-run

Install: pip install "agentsynth-ai[train]"  (plus `unsloth` for the fast 4-bit path).
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional


def _load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SFT on an AgentSynth dataset.")
    parser.add_argument("--data", required=True, help="SFT JSONL (conversational `messages`).")
    parser.add_argument("--model", default="unsloth/llama-3.1-8b-bnb-4bit")
    parser.add_argument("--out", default="out/sft")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate data + config, no training."
    )
    args = parser.parse_args(argv)

    records = _load_records(args.data)
    print(f"Loaded {len(records)} SFT records from {args.data}")
    if not records or "messages" not in records[0]:
        raise SystemExit("error: expected conversational SFT records with a 'messages' field")
    if args.dry_run:
        print("dry-run OK: data and config look valid; skipping training.")
        return 0

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            args.model, max_seq_length=args.max_seq_len, load_in_4bit=True
        )
        model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16, lora_dropout=0.0)
    except Exception:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)

    dataset = Dataset.from_list([{"messages": r["messages"]} for r in records])
    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=10,
    )
    # trl >= 1.0 uses processing_class (older versions used tokenizer=)
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, train_dataset=dataset, args=config
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"Saved fine-tuned model to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
