"""Trainer-ready dataset builders for SFT and DPO."""

from __future__ import annotations

from .datasets import (
    build_dpo_dataset,
    build_sft_dataset,
    to_dpo_records,
    to_sft_records,
    write_jsonl,
)

__all__ = [
    "to_sft_records",
    "to_dpo_records",
    "write_jsonl",
    "build_sft_dataset",
    "build_dpo_dataset",
]
