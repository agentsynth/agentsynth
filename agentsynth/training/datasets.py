"""Turn trajectories and preference pairs into trainer-ready records.

SFT records are conversational (`{"messages": [...]}`) and DPO records are
`{"prompt", "chosen", "rejected"}` — the shapes TRL's SFTTrainer and DPOTrainer
read directly.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..schemas import EvalResult, Trajectory


def to_sft_records(
    trajectories: List[Trajectory],
    only_passed: bool = False,
    eval_results: Optional[List[EvalResult]] = None,
) -> List[Dict[str, Any]]:
    """Conversational SFT records. With `only_passed`, keep just the trajectories
    whose eval result passed."""
    keep: Optional[set] = None
    if only_passed and eval_results is not None:
        keep = {e.trajectory_id for e in eval_results if e.passed}
    records: List[Dict[str, Any]] = []
    for traj in trajectories:
        if keep is not None and traj.id not in keep:
            continue
        records.append({"id": traj.id, "mode": traj.mode, "messages": traj.to_messages()})
    return records


def to_dpo_records(pairs: List[Any]) -> List[Dict[str, Any]]:
    """Prompt/chosen/rejected records from preference pairs."""
    return [{"prompt": p.query, "chosen": p.chosen, "rejected": p.rejected} for p in pairs]


def write_jsonl(records: List[Dict[str, Any]], path: str) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def build_sft_dataset(
    trajectories: List[Trajectory],
    path: Optional[str] = None,
    only_passed: bool = False,
    eval_results: Optional[List[EvalResult]] = None,
) -> List[Dict[str, Any]]:
    records = to_sft_records(trajectories, only_passed=only_passed, eval_results=eval_results)
    if path:
        write_jsonl(records, path)
    return records


def build_dpo_dataset(pairs: List[Any], path: Optional[str] = None) -> List[Dict[str, Any]]:
    records = to_dpo_records(pairs)
    if path:
        write_jsonl(records, path)
    return records


__all__ = [
    "to_sft_records",
    "to_dpo_records",
    "write_jsonl",
    "build_sft_dataset",
    "build_dpo_dataset",
]
