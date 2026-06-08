"""Preference pairs for DPO / KTO-style training.

For each query, generate several trajectories, score them, and pair the best
("chosen") against the worst ("rejected"). Export in the prompt/chosen/rejected
shape TRL's DPOTrainer expects.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .schemas import Trajectory


class PreferencePair(BaseModel):
    query: str
    prompt: List[Dict[str, Any]]  # conversational prompt (the user turn)
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float
    margin: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


def _render_completion(trajectory: Trajectory) -> str:
    """A readable assistant-side rendering of a trajectory, used as a completion."""
    lines: List[str] = []
    for step in trajectory.steps:
        agent = f"[{step.agent}] " if step.agent else ""
        if step.step_type in ("thought", "plan", "critique"):
            text = (step.thought or step.content or "").strip()
            if text:
                lines.append(f"{agent}{text}")
        elif step.step_type == "tool_call":
            lines.append(f"{agent}Action: {step.tool_name}({step.tool_args or {}})")
        elif step.step_type == "observation":
            lines.append(f"Observation: {(step.observation or '').strip()}")
        elif step.step_type == "code_execution":
            lines.append(f"Code:\n{(step.code or '').strip()}")
            lines.append(f"Output: {(step.code_output or '').strip()}")
    answer = trajectory.final_answer.strip() if trajectory.final_answer else ""
    if answer:
        lines.append(f"Final answer: {answer}")
    return "\n".join(lines).strip()


def build_preference_pairs(
    generator: Any,
    evaluator: Any,
    queries: Union[str, List[str]],
    k: int = 4,
    mode: str = "single_agent",
    tools: Optional[Any] = None,
    min_margin: float = 0.0,
) -> List[PreferencePair]:
    """Generate `k` trajectories per query and pair best vs worst by judge score.

    A pair is emitted when the two trajectories differ and their score gap is at
    least `min_margin`.
    """
    if isinstance(queries, str):
        queries = [queries]
    k = max(2, int(k))

    pairs: List[PreferencePair] = []
    for query in queries:
        trajectories = [
            generator.generate(query, tools=tools, mode=mode, index=i) for i in range(k)
        ]
        scored = [(t, evaluator.evaluate(t)) for t in trajectories]
        scored.sort(key=lambda pair: pair[1].overall, reverse=True)

        (chosen_traj, chosen_eval) = scored[0]
        (rejected_traj, rejected_eval) = scored[-1]
        margin = round(chosen_eval.overall - rejected_eval.overall, 4)
        if chosen_traj.id == rejected_traj.id or margin < min_margin:
            continue

        pairs.append(
            PreferencePair(
                query=query,
                prompt=[{"role": "user", "content": query}],
                chosen=_render_completion(chosen_traj),
                rejected=_render_completion(rejected_traj),
                chosen_score=chosen_eval.overall,
                rejected_score=rejected_eval.overall,
                margin=margin,
                metadata={
                    "chosen_id": chosen_traj.id,
                    "rejected_id": rejected_traj.id,
                    "mode": mode,
                    "k": k,
                },
            )
        )
    return pairs


def to_dpo_jsonl(pairs: List[PreferencePair], path: Optional[str] = None) -> str:
    """Serialize pairs as prompt/chosen/rejected JSONL (TRL DPO compatible)."""
    lines = [
        json.dumps(
            {
                "prompt": p.query,
                "chosen": p.chosen,
                "rejected": p.rejected,
                "margin": p.margin,
            },
            ensure_ascii=False,
        )
        for p in pairs
    ]
    text = "\n".join(lines)
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return text


def load_dpo_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


__all__ = ["PreferencePair", "build_preference_pairs", "to_dpo_jsonl", "load_dpo_jsonl"]
