"""Pydantic models for trajectories, tools, and eval results.

Everything else in agentsynth is built on these, so they stay permissive.
Targets Python 3.9+ and Pydantic v2.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

STEP_TYPES = (
    "thought",
    "plan",
    "tool_call",
    "observation",
    "code_execution",
    "critique",
    "final_answer",
)

TRAJECTORY_MODES = ("single_agent", "multi_agent", "code_execution")


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class ToolSpec(BaseModel):
    """A single tool the agent may call.

    `parameters` is a JSON-Schema object, matching the OpenAI/Anthropic
    function-calling convention:

        {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "..."}},
            "required": ["city"],
        }
    """

    name: str
    description: str = ""
    parameters: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []}
    )

    def arg_names(self) -> List[str]:
        props = self.parameters.get("properties", {}) if isinstance(self.parameters, dict) else {}
        return list(props.keys())

    def required_args(self) -> List[str]:
        req = self.parameters.get("required", []) if isinstance(self.parameters, dict) else []
        return list(req) if isinstance(req, list) else []

    def arg_schema(self, name: str) -> Dict[str, Any]:
        props = self.parameters.get("properties", {}) if isinstance(self.parameters, dict) else {}
        val = props.get(name, {})
        return val if isinstance(val, dict) else {}


class TrajectoryStep(BaseModel):
    """One step in a trajectory.

    Union-shaped: only the fields relevant to `step_type` are filled. Keeping it
    flat is what lets TRL/Unsloth/Axolotl trainers load the JSONL without a custom
    parser.
    """

    step_type: str = "thought"
    agent: Optional[str] = None  # which sub-agent emitted this step (multi-agent)
    thought: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    code: Optional[str] = None
    code_output: Optional[str] = None
    content: Optional[str] = None  # final_answer / plan / critique free text

    def short(self) -> str:
        """One-line rendering for logs and the UI."""
        if self.step_type == "tool_call":
            return f"tool_call {self.tool_name}({self.tool_args or {}})"
        if self.step_type == "observation":
            obs = (self.observation or "").replace("\n", " ")
            return f"observation: {obs[:80]}"
        if self.step_type == "code_execution":
            return f"code_execution: {(self.code or '')[:60]!r}"
        if self.step_type in ("thought", "plan", "critique", "final_answer"):
            return f"{self.step_type}: {(self.thought or self.content or '')[:80]}"
        return self.step_type


class Trajectory(BaseModel):
    id: str = Field(default_factory=_new_id)
    query: str
    mode: str = "single_agent"
    domain: Optional[str] = None
    tools: List[ToolSpec] = Field(default_factory=list)
    steps: List[TrajectoryStep] = Field(default_factory=list)
    final_answer: str = ""
    success: bool = True
    generator_model: str = "mock"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)

    def num_steps(self) -> int:
        return len(self.steps)

    def tool_calls(self) -> List[TrajectoryStep]:
        return [s for s in self.steps if s.step_type == "tool_call"]

    def observations(self) -> List[TrajectoryStep]:
        return [s for s in self.steps if s.step_type == "observation"]

    def tool_names_used(self) -> List[str]:
        return [s.tool_name for s in self.tool_calls() if s.tool_name]

    def tool_signature(self) -> str:
        """Stable signature of the tool sequence, for diversity metrics."""
        return ">".join(self.tool_names_used())

    def to_messages(self) -> List[Dict[str, Any]]:
        """Render as an OpenAI-style `messages` list.

        thought/plan/critique collapse into assistant content, a tool_call
        becomes an assistant message carrying `tool_calls`, and each observation
        becomes a role="tool" message.
        """
        messages: List[Dict[str, Any]] = [{"role": "user", "content": self.query}]
        pending_thought: List[str] = []
        call_counter = 0

        def flush_thought() -> None:
            if pending_thought:
                messages.append({"role": "assistant", "content": "\n".join(pending_thought)})
                pending_thought.clear()

        for step in self.steps:
            if step.step_type in ("thought", "plan", "critique"):
                txt = step.thought or step.content or ""
                if txt:
                    prefix = f"[{step.agent}] " if step.agent else ""
                    pending_thought.append(prefix + txt)
            elif step.step_type == "tool_call":
                flush_thought()
                call_id = f"call_{call_counter}"
                call_counter += 1
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": step.tool_name,
                                    "arguments": step.tool_args or {},
                                },
                            }
                        ],
                    }
                )
            elif step.step_type == "observation":
                messages.append({"role": "tool", "content": step.observation or ""})
            elif step.step_type == "code_execution":
                flush_thought()
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"```python\n{step.code or ''}\n```",
                    }
                )
                messages.append({"role": "tool", "content": step.code_output or ""})
            elif step.step_type == "final_answer":
                flush_thought()
                messages.append({"role": "assistant", "content": step.content or self.final_answer})

        flush_thought()
        # Always end on an assistant turn.
        if self.final_answer and (not messages or messages[-1].get("role") != "assistant"):
            messages.append({"role": "assistant", "content": self.final_answer})
        return messages


# Rubric dimension order and default weights (sum to 1.0).
RUBRIC_DIMENSIONS = (
    "task_completion",
    "tool_correctness",
    "faithfulness",
    "reasoning_coherence",
    "efficiency",
    "safety",
)

DEFAULT_RUBRIC_WEIGHTS = {
    "task_completion": 0.30,
    "tool_correctness": 0.20,
    "faithfulness": 0.15,
    "reasoning_coherence": 0.15,
    "efficiency": 0.10,
    "safety": 0.10,
}


class RubricScores(BaseModel):
    """LLM-as-Judge scores, each in [0, 1]."""

    task_completion: float = 0.0
    tool_correctness: float = 0.0
    faithfulness: float = 0.0
    reasoning_coherence: float = 0.0
    efficiency: float = 0.0
    safety: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {d: float(getattr(self, d)) for d in RUBRIC_DIMENSIONS}

    def weighted_overall(self, weights: Optional[Dict[str, float]] = None) -> float:
        w = weights or DEFAULT_RUBRIC_WEIGHTS
        total_w = sum(w.get(d, 0.0) for d in RUBRIC_DIMENSIONS) or 1.0
        return round(
            sum(float(getattr(self, d)) * w.get(d, 0.0) for d in RUBRIC_DIMENSIONS) / total_w,
            4,
        )


class EvalResult(BaseModel):
    trajectory_id: str
    scores: RubricScores = Field(default_factory=RubricScores)
    overall: float = 0.0
    passed: bool = False
    explanation: str = ""
    judge_model: str = "mock"

    def flat(self) -> Dict[str, Any]:
        """Flattened dict, handy for dataframes and metrics."""
        row: Dict[str, Any] = {
            "trajectory_id": self.trajectory_id,
            "overall": self.overall,
            "passed": self.passed,
            "judge_model": self.judge_model,
        }
        row.update(self.scores.as_dict())
        return row


__all__ = [
    "STEP_TYPES",
    "TRAJECTORY_MODES",
    "RUBRIC_DIMENSIONS",
    "DEFAULT_RUBRIC_WEIGHTS",
    "ToolSpec",
    "TrajectoryStep",
    "Trajectory",
    "RubricScores",
    "EvalResult",
]
