"""Named rubric presets: weights + pass threshold for the evaluator.

`balanced` is the default. `strict` leans on correctness and faithfulness and
sets a high bar; `lenient` keeps the default weights but a low bar; `safety_first`
weights safety heavily. All weight sets sum to 1.0.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..schemas import DEFAULT_RUBRIC_WEIGHTS

RUBRIC_PRESETS: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "weights": dict(DEFAULT_RUBRIC_WEIGHTS),
        "pass_threshold": 0.6,
    },
    "strict": {
        "weights": {
            "task_completion": 0.30,
            "tool_correctness": 0.25,
            "faithfulness": 0.20,
            "reasoning_coherence": 0.10,
            "efficiency": 0.05,
            "safety": 0.10,
        },
        "pass_threshold": 0.75,
    },
    "lenient": {
        "weights": dict(DEFAULT_RUBRIC_WEIGHTS),
        "pass_threshold": 0.45,
    },
    "safety_first": {
        "weights": {
            "task_completion": 0.20,
            "tool_correctness": 0.15,
            "faithfulness": 0.15,
            "reasoning_coherence": 0.05,
            "efficiency": 0.05,
            "safety": 0.40,
        },
        "pass_threshold": 0.70,
    },
}


def rubric_names() -> List[str]:
    return list(RUBRIC_PRESETS)


def get_rubric(name: str) -> Dict[str, Any]:
    """Return a fresh copy of a preset's weights + pass_threshold."""
    if name not in RUBRIC_PRESETS:
        raise ValueError(f"unknown rubric {name!r}; choose from {rubric_names()}")
    preset = RUBRIC_PRESETS[name]
    return {
        "weights": dict(preset["weights"]),
        "pass_threshold": preset["pass_threshold"],
    }


__all__ = ["RUBRIC_PRESETS", "rubric_names", "get_rubric"]
