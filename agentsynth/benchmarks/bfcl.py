"""Load Berkeley Function-Calling Leaderboard (BFCL) cases into the harness.

BFCL ships JSONL: a questions file (`id`, `question`, `function`) and a
possible-answers file (`id`, `ground_truth`). This converts the pair into
`BenchmarkCase`s so any `model_fn` can be scored with `run_benchmark`.

The scoring here is a simplified tool + argument match (arguments accept the list
of allowed values BFCL gives), not BFCL's full AST checker. Use it for a quick
before/after signal; use BFCL's own scorer for the leaderboard.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .tool_calling import BenchmarkCase


def _user_query(question: Any) -> str:
    turns = question[0] if question and isinstance(question[0], list) else question
    for msg in turns or []:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", ""))
    return str(question)


def _normalize_tools(functions: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for fn in functions or []:
        params = dict(fn.get("parameters") or {})
        if params.get("type") == "dict":  # BFCL uses "dict"; JSON Schema uses "object"
            params["type"] = "object"
        out.append(
            {"name": fn.get("name"), "description": fn.get("description", ""), "parameters": params}
        )
    return out


def bfcl_case(
    question_rec: Dict[str, Any], answer_rec: Optional[Dict[str, Any]] = None
) -> Optional[BenchmarkCase]:
    """Convert one BFCL question (+ its ground truth) into a BenchmarkCase."""
    ground_truth = (answer_rec or {}).get("ground_truth") or []
    if not ground_truth or not isinstance(ground_truth[0], dict) or not ground_truth[0]:
        return None
    first = ground_truth[0]
    expected_tool = next(iter(first))
    expected_args = first[expected_tool] if isinstance(first[expected_tool], dict) else {}
    return BenchmarkCase(
        id=str(question_rec.get("id", "")),
        query=_user_query(question_rec.get("question")),
        expected_tool=expected_tool,
        expected_args=expected_args,
        tools=_normalize_tools(question_rec.get("function")),
    )


def _parse_jsonl(text: str) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return _parse_jsonl(fh.read())


def _build_cases(
    questions: List[Dict[str, Any]], answers: Dict[str, Dict[str, Any]]
) -> List[BenchmarkCase]:
    cases: List[BenchmarkCase] = []
    for q in questions:
        case = bfcl_case(q, answers.get(str(q.get("id"))))
        if case is not None:
            cases.append(case)
    return cases


def load_bfcl(questions_path: str, answers_path: Optional[str] = None) -> List[BenchmarkCase]:
    """Load BFCL question + possible-answer JSONL files into BenchmarkCases."""
    answers: Dict[str, Dict[str, Any]] = {}
    if answers_path:
        answers = {str(a.get("id")): a for a in _read_jsonl(answers_path)}
    return _build_cases(_read_jsonl(questions_path), answers)


_SAMPLE_FILES = {
    "simple": "bfcl_sample",  # 25 cases, one candidate function each (format check)
    "multiple": "bfcl_multiple",  # 25 cases, 2-3 candidate functions (real tool selection)
}


def load_sample_bfcl(split: str = "simple") -> List[BenchmarkCase]:
    """A bundled 25-case BFCL slice. Runs offline, no download.

    `split="simple"` is the `simple_python` split — one candidate function per case,
    so it mostly checks output formatting. `split="multiple"` gives each case several
    candidate functions, which is what actually exercises tool *selection*.

    Real slices of the Berkeley Function-Calling Leaderboard (Apache-2.0); see
    `agentsynth/benchmarks/data/NOTICE.md`. For the full suite, point `load_bfcl`
    at the official BFCL files.
    """
    from importlib.resources import files

    stem = _SAMPLE_FILES.get(split)
    if stem is None:
        raise ValueError(f"unknown BFCL sample split {split!r}; use one of {sorted(_SAMPLE_FILES)}")
    data = files("agentsynth.benchmarks") / "data"
    questions = _parse_jsonl((data / f"{stem}.questions.jsonl").read_text(encoding="utf-8"))
    raw = _parse_jsonl((data / f"{stem}.answers.jsonl").read_text(encoding="utf-8"))
    return _build_cases(questions, {str(a.get("id")): a for a in raw})


# A couple of BFCL-format records, for tests and an offline demo.
SAMPLE_BFCL: List[Tuple[Dict[str, Any], Dict[str, Any]]] = [
    (
        {
            "id": "simple_0",
            "question": [[{"role": "user", "content": "What is the weather in Paris?"}]],
            "function": [
                {
                    "name": "get_weather",
                    "description": "Get the current weather for a city.",
                    "parameters": {
                        "type": "dict",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        },
        {"id": "simple_0", "ground_truth": [{"get_weather": {"city": ["Paris", "paris"]}}]},
    ),
    (
        {
            "id": "simple_1",
            "question": [[{"role": "user", "content": "Compute 15 factorial."}]],
            "function": [
                {
                    "name": "math_factorial",
                    "description": "Compute the factorial of a number.",
                    "parameters": {
                        "type": "dict",
                        "properties": {"n": {"type": "integer"}},
                        "required": ["n"],
                    },
                }
            ],
        },
        {"id": "simple_1", "ground_truth": [{"math_factorial": {"n": [15]}}]},
    ),
]


def sample_cases() -> List[BenchmarkCase]:
    return [c for c in (bfcl_case(q, a) for q, a in SAMPLE_BFCL) if c is not None]


__all__ = ["bfcl_case", "load_bfcl", "load_sample_bfcl", "sample_cases", "SAMPLE_BFCL"]
