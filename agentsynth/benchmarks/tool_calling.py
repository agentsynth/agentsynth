"""A small function-calling benchmark.

Each case gives a query and the full tool catalog; the model has to pick the right
tool and supply sane arguments. Scoring is deliberately simple — tool match plus a
lenient arg check — so it's easy to reason about and produces a clear before/after
table. A "model" here is any callable `(query, tools) -> (tool_name, tool_args)`,
so you can score a real LLM, a fine-tuned model, or AgentSynth's own generator.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ..schemas import ToolSpec
from ..utils import DEFAULT_TOOL_CATALOG, extract_json, parse_tool_catalog

ModelFn = Callable[[str, List[ToolSpec]], Tuple[Optional[str], Dict[str, Any]]]


class BenchmarkCase(BaseModel):
    id: str
    query: str
    expected_tool: str
    expected_args: Dict[str, Any] = Field(default_factory=dict)  # value None => key presence only
    # The full default catalog (with parameter schemas), so the model must both pick the
    # right tool and fill its arguments.
    tools: List[Dict[str, Any]] = Field(default_factory=lambda: list(DEFAULT_TOOL_CATALOG))

    def tool_specs(self) -> List[ToolSpec]:
        return parse_tool_catalog(self.tools)


class CaseResult(BaseModel):
    id: str
    expected_tool: str
    predicted_tool: Optional[str]
    tool_ok: bool
    args_ok: bool
    score: float


class BenchmarkReport(BaseModel):
    n: int
    tool_accuracy: float
    arg_accuracy: float
    score: float
    results: List[CaseResult] = Field(default_factory=list)


BUILTIN_CASES: List[BenchmarkCase] = [
    BenchmarkCase(
        id="weather_paris",
        query="What's the weather in Paris right now?",
        expected_tool="get_weather",
        expected_args={"city": "Paris"},
    ),
    BenchmarkCase(
        id="weather_tokyo",
        query="Is it raining in Tokyo today?",
        expected_tool="get_weather",
        expected_args={"city": "Tokyo"},
    ),
    BenchmarkCase(
        id="math_mult",
        query="What is 23 times 7 plus 4?",
        expected_tool="calculator",
        expected_args={"expression": None},
    ),
    BenchmarkCase(
        id="math_tip",
        query="Calculate an 18% tip on a $54 bill.",
        expected_tool="calculator",
        expected_args={"expression": None},
    ),
    BenchmarkCase(
        id="search_news",
        query="Find recent news about open-source AI agents.",
        expected_tool="web_search",
        expected_args={"query": None},
    ),
    BenchmarkCase(
        id="search_fact",
        query="Search the web for the population of Vietnam.",
        expected_tool="web_search",
        expected_args={"query": None},
    ),
    BenchmarkCase(
        id="file_csv",
        query="Read the file data/report.csv and summarize it.",
        expected_tool="read_file",
        expected_args={"path": None},
    ),
    BenchmarkCase(
        id="file_notes",
        query="Open notes.md and list the action items.",
        expected_tool="read_file",
        expected_args={"path": None},
    ),
    BenchmarkCase(
        id="sql_revenue",
        query="Query the database for total revenue by region.",
        expected_tool="sql_query",
        expected_args={"query": None},
    ),
    BenchmarkCase(
        id="sql_count",
        query="Run a SQL query to count rows in the sales table.",
        expected_tool="sql_query",
        expected_args={"query": None},
    ),
    BenchmarkCase(
        id="email_launch",
        query="Send an email to team@example.com about the launch.",
        expected_tool="send_email",
        expected_args={"to": None},
    ),
    BenchmarkCase(
        id="email_summary",
        query="Email a summary of the report to alex@example.com.",
        expected_tool="send_email",
        expected_args={"to": None},
    ),
]


def _args_ok(expected: Dict[str, Any], predicted: Dict[str, Any]) -> bool:
    for key, value in (expected or {}).items():
        if key not in (predicted or {}):
            return False
        if value is None:
            continue  # key presence is enough
        pred = str(predicted[key]).strip().lower()
        if isinstance(value, (list, tuple)):  # any acceptable value (BFCL-style)
            if pred not in [str(v).strip().lower() for v in value]:
                return False
        elif pred != str(value).strip().lower():
            return False
    return True


def _mean(values: List[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def run_benchmark(
    model_fn: ModelFn, cases: Optional[List[BenchmarkCase]] = None
) -> BenchmarkReport:
    """Score a model on the function-calling cases.

    `model_fn(query, tools)` returns the `(tool_name, tool_args)` the model would call.
    """
    cases = cases or BUILTIN_CASES
    results: List[CaseResult] = []
    for case in cases:
        try:
            predicted_tool, predicted_args = model_fn(case.query, case.tool_specs())
        except Exception:
            predicted_tool, predicted_args = None, {}
        tool_ok = predicted_tool == case.expected_tool
        args_ok = tool_ok and _args_ok(case.expected_args, predicted_args or {})
        score = 1.0 if (tool_ok and args_ok) else (0.5 if tool_ok else 0.0)
        results.append(
            CaseResult(
                id=case.id,
                expected_tool=case.expected_tool,
                predicted_tool=predicted_tool,
                tool_ok=tool_ok,
                args_ok=args_ok,
                score=score,
            )
        )
    return BenchmarkReport(
        n=len(results),
        tool_accuracy=_mean([1.0 if r.tool_ok else 0.0 for r in results]),
        arg_accuracy=_mean([1.0 if r.args_ok else 0.0 for r in results]),
        score=_mean([r.score for r in results]),
        results=results,
    )


def compare_models(
    before: ModelFn, after: ModelFn, cases: Optional[List[BenchmarkCase]] = None
) -> Dict[str, Any]:
    """Run two models on the same cases and return a before/after comparison."""
    cases = cases or BUILTIN_CASES
    before_report = run_benchmark(before, cases)
    after_report = run_benchmark(after, cases)
    return {
        "before": before_report,
        "after": after_report,
        "delta_tool_accuracy": round(after_report.tool_accuracy - before_report.tool_accuracy, 4),
        "delta_score": round(after_report.score - before_report.score, 4),
        "n": before_report.n,
    }


def agentsynth_model(generator: Any, mode: str = "single_agent") -> ModelFn:
    """Adapt an AgentTrajectoryGenerator into a benchmark model: it takes the first
    tool call the generated trajectory makes."""

    def model_fn(query: str, tools: List[ToolSpec]) -> Tuple[Optional[str], Dict[str, Any]]:
        traj = generator.generate(query, tools=tools, mode=mode)
        calls = traj.tool_calls()
        if not calls:
            return None, {}
        return calls[0].tool_name, calls[0].tool_args or {}

    return model_fn


def prompted_model(complete_fn: Callable[[str], str]) -> ModelFn:
    """Turn a text-completion function `(prompt) -> text` into a benchmark model.

    It asks the model for a single JSON tool call and parses the reply, so it works
    with any instruction-following model (a base or fine-tuned HF model, etc.).
    """

    def model_fn(query: str, tools: List[ToolSpec]) -> Tuple[Optional[str], Dict[str, Any]]:
        tool_json = json.dumps(
            [
                {"name": t.name, "description": t.description, "parameters": t.parameters}
                for t in tools
            ]
        )
        prompt = (
            "You can call exactly one tool to help the user.\n"
            f"Tools (JSON): {tool_json}\n\n"
            f"User: {query}\n"
            'Respond with ONLY a JSON object: {"tool": "<tool name>", "args": {<arguments>}}'
        )
        parsed = extract_json(complete_fn(prompt))
        if isinstance(parsed, dict):
            args = parsed.get("args")
            return parsed.get("tool"), args if isinstance(args, dict) else {}
        return None, {}

    return model_fn


def report_table_md(comparison: Dict[str, Any]) -> str:
    """Render a before/after comparison as a markdown table."""
    before = comparison["before"]
    after = comparison["after"]

    def pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    lines = [
        f"Function-calling benchmark ({comparison['n']} cases)",
        "",
        "| Metric | Before | After | Δ |",
        "| --- | --- | --- | --- |",
        f"| Tool accuracy | {pct(before.tool_accuracy)} | {pct(after.tool_accuracy)} "
        f"| {pct(comparison['delta_tool_accuracy'])} |",
        f"| Arg accuracy | {pct(before.arg_accuracy)} | {pct(after.arg_accuracy)} | |",
        f"| Overall score | {pct(before.score)} | {pct(after.score)} "
        f"| {pct(comparison['delta_score'])} |",
    ]
    return "\n".join(lines)


__all__ = [
    "BenchmarkCase",
    "CaseResult",
    "BenchmarkReport",
    "BUILTIN_CASES",
    "run_benchmark",
    "compare_models",
    "agentsynth_model",
    "prompted_model",
    "report_table_md",
]
