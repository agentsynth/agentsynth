#!/usr/bin/env python
"""Score a model on the AgentSynth function-calling benchmark.

python scripts/run_benchmark.py --model mock
python scripts/run_benchmark.py --model gpt-4o-mini             # via LiteLLM
python scripts/run_benchmark.py --before gpt-4o-mini --after my-finetuned-model
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple


def _litellm_model(model_id: str):
    def model_fn(query: str, tools: List[Any]) -> Tuple[Optional[str], Dict[str, Any]]:
        import litellm

        tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]
        try:
            resp = litellm.completion(
                model=model_id,
                messages=[{"role": "user", "content": query}],
                tools=tool_defs,
                tool_choice="auto",
            )
            calls = resp["choices"][0]["message"].get("tool_calls") or []
            if not calls:
                return None, {}
            fn = calls[0]["function"]
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            return fn.get("name"), args or {}
        except Exception:
            return None, {}

    return model_fn


def _make_model(spec: Optional[str]):
    if spec in (None, "mock"):
        from agentsynth import AgentTrajectoryGenerator
        from agentsynth.benchmarks import agentsynth_model

        return agentsynth_model(AgentTrajectoryGenerator(use_mock=True))
    return _litellm_model(spec)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the AgentSynth function-calling benchmark.")
    parser.add_argument("--model", default=None, help="Model id, or 'mock'.")
    parser.add_argument("--before", default=None, help="Baseline model for a before/after table.")
    parser.add_argument("--after", default=None, help="Candidate model for a before/after table.")
    args = parser.parse_args(argv)

    from agentsynth.benchmarks import compare_models, report_table_md, run_benchmark

    if args.before and args.after:
        comparison = compare_models(_make_model(args.before), _make_model(args.after))
        print(report_table_md(comparison))
    else:
        report = run_benchmark(_make_model(args.model))
        print(
            f"tool_accuracy={report.tool_accuracy:.1%}  "
            f"arg_accuracy={report.arg_accuracy:.1%}  "
            f"score={report.score:.1%}  (n={report.n})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
