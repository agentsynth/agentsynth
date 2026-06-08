"""argparse front-end for the `agentsynth` console script.

Heavy submodule imports live inside the command handlers so `--help` stays fast.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence

_FORMATS = ("jsonl", "sharegpt", "adp")


def _truthy(value: Optional[str]) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on") if value else False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentsynth",
        description=(
            "Synthetic Agentic Trajectories Generator + LLM-as-Judge Eval Loop. "
            "Works fully offline (deterministic mock) with no API keys."
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="{generate,eval}")

    gen = sub.add_parser(
        "generate",
        help="Generate synthetic agent trajectories and export them to a dataset file.",
        description="Generate N synthetic agent trajectories for a query and export them.",
    )
    gen.add_argument(
        "--query",
        "-q",
        required=True,
        help="The user task/query the agent should solve.",
    )
    gen.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of trajectories to generate (default: 10).",
    )
    gen.add_argument(
        "--mode",
        choices=("single_agent", "multi_agent", "code_execution"),
        default="single_agent",
        help="Trajectory mode to generate (default: single_agent).",
    )
    gen.add_argument(
        "--tools",
        default=None,
        metavar="PATH",
        help="Path to a JSON file describing the tool catalog (defaults to the built-in catalog).",
    )
    gen.add_argument(
        "--out",
        "-o",
        default=None,
        metavar="PATH",
        help="Output dataset path (default: ./agentsynth_dataset.jsonl).",
    )
    gen.add_argument(
        "--format",
        "-f",
        choices=_FORMATS,
        default="jsonl",
        help="Export format (default: jsonl).",
    )
    gen.add_argument(
        "--vary-modes",
        action="store_true",
        help="Cycle across all trajectory modes instead of using a single --mode.",
    )
    gen.add_argument(
        "--max-steps",
        type=int,
        default=6,
        help="Soft cap on the number of steps per trajectory (default: 6).",
    )
    gen.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the (optional) LLM backend (default: 0.7).",
    )

    ev = sub.add_parser(
        "eval",
        help="Evaluate a JSONL dataset of trajectories with the LLM-as-Judge loop.",
        description="Load trajectories from JSONL, score them and print metrics.",
    )
    ev.add_argument(
        "--in",
        "-i",
        dest="in_path",
        required=True,
        metavar="PATH",
        help="Path to a JSONL file of trajectories to evaluate.",
    )
    ev.add_argument(
        "--out",
        "-o",
        default=None,
        metavar="PATH",
        help="Optional path to write per-trajectory flat scores as JSONL.",
    )

    return parser


def _load_tools(path: Optional[str]):
    """Parse a tool catalog file into ToolSpecs, or None to use the defaults."""
    if not path:
        return None
    from .utils import parse_tool_catalog

    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except OSError as exc:
        raise SystemExit(f"error: could not read tools file '{path}': {exc}")
    return parse_tool_catalog(raw)


def _summarize_modes(trajectories: Sequence) -> str:
    """Compact `mode=count` summary for a batch."""
    counts: Dict[str, int] = {}
    for traj in trajectories:
        mode = getattr(traj, "mode", "?")
        counts[mode] = counts.get(mode, 0) + 1
    return ", ".join(f"{mode}={counts[mode]}" for mode in sorted(counts))


def _cmd_generate(args: argparse.Namespace) -> int:
    from .exporters import save_dataset
    from .generator import AgentTrajectoryGenerator

    tools = _load_tools(args.tools)

    out_path = args.out or "./agentsynth_dataset.jsonl"

    generator = AgentTrajectoryGenerator(
        use_mock="auto",
        temperature=args.temperature,
        max_steps=args.max_steps,
        tools=tools,
    )

    trajectories = generator.generate_batch(
        args.query,
        num_trajectories=args.n,
        mode=args.mode,
        vary_modes=args.vary_modes,
    )

    save_dataset(trajectories, out_path, fmt=args.format)

    modes = _summarize_modes(trajectories)
    backend = "llm" if getattr(generator, "use_llm", False) else "mock"
    print(
        f"Generated {len(trajectories)} trajectories "
        f"[{modes}] (backend={backend}) -> {out_path} (format={args.format})"
    )
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .evaluator import TrajectoryEvaluator
    from .exporters import load_jsonl
    from .metrics import compute_dataset_metrics, metrics_summary_md

    in_path = args.in_path
    if not os.path.exists(in_path):
        raise SystemExit(f"error: input file not found: '{in_path}'")

    trajectories = load_jsonl(in_path)
    if not trajectories:
        print(f"No trajectories found in '{in_path}'.")
        return 0

    evaluator = TrajectoryEvaluator()
    results = evaluator.evaluate_batch(trajectories)

    metrics = compute_dataset_metrics(trajectories, results)
    print(metrics_summary_md(metrics))

    if args.out:
        _write_eval_scores(results, args.out)
        print(f"Wrote {len(results)} score rows -> {args.out}")

    return 0


def _write_eval_scores(results: Sequence, out_path: str) -> None:
    import json

    with open(out_path, "w", encoding="utf-8") as fh:
        for res in results:
            fh.write(json.dumps(res.flat(), ensure_ascii=False) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns a process exit code (0 on success)."""
    if argv is None:
        argv = sys.argv[1:]

    # Pin the global force-mock switch early so subcommands inherit it.
    if _truthy(os.environ.get("AGENTSYNTH_FORCE_MOCK")):
        os.environ["AGENTSYNTH_FORCE_MOCK"] = "1"

    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "generate":
        return _cmd_generate(args)
    if args.command == "eval":
        return _cmd_eval(args)

    # Unreachable given argparse validation.
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
