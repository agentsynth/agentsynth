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
    sub = parser.add_subparsers(dest="command", metavar="{generate,eval,import,flywheel,bench}")

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

    imp = sub.add_parser(
        "import",
        help="Convert agent traces (OpenAI/Anthropic/OTel logs) into a trajectory dataset.",
        description="Read one trace per line, convert to trajectories, write JSONL.",
    )
    imp.add_argument("--in", "-i", dest="in_path", required=True, metavar="PATH")
    imp.add_argument("--out", "-o", default="./imported_trajectories.jsonl", metavar="PATH")
    imp.add_argument(
        "--trace-format",
        choices=("auto", "openai", "anthropic", "otel"),
        default="auto",
        help="Trace format (default: auto-detect per record).",
    )

    fly = sub.add_parser(
        "flywheel",
        help="Judge a dataset, mine its failures, and generate a verified patch dataset.",
        description="One turn of the loop: evaluate -> mine failures -> regenerate.",
    )
    fly.add_argument("--in", "-i", dest="in_path", required=True, metavar="PATH")
    fly.add_argument("--out", "-o", default="./flywheel_patch.jsonl", metavar="PATH")
    fly.add_argument("--k", type=int, default=20, help="Patch dataset size (default: 20).")
    fly.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Judge dimensions under this score count as failures (default: 0.7).",
    )

    bench = sub.add_parser(
        "bench",
        help="Run a scenario pack and report the outcome pass-rate.",
        description="Score a model (or a custom policy) against an outcome-checked pack.",
    )
    bench.add_argument(
        "--pack",
        default="packs/core_v1.yaml",
        metavar="PATH",
        help="Scenario pack file (default: packs/core_v1.yaml).",
    )
    bench.add_argument("--model", default=None, help="LiteLLM model id to drive the episodes.")
    bench.add_argument(
        "--policy",
        default=None,
        metavar="MODULE:FUNC",
        help="A custom policy instead of --model, e.g. mypkg.policies:my_policy.",
    )
    bench.add_argument("--seed", type=int, default=7)
    bench.add_argument(
        "--submit",
        default=None,
        metavar="URL",
        help="Hub base URL to submit the result to (e.g. https://api.agentsynth.tech).",
    )
    bench.add_argument(
        "--name", default=None, help="Label for the submission (defaults to --model)."
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


def _cmd_import(args: argparse.Namespace) -> int:
    from .exporters import to_jsonl
    from .importers import load_traces_jsonl

    if not os.path.exists(args.in_path):
        raise SystemExit(f"error: input file not found: '{args.in_path}'")
    trajectories = load_traces_jsonl(args.in_path, format=args.trace_format)
    if not trajectories:
        print(f"No traces recognized in '{args.in_path}'.")
        return 1
    to_jsonl(trajectories, args.out)
    sources: Dict[str, int] = {}
    for traj in trajectories:
        src = traj.metadata.get("source", "?")
        sources[src] = sources.get(src, 0) + 1
    breakdown = ", ".join(f"{s}={n}" for s, n in sorted(sources.items()))
    print(f"Imported {len(trajectories)} trajectories [{breakdown}] -> {args.out}")
    return 0


def _cmd_flywheel(args: argparse.Namespace) -> int:
    from .evaluator import TrajectoryEvaluator
    from .exporters import load_jsonl, to_jsonl
    from .mining import mine_judge_failures, recipe_from_failures
    from .pipelines import run_recipe

    if not os.path.exists(args.in_path):
        raise SystemExit(f"error: input file not found: '{args.in_path}'")
    trajectories = load_jsonl(args.in_path)
    if not trajectories:
        print(f"No trajectories found in '{args.in_path}'.")
        return 1

    results = TrajectoryEvaluator().evaluate_batch(trajectories)
    mined = mine_judge_failures(trajectories, results, threshold=args.threshold)
    print(mined.summary_md())
    if not mined.failures:
        print("Nothing under the threshold — no patch needed.")
        return 0

    patch = run_recipe(recipe_from_failures(mined, k=args.k))
    to_jsonl(patch.trajectories, args.out)
    print(
        f"Patch dataset: {len(patch.trajectories)} trajectories "
        f"(verified_rate={patch.metrics.get('verified_rate')}) -> {args.out}"
    )
    return 0


def _resolve_policy(args: argparse.Namespace):
    if args.policy:
        import importlib

        module_name, _, attr = args.policy.partition(":")
        if not attr:
            raise SystemExit("error: --policy needs the form module:function")
        return getattr(importlib.import_module(module_name), attr)
    if args.model:
        from .rl import llm_policy
        from .scale import CachingLLMClient

        client = CachingLLMClient(model=args.model)
        if not client.available:
            raise SystemExit(
                f"error: model '{args.model}' is not usable ({client.last_error}); "
                "set the provider key, or use --policy"
            )
        return llm_policy(client)
    raise SystemExit("error: pass --model <litellm id> or --policy module:function")


def _cmd_bench(args: argparse.Namespace) -> int:
    from .scenarios import load_scenarios, run_scenario_suite

    if not os.path.exists(args.pack):
        raise SystemExit(f"error: pack not found: '{args.pack}'")
    scenarios = load_scenarios(args.pack)
    policy = _resolve_policy(args)

    report = run_scenario_suite(policy, scenarios, seed=args.seed)
    for row in report.results:
        mark = "pass" if row["passed"] else "FAIL"
        print(f"[{mark}] {row['id']}  outcome={row['outcome_score']:.2f}")
    print(f"\n{report.passed}/{report.n} scenarios passed (pass_rate={report.pass_rate})")

    if args.submit:
        from . import __version__

        name = args.name or args.model or args.policy or "anonymous"
        payload = {
            "pack_id": os.path.splitext(os.path.basename(args.pack))[0],
            "model": name,
            "report": report.model_dump(),
            "client_version": __version__,
        }
        url = args.submit.rstrip("/") + "/v1/submissions"
        print(f"submitting to {url} ...")
        print(_post_json(url, payload))
    return 0


def _post_json(url: str, payload: Dict) -> str:
    import json
    from urllib import error, request

    from . import __version__

    data = json.dumps(payload).encode("utf-8")
    # Some WAFs (Cloudflare among them) block the default urllib user-agent, so
    # identify the client by name instead.
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"agentsynth/{__version__}",
    }
    req = request.Request(url, data=data, headers=headers)
    try:
        with request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", "replace")
    except error.HTTPError as exc:
        return f"submit failed: HTTP {exc.code} {exc.read().decode('utf-8', 'replace')[:200]}"
    except Exception as exc:
        return f"submit failed: {exc}"


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
    if args.command == "import":
        return _cmd_import(args)
    if args.command == "flywheel":
        return _cmd_flywheel(args)
    if args.command == "bench":
        return _cmd_bench(args)

    # Unreachable given argparse validation.
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
