"""argparse front-end for the `agentsynth` console script.

Heavy submodule imports live inside the command handlers so `--help` stays fast.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

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
    sub = parser.add_subparsers(
        dest="command", metavar="{generate,eval,import,flywheel,bench,pack}"
    )

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
    imp.add_argument(
        "--redact",
        action="store_true",
        help="Strip emails, keys, tokens, and phone-shaped numbers before export.",
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
        default="core_v1",
        metavar="PACK",
        help="Pack name, file, or URL. Names check packs/ locally, then the hub "
        "(default: core_v1).",
    )
    bench.add_argument("--model", default=None, help="LiteLLM model id to drive the episodes.")
    bench.add_argument(
        "--policy",
        default=None,
        metavar="MODULE:FUNC",
        help="A custom policy instead of --model, e.g. mypkg.policies:my_policy.",
    )
    bench.add_argument(
        "--compare",
        default=None,
        metavar="ITEMS",
        help="Comma-separated model ids and/or policy refs to run side by side, "
        "e.g. gpt-4o-mini,my_agent.py:solve.",
    )
    bench.add_argument("--seed", type=int, default=7)
    bench.add_argument(
        "--trials",
        type=int,
        default=1,
        metavar="K",
        help="Run the pack K times (seeds seed..seed+K-1) and score pass^K: a "
        "scenario counts only when every trial passes (default: 1).",
    )
    bench.add_argument(
        "--hub",
        default="https://api.agentsynth.tech",
        metavar="URL",
        help="Hub used for by-name packs and bare --submit.",
    )
    bench.add_argument(
        "--submit",
        nargs="?",
        const="",
        default=None,
        metavar="URL",
        help="Submit the result: to URL, or to the --hub when given without a value.",
    )
    bench.add_argument(
        "--name", default=None, help="Label for the submission (defaults to --model)."
    )
    bench.add_argument(
        "--json",
        dest="json_out",
        default=None,
        metavar="PATH",
        help="Write the full report as JSON — for CI gates and analysis.",
    )

    pack = sub.add_parser(
        "pack",
        help="Scaffold and validate scenario packs.",
        description="Create a pack skeleton, or run the gates a pack must pass to ship.",
    )
    pack_sub = pack.add_subparsers(
        dest="pack_command",
        metavar="{new,validate,teach,audit,export,contamination,verify-run}",
    )

    pack_new = pack_sub.add_parser("new", help="Write a pack skeleton plus its oracle next to it.")
    pack_new.add_argument("pack_id", metavar="ID", help="Pack id, e.g. devops_v1.")
    pack_new.add_argument(
        "--dir", default="packs", metavar="PATH", help="Where to put the files (default: packs)."
    )
    pack_new.add_argument(
        "--from-schema",
        dest="from_schema",
        default=None,
        metavar="FILE.sql",
        help="Generate a starter pack from a CREATE TABLE schema (validates out of the box).",
    )
    pack_new.add_argument(
        "--from-demo",
        dest="from_demo",
        default=None,
        metavar="FILE.json",
        help="Generate a pack from worked demonstrations (a JSON list of "
        "{task, schema, rows, actions, answer}); checkers are derived from each demo's "
        "end state, so the pack validates and audits clean out of the box.",
    )
    pack_new.add_argument("--seed", type=int, default=7)

    pack_val = pack_sub.add_parser(
        "validate", help="Check schema, oracle, determinism, and the lazy guard."
    )
    pack_val.add_argument("pack", metavar="PATH", help="Pack file to validate.")
    pack_val.add_argument(
        "--oracle",
        default=None,
        metavar="REF",
        help="Reference solution as module:fn or file.py:fn "
        "(default: <pack>_oracle.py:solve next to the pack).",
    )
    pack_val.add_argument("--seed", type=int, default=7)

    pack_teach = pack_sub.add_parser(
        "teach", help="Run the oracle through the pack and export gold trajectories."
    )
    pack_teach.add_argument("pack", metavar="PATH", help="Pack file to teach from.")
    pack_teach.add_argument(
        "--oracle", default=None, metavar="REF", help="Same form and default as validate."
    )
    pack_teach.add_argument("--out", default="gold.jsonl", metavar="PATH")
    pack_teach.add_argument("--seed", type=int, default=7)

    pack_audit = pack_sub.add_parser(
        "audit",
        help="Measure how gameable a pack's checkers are (reward-hacking resistance).",
    )
    pack_audit.add_argument("pack", metavar="PATH", help="Pack file to audit.")
    pack_audit.add_argument(
        "--min-robustness",
        dest="min_robustness",
        type=float,
        default=0.0,
        metavar="FRAC",
        help="Fail (exit 1) if fewer than this fraction of scenarios resist every "
        "trivial adversary. Default 0.0 (report only).",
    )
    pack_audit.add_argument("--seed", type=int, default=7)

    pack_export = pack_sub.add_parser(
        "export",
        help="Export a pack as an OpenEnv or Prime Intellect verifiers environment.",
    )
    pack_export.add_argument("pack", metavar="PATH", help="Pack file to export.")
    pack_export.add_argument(
        "--format",
        dest="fmt",
        choices=["verifiers", "openenv"],
        required=True,
        help="Target ecosystem: a Prime Intellect verifiers env, or an OpenEnv server.",
    )
    pack_export.add_argument(
        "--out",
        dest="out",
        default=None,
        metavar="DIR",
        help="Output folder (default: dist/<pack>-<format>).",
    )

    pack_contam = pack_sub.add_parser(
        "contamination",
        help="Audit a pack for benchmark contamination (canaries, corpus overlap).",
    )
    pack_contam.add_argument("pack", metavar="PATH", help="Pack file to audit.")
    pack_contam.add_argument(
        "--corpus",
        default=None,
        metavar="FILE",
        help="A training corpus (JSONL/JSON of records, or one document per line) to "
        "check each scenario's task against for overlap.",
    )
    pack_contam.add_argument(
        "--held-out",
        dest="held_out",
        default=None,
        metavar="OUT.yaml",
        help="Write contamination-resistant isomorphic siblings of every scenario here.",
    )
    pack_contam.add_argument("--threshold", type=float, default=0.8, metavar="FRAC")

    pack_verify = pack_sub.add_parser(
        "verify-run",
        help="Re-run a submission's manifest and confirm it reproduces (anti-fabrication).",
    )
    pack_verify.add_argument(
        "manifest", metavar="MANIFEST.json", help="A run manifest (or a bench --json report)."
    )
    pack_verify.add_argument(
        "--pack", default=None, metavar="PACK", help="Pack to check against (default: the "
        "manifest's pack_id, resolved locally or from the hub)."
    )
    pack_verify.add_argument("--policy", default=None, metavar="REF", help="Policy module:fn.")
    pack_verify.add_argument("--model", default=None, metavar="ID", help="Or a LiteLLM model id.")
    pack_verify.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        metavar="FRAC",
        help="Allowed pass-rate difference for a stochastic model (default 0, exact).",
    )
    pack_verify.add_argument("--hub", default="https://api.agentsynth.tech", metavar="URL")

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
    from .importers import load_traces_jsonl, redact_trajectory

    if not os.path.exists(args.in_path):
        raise SystemExit(f"error: input file not found: '{args.in_path}'")
    trajectories = load_traces_jsonl(args.in_path, format=args.trace_format)
    if not trajectories:
        print(f"No traces recognized in '{args.in_path}'.")
        return 1
    if args.redact:
        trajectories = [redact_trajectory(t) for t in trajectories]
        print("Redacted emails, keys, tokens, and phone-shaped numbers.")
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


def _load_policy_ref(ref: str):
    """Resolve `module:fn` or `path/to/file.py:fn` to a callable."""
    target, _, attr = ref.partition(":")
    if not attr:
        raise SystemExit("error: policy needs the form module:fn or file.py:fn")
    if target.endswith(".py") or os.path.sep in target:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_agentsynth_policy", target)
        if spec is None or spec.loader is None:
            raise SystemExit(f"error: cannot load '{target}'")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except FileNotFoundError:
            raise SystemExit(f"error: no such file: '{target}'")
    else:
        import importlib

        module = importlib.import_module(target)
    try:
        return getattr(module, attr)
    except AttributeError:
        raise SystemExit(f"error: '{target}' has no attribute '{attr}'")


def _policy_from_model(model: str):
    from .rl import llm_policy
    from .scale import CachingLLMClient

    client = CachingLLMClient(model=model)
    if not client.available:
        raise SystemExit(
            f"error: model '{model}' is not usable ({client.last_error}); "
            "set the provider key, or use a policy ref"
        )
    return llm_policy(client)


def _resolve_policy(args: argparse.Namespace):
    if args.policy:
        return _load_policy_ref(args.policy)
    if args.model:
        return _policy_from_model(args.model)
    raise SystemExit("error: pass --model <litellm id>, --policy module:function, or --compare a,b")


def _load_pack(pack: str, hub: str):
    """Resolve --pack: a local file, a name under packs/, a hub pack, or a URL."""
    import re

    from .scenarios import Scenario, load_scenarios

    if os.path.exists(pack):
        return load_scenarios(pack), os.path.splitext(os.path.basename(pack))[0]
    is_name = bool(re.fullmatch(r"[A-Za-z0-9_-]+", pack))
    local = os.path.join("packs", f"{pack}.yaml")
    if is_name and os.path.exists(local):
        return load_scenarios(local), pack
    if pack.startswith(("http://", "https://")):
        url, pack_id = pack, pack.rstrip("/").rsplit("/", 1)[-1]
    elif is_name:
        url, pack_id = f"{hub.rstrip('/')}/v1/packs/{pack}", pack
    else:
        raise SystemExit(f"error: pack not found: '{pack}'")
    payload = _get_json(url)
    if not isinstance(payload, list):
        raise SystemExit(f"error: {url} did not return a scenario pack")
    return [Scenario(**item) for item in payload], pack_id


def _failed_checks(row: Dict) -> str:
    names = [c.get("name", "?") for c in row.get("checks", []) if not c.get("passed")]
    return f"  failed: {', '.join(names)}" if names else ""


def _run_trials(policy, scenarios, seed: int, trials: int):
    """Aggregate K trials into one pass^K report. Returns (report, pass1_avg)."""
    from .scenarios import ScenarioReport, run_scenario_suite

    reports = [run_scenario_suite(policy, scenarios, seed=seed + t) for t in range(trials)]
    if trials == 1:
        return reports[0], None
    order = [row["id"] for row in reports[0].results]
    wins: Dict[str, int] = {sid: 0 for sid in order}
    scores: Dict[str, float] = {sid: 1.0 for sid in order}
    for rep in reports:
        for row in rep.results:
            wins[row["id"]] += 1 if row["passed"] else 0
            scores[row["id"]] = min(scores[row["id"]], float(row["outcome_score"]))
    agg = [
        {"id": sid, "passed": wins[sid] == trials, "outcome_score": scores[sid]} for sid in order
    ]
    passed = sum(1 for row in agg if row["passed"])
    report = ScenarioReport(
        n=len(agg), passed=passed, pass_rate=round(passed / (len(agg) or 1), 4), results=agg
    )
    pass1_avg = round(sum(rep.pass_rate for rep in reports) / len(reports), 4)
    return report, pass1_avg


def _cmd_bench_compare(args: argparse.Namespace, scenarios, pack_id: str) -> int:
    items = [item.strip() for item in args.compare.split(",") if item.strip()]
    if len(items) < 2:
        raise SystemExit("error: --compare needs at least two comma-separated items")

    trials = max(1, int(args.trials))
    runs = []
    for item in items:
        policy = _load_policy_ref(item) if ":" in item else _policy_from_model(item)
        report, pass1_avg = _run_trials(policy, scenarios, args.seed, trials)
        runs.append({"name": item, "pass1_avg": pass1_avg, "report": report})

    def disp(name: str) -> str:
        return name if len(name) <= 22 else "…" + name[-21:]

    id_width = max(len(r["id"]) for r in runs[0]["report"].results)
    header = "scenario".ljust(id_width) + "".join(f"  {disp(run['name']):>22}" for run in runs)
    print(header)
    by_name = {
        run["name"]: {row["id"]: row["passed"] for row in run["report"].results} for run in runs
    }
    for row in runs[0]["report"].results:
        sid = row["id"]
        cells = "".join(f"  {'✓' if by_name[run['name']][sid] else '✗':>22}" for run in runs)
        print(sid.ljust(id_width) + cells)
    label = f"pass^{trials}" if trials > 1 else "pass rate"
    print(label.ljust(id_width) + "".join(f"  {run['report'].pass_rate:>22.0%}" for run in runs))
    if trials > 1:
        print(
            "pass^1 avg".ljust(id_width) + "".join(f"  {run['pass1_avg']:>22.0%}" for run in runs)
        )

    if args.json_out:
        import json

        blob = {
            "pack_id": pack_id,
            "seed": args.seed,
            "trials": trials,
            "compare": [
                {"name": run["name"], "pass1_avg": run["pass1_avg"], **run["report"].model_dump()}
                for run in runs
            ],
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=2)
        print(f"report -> {args.json_out}")

    if args.submit is not None:
        from . import __version__

        url = (args.submit or args.hub).rstrip("/") + "/v1/submissions"
        for run in runs:
            payload = {
                "pack_id": pack_id,
                "model": run["name"][:200],
                "report": run["report"].model_dump(),
                "client_version": __version__,
            }
            print(f"submitting {run['name']} to {url} ...")
            print(_post_json(url, payload))
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from .scenarios import ScenarioReport, run_scenario_suite

    scenarios, pack_id = _load_pack(args.pack, args.hub)

    if args.compare:
        if args.model or args.policy:
            raise SystemExit("error: --compare replaces --model/--policy")
        return _cmd_bench_compare(args, scenarios, pack_id)

    policy = _resolve_policy(args)

    import time

    trials = max(1, int(args.trials))
    t0 = time.perf_counter()
    reports = [run_scenario_suite(policy, scenarios, seed=args.seed + t) for t in range(trials)]
    elapsed = time.perf_counter() - t0
    pass1_avg = None
    rel = None

    if trials == 1:
        report = reports[0]
        for row in report.results:
            mark = "pass" if row["passed"] else "FAIL"
            print(f"[{mark}] {row['id']}  outcome={row['outcome_score']:.2f}{_failed_checks(row)}")
        print(f"\n{report.passed}/{report.n} scenarios passed (pass_rate={report.pass_rate})")
    else:
        # pass^k after tau-bench: reliability means passing every one of k trials.
        order = [row["id"] for row in reports[0].results]
        wins: Dict[str, List[bool]] = {sid: [] for sid in order}
        scores: Dict[str, float] = {sid: 1.0 for sid in order}
        for rep in reports:
            for row in rep.results:
                wins[row["id"]].append(bool(row["passed"]))
                scores[row["id"]] = min(scores[row["id"]], float(row["outcome_score"]))
        agg = []
        for sid in order:
            k_pass = sum(wins[sid])
            all_pass = k_pass == trials
            mark = "pass" if all_pass else ("FLAKY" if k_pass else "FAIL")
            print(f"[{mark}] {sid}  {k_pass}/{trials} trials  worst_outcome={scores[sid]:.2f}")
            agg.append({"id": sid, "passed": all_pass, "outcome_score": scores[sid]})
        passed = sum(1 for row in agg if row["passed"])
        n = len(agg) or 1
        avg = sum(rep.pass_rate for rep in reports) / len(reports)
        pass1_avg = round(avg, 4)
        report = ScenarioReport(
            n=len(agg), passed=passed, pass_rate=round(passed / n, 4), results=agg
        )
        print(f"\npass^1 (avg of {trials} trials): {avg:.0%}")
        print(f"pass^{trials} (all trials must pass): {passed}/{len(agg)} ({report.pass_rate:.0%})")

        from .reliability import reliability_report

        rel = reliability_report(wins, trials)
        print()
        print(rel.summary_md())
        runs_total = report.n * trials
        if runs_total:
            print(
                f"throughput: {runs_total} runs in {elapsed:.1f}s "
                f"({elapsed / runs_total:.3f}s/run)"
            )

    from .provenance import run_manifest

    bench_name = args.name or args.model or args.policy or "anonymous"
    manifest = run_manifest(
        pack_id, scenarios, report, model=bench_name, seed=args.seed, trials=trials
    )
    print(
        f"\nrun_hash {manifest['run_hash']} (pack {manifest['pack_fingerprint']}) — "
        "reproducible with `agentsynth pack verify-run`"
    )

    if args.submit is None and report.passed > 0:
        print("→ add --submit to put this run on the live leaderboard (agentsynth.tech)")

    if args.json_out:
        import json

        blob = {
            "pack_id": pack_id,
            "name": bench_name,
            "seed": args.seed,
            "trials": trials,
            "pass1_avg": pass1_avg,
            "elapsed_s": round(elapsed, 3),
            "reliability": rel.model_dump() if rel is not None else None,
            "manifest": manifest,
            **report.model_dump(),
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=2)
        print(f"report -> {args.json_out}")

    if args.submit is not None:
        from . import __version__

        if trials > 1:
            print(f"submitting the pass^{trials} numbers (reliability-adjusted)")
        payload = {
            "pack_id": pack_id,
            "model": bench_name,
            "report": report.model_dump(),
            "manifest": manifest,
            "client_version": __version__,
        }
        url = (args.submit or args.hub).rstrip("/") + "/v1/submissions"
        print(f"submitting to {url} ...")
        print(_post_json(url, payload))
    return 0


_PACK_TEMPLATE = """\
# {pack_id} — outcome-checked scenarios over a writable SQL world.
# A scenario passes only when every checker holds against the world's end state.
# Validate before opening a PR:
#   agentsynth pack validate {dir}/{pack_id}.yaml
- id: close-ticket
  task: Close ticket 3 in the tickets table, then confirm what you did.
  environment:
    type: sql
    schema: CREATE TABLE tickets (id INTEGER PRIMARY KEY, status TEXT)
    table: tickets
    rows:
      - [3, "open"]
      - [4, "open"]
  checkers:
    - kind: sql
      query: SELECT status FROM tickets WHERE id=3
      equals: [["closed"]]
    - kind: sql
      query: SELECT status FROM tickets WHERE id=4
      equals: [["open"]]
    - kind: answer
      any_of: ["closed"]

- id: count-open-tickets
  task: How many tickets are still open? Answer with the number.
  environment:
    type: sql
    schema: CREATE TABLE tickets (id INTEGER PRIMARY KEY, status TEXT)
    table: tickets
    rows:
      - [3, "open"]
      - [4, "closed"]
      - [5, "open"]
  checkers:
    - kind: called_tool
      name: sql_query
    - kind: answer
      any_of: ["2", "two"]

- id: reopen-refused
  task: A user asks to reopen ticket 4, but policy forbids reopening closed tickets.
    Check its status and answer accordingly without changing anything.
  environment:
    type: sql
    schema: CREATE TABLE tickets (id INTEGER PRIMARY KEY, status TEXT)
    table: tickets
    rows:
      - [4, "closed"]
  checkers:
    - kind: sql
      query: SELECT status FROM tickets WHERE id=4
      equals: [["closed"]]
    - kind: called_tool
      name: sql_query
    - kind: answer
      any_of: ["closed", "cannot", "can't"]
"""

_ORACLE_TEMPLATE = '''\
"""Reference solution for {pack_id} — the ceiling the leaderboard chases.

`agentsynth pack validate` runs this and requires a perfect score, which keeps
the pack honest, and `agentsynth pack teach` exports its episodes as gold
trajectories. Work like a careful operator: look first, act, read it back.
"""

# inspect -> act -> verify, one statement per step. Read-only tasks just read.
_PLAN = {{
    "close-ticket": [
        "SELECT id, status FROM tickets WHERE id IN (3, 4)",
        "UPDATE tickets SET status='closed' WHERE id=3",
        "SELECT id, status FROM tickets WHERE id IN (3, 4)",
    ],
    "count-open-tickets": [
        "SELECT COUNT(*) FROM tickets WHERE status='open'",
    ],
    "reopen-refused": [
        "SELECT id, status FROM tickets WHERE id=4",
    ],
}}

_ANSWER = {{
    "close-ticket": "Ticket 3 is closed and verified; ticket 4 left open.",
    "count-open-tickets": "2 tickets are still open.",
    "reopen-refused": "Ticket 4 is closed, so I cannot reopen it.",
}}


def solve(observation, gym):
    sid = gym.scenario.id if gym.scenario is not None else ""
    plan = _PLAN.get(sid, [])
    if gym.step_count < len(plan):
        return {{"tool_name": "sql_query", "arguments": {{"query": plan[gym.step_count]}}}}
    return {{"answer": _ANSWER.get(sid, "Done.")}}
'''


def _parse_create_table(sql: str):
    """First CREATE TABLE in a schema → (table, [(col, TYPE, raw_upper)], create_sql)."""
    import re

    m = re.search(
        r"create\s+table\s+(?:if\s+not\s+exists\s+)?[\"`\[]?(\w+)[\"`\]]?\s*\((.*)\)",
        sql,
        re.I | re.S,
    )
    if not m:
        raise SystemExit("error: no CREATE TABLE statement found in the schema file")
    table, body = m.group(1), m.group(2)
    parts, depth, cur = [], 0, ""
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)

    skip = ("primary", "foreign", "unique", "check", "constraint")
    cols = []
    for part in parts:
        toks = part.strip().split()
        if not toks or toks[0].lower() in skip:
            continue
        name = toks[0].strip('"`[]')
        ctype = toks[1].upper() if len(toks) > 1 else "TEXT"
        cols.append((name, ctype, part.upper()))
    if not cols:
        raise SystemExit("error: could not read any columns from the schema")
    return table, cols, m.group(0).strip()


def _is_text(ctype: str) -> bool:
    return any(t in ctype for t in ("TEXT", "CHAR", "CLOB", "STRING"))


def _seed_value(name: str, ctype: str, idx: int):
    """A per-row value, unique across rows so PK/UNIQUE columns never collide."""
    if "INT" in ctype:
        return idx
    if any(t in ctype for t in ("REAL", "FLOA", "DOUB", "NUM", "DEC")):
        return float(idx)
    return f"{name}{idx}"


# Text columns whose name suggests a state make the friendliest update target.
_STATE_HINTS = ("status", "state", "stage", "phase", "priority", "kind", "type")


def _pack_from_schema(schema_sql: str, pack_id: str):
    """A starter pack + oracle generated from a CREATE TABLE, guaranteed to validate.

    World, checkers, and oracle are emitted together from templates, so the three
    stay consistent: the oracle solves every scenario and a do-nothing policy fails.
    """
    import json

    import yaml

    table, cols, create_sql = _parse_create_table(schema_sql)

    # an integer key for stable row ids (avoids SQLite text/number affinity surprises)
    id_col = next(
        (c for c, t, raw in cols if "INT" in t and "PRIMARY KEY" in raw),
        next((c for c, t, _ in cols if "INT" in t), None),
    )
    if id_col is None:
        raise SystemExit(
            f"error: need an integer key column to template row ids for '{table}' — "
            f"run `agentsynth pack new {pack_id}` and edit by hand"
        )
    # a writable, non-unique text column to mutate and count on
    text_cols = [
        c
        for c, t, raw in cols
        if c != id_col and _is_text(t) and "PRIMARY KEY" not in raw and "UNIQUE" not in raw
    ]
    if not text_cols:
        raise SystemExit(
            f"error: need a non-unique text column to template scenarios for '{table}' — "
            f"run `agentsynth pack new {pack_id}` and edit by hand"
        )
    text_col = next((c for c in text_cols if c.lower() in _STATE_HINTS), text_cols[0])

    def row(idx, text_val):
        out = []
        for name, ctype, _ in cols:
            if name == id_col:
                out.append(idx)
            elif name == text_col:
                out.append(text_val)
            else:
                out.append(_seed_value(name, ctype, idx))
        return out

    env = {"type": "sql", "schema": create_sql, "table": table}
    scenarios = [
        {
            "id": "set-field",
            "task": f"Set {text_col} of {table} row 1 to 'done'. Leave the other rows unchanged.",
            "environment": {**env, "rows": [row(1, "alpha"), row(2, "beta"), row(3, "gamma")]},
            "checkers": [
                {
                    "kind": "sql",
                    "query": f"SELECT {text_col} FROM {table} WHERE {id_col}=1",
                    "equals": [["done"]],
                },
                {
                    "kind": "sql",
                    "query": f"SELECT {text_col} FROM {table} WHERE {id_col}=2",
                    "equals": [["beta"]],
                },
            ],
        },
        {
            "id": "count-rows",
            "task": f"How many rows in {table} have {text_col} = 'alpha'? Answer with the number.",
            "environment": {**env, "rows": [row(1, "alpha"), row(2, "beta"), row(3, "alpha")]},
            "checkers": [
                {"kind": "called_tool", "name": "sql_query"},
                {"kind": "answer", "any_of": ["2", "two"]},
            ],
        },
        {
            "id": "refuse-delete",
            "task": (
                f"A request asks to delete {table} row 1, but policy forbids deletes here. "
                "Check it exists and refuse, without changing anything."
            ),
            "environment": {**env, "rows": [row(1, "alpha"), row(2, "beta"), row(3, "gamma")]},
            "checkers": [
                {
                    "kind": "sql",
                    "query": f"SELECT COUNT(*) FROM {table} WHERE {id_col}=1",
                    "equals": [[1]],
                },
                {"kind": "called_tool", "name": "sql_query"},
                {"kind": "answer", "any_of": ["cannot", "policy", "refuse"]},
            ],
        },
    ]
    header = (
        f"# {pack_id} — generated from the {table} schema. A starter, not a finished pack:\n"
        f"# rename the scenarios to your real tasks, then re-run pack validate.\n"
    )
    pack_yaml = header + yaml.safe_dump(scenarios, sort_keys=False, allow_unicode=True)

    plan = {
        "set-field": [f"UPDATE {table} SET {text_col}='done' WHERE {id_col}=1"],
        "count-rows": [f"SELECT COUNT(*) FROM {table} WHERE {text_col}='alpha'"],
        "refuse-delete": [f"SELECT * FROM {table} WHERE {id_col}=1"],
    }
    answer = {
        "set-field": "Set row 1 to done; the other rows are unchanged.",
        "count-rows": "2 rows match.",
        "refuse-delete": "I cannot delete it — policy forbids deletes here.",
    }
    oracle_py = (
        f'"""Auto-generated reference solution for {pack_id} (from the {table} schema)."""\n\n'
        f"_PLAN = {json.dumps(plan, indent=4)}\n\n"
        f"_ANSWER = {json.dumps(answer, indent=4)}\n\n\n"
        "def solve(observation, gym):\n"
        '    sid = gym.scenario.id if gym.scenario is not None else ""\n'
        "    plan = _PLAN.get(sid, [])\n"
        "    if gym.step_count < len(plan):\n"
        '        return {"tool_name": "sql_query", "arguments": {"query": plan[gym.step_count]}}\n'
        '    return {"answer": _ANSWER.get(sid, "Done.")}\n'
    )
    return pack_yaml, oracle_py


def _cmd_pack_new(args: argparse.Namespace) -> int:
    os.makedirs(args.dir, exist_ok=True)
    pack_path = os.path.join(args.dir, f"{args.pack_id}.yaml")
    oracle_path = os.path.join(args.dir, f"{args.pack_id}_oracle.py")
    for path in (pack_path, oracle_path):
        if os.path.exists(path):
            raise SystemExit(f"error: refusing to overwrite '{path}'")

    from_schema = getattr(args, "from_schema", None)
    from_demo = getattr(args, "from_demo", None)
    if from_schema and from_demo:
        raise SystemExit("error: pass --from-schema or --from-demo, not both")
    if from_schema:
        if not os.path.exists(from_schema):
            raise SystemExit(f"error: schema file not found: '{from_schema}'")
        with open(from_schema, encoding="utf-8") as fh:
            pack_text, oracle_text = _pack_from_schema(fh.read(), args.pack_id)
    elif from_demo:
        import json

        from .synth import pack_from_demonstrations

        if not os.path.exists(from_demo):
            raise SystemExit(f"error: demo file not found: '{from_demo}'")
        with open(from_demo, encoding="utf-8") as fh:
            demos = json.load(fh)
        if isinstance(demos, dict):
            demos = [demos]
        try:
            pack_text, oracle_text = pack_from_demonstrations(demos, args.pack_id)
        except (KeyError, ValueError) as exc:
            raise SystemExit(f"error: could not build pack from demos: {exc}")
    else:
        pack_text = _PACK_TEMPLATE.format(pack_id=args.pack_id, dir=args.dir)
        oracle_text = _ORACLE_TEMPLATE.format(pack_id=args.pack_id)

    with open(pack_path, "w", encoding="utf-8") as fh:
        fh.write(pack_text)
    with open(oracle_path, "w", encoding="utf-8") as fh:
        fh.write(oracle_text)

    print(f"wrote {pack_path}")
    print(f"wrote {oracle_path}")

    if from_schema or from_demo:
        # prove the generated pack already passes the gate
        from .robustness import audit_pack
        from .scenarios import load_scenarios, run_scenario_suite

        scenarios = load_scenarios(pack_path)
        oracle = _load_policy_ref(f"{oracle_path}:solve")
        ok = run_scenario_suite(oracle, scenarios, seed=args.seed)
        lazy = run_scenario_suite(_lazy_policy, scenarios, seed=args.seed)
        audit = audit_pack(scenarios, seed=args.seed)
        print(
            f"self-check: oracle {ok.passed}/{ok.n}, do-nothing {lazy.passed}/{lazy.n}, "
            f"robustness {audit.robustness_score:.0%} — "
            + ("PACK OK" if ok.passed == ok.n and lazy.pass_rate < 0.5 else "review needed")
        )
        print("Rename the scenarios to your real tasks, keep the oracle solving them, then:")
    else:
        print("Replace the sample scenarios with your domain, keep the oracle solving them,")
    print(f"  agentsynth pack validate {pack_path}")
    return 0


def _lazy_policy(observation, gym):
    """What a pack must not reward: talk, no action."""
    return {"answer": "all done"}


def _oracle_ref(pack_path: str, oracle_arg: Optional[str]) -> str:
    """Explicit --oracle, or `<pack>_oracle.py:solve` sitting next to the pack."""
    if oracle_arg:
        return oracle_arg
    default = f"{os.path.splitext(pack_path)[0]}_oracle.py"
    if not os.path.exists(default):
        raise SystemExit(
            f"error: no oracle: expected '{default}' next to the pack, or pass --oracle"
        )
    return f"{default}:solve"


def _cmd_pack_validate(args: argparse.Namespace) -> int:
    from .scenarios import load_scenarios, run_scenario_suite

    if not os.path.exists(args.pack):
        raise SystemExit(f"error: pack not found: '{args.pack}'")
    try:
        scenarios = load_scenarios(args.pack)
    except Exception as exc:
        print(f"[fail] schema: {exc}")
        return 1

    problems = []
    ids = [s.id for s in scenarios]
    if len(scenarios) < 3:
        problems.append(f"a pack needs at least 3 scenarios (found {len(scenarios)})")
    if len(set(ids)) != len(ids):
        problems.append("scenario ids must be unique")
    if any(not (s.task or "").strip() for s in scenarios):
        problems.append("every scenario needs a task")
    if problems:
        for p in problems:
            print(f"[fail] schema: {p}")
        return 1
    print(f"[ok] schema — {len(scenarios)} scenarios, unique ids")

    oracle = _load_policy_ref(_oracle_ref(args.pack, args.oracle))

    first = run_scenario_suite(oracle, scenarios, seed=args.seed)
    if first.passed != first.n:
        for row in first.results:
            if not row["passed"]:
                print(f"[fail] oracle: {row['id']} (outcome={row['outcome_score']:.2f})")
        print(f"[fail] oracle passes {first.passed}/{first.n} — every scenario must be solvable")
        return 1
    print(f"[ok] oracle passes {first.n}/{first.n}")

    second = run_scenario_suite(oracle, scenarios, seed=args.seed)
    if [r["passed"] for r in first.results] != [r["passed"] for r in second.results]:
        print("[fail] determinism: the same seed gave different results across reruns")
        return 1
    print("[ok] deterministic across reruns")

    lazy = run_scenario_suite(_lazy_policy, scenarios, seed=args.seed)
    if lazy.pass_rate >= 0.5:
        print(
            f"[fail] lazy guard: a do-nothing policy passes {lazy.passed}/{lazy.n} — "
            "checkers must assert on the world, not the words"
        )
        return 1
    print(f"[ok] lazy guard — do-nothing policy passes {lazy.passed}/{lazy.n}")

    from .robustness import audit_pack

    audit = audit_pack(scenarios, seed=args.seed)
    if audit.robustness_score >= 1.0:
        print(f"[ok] robustness — {audit.robust}/{audit.n} resist every trivial adversary")
    else:
        weak = ", ".join(r.scenario_id for r in audit.rows if not r.robust) or "—"
        print(
            f"[warn] robustness {audit.robustness_score:.0%} — gameable: {weak}. "
            "Run `agentsynth pack audit` for the breakdown."
        )

    print("PACK OK")
    print(
        "→ packs this clean belong in the public registry: open a PR adding it to "
        "packs/ (see packs/README.md) and it gets its own live leaderboard"
    )
    return 0


def _cmd_pack_teach(args: argparse.Namespace) -> int:
    from .exporters import to_jsonl
    from .rl import AgentGym
    from .scenarios import load_scenarios

    if not os.path.exists(args.pack):
        raise SystemExit(f"error: pack not found: '{args.pack}'")
    scenarios = load_scenarios(args.pack)
    oracle = _load_policy_ref(_oracle_ref(args.pack, args.oracle))

    trajectories = []
    rewards = []
    for scenario in scenarios:
        gym = AgentGym.from_scenario(scenario, seed=args.seed)
        try:
            episode = gym.rollout(oracle)
        finally:
            gym.close()
        outcome = episode.info.get("outcome", {})
        if outcome.get("score", 0.0) < 1.0:
            print(
                f"[fail] {scenario.id}: oracle scored {outcome.get('score', 0.0):.2f} — "
                "gold data has to pass every checker"
            )
            return 1
        trajectories.append(episode.trajectory)
        rewards.append(episode.total_reward)

    to_jsonl(trajectories, args.out)
    avg = sum(rewards) / (len(rewards) or 1)
    print(f"{len(trajectories)} gold trajectories (avg reward {avg:.3f}) -> {args.out}")
    return 0


def _cmd_pack_audit(args: argparse.Namespace) -> int:
    from .robustness import audit_pack
    from .scenarios import load_scenarios

    if not os.path.exists(args.pack):
        raise SystemExit(f"error: pack not found: '{args.pack}'")
    scenarios = load_scenarios(args.pack)
    report = audit_pack(scenarios, seed=args.seed)
    print(report.summary_md())
    if report.robustness_score < args.min_robustness:
        print(
            f"\n[fail] robustness {report.robustness_score:.0%} is below the "
            f"{args.min_robustness:.0%} floor"
        )
        return 1
    return 0


def _cmd_pack_export(args: argparse.Namespace) -> int:
    from .pack_export import export_pack

    if not os.path.exists(args.pack):
        raise SystemExit(f"error: pack not found: '{args.pack}'")
    pack_id = os.path.splitext(os.path.basename(args.pack))[0]
    out = args.out or os.path.join("dist", f"{pack_id}-{args.fmt}")
    try:
        paths = export_pack(args.pack, args.fmt, out)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    for path in paths:
        print(f"wrote {path}")
    if args.fmt == "verifiers":
        print(f"→ a verifiers environment is in {out}/ — `pip install verifiers` to run it,")
        print("  then push the folder to the Prime Intellect Environments Hub")
    else:
        print(f"→ an OpenEnv environment is in {out}/ — needs agentsynth-ai[rl] (Python 3.10+)")
    return 0


def _load_corpus(path: str) -> List[str]:
    """Read a contamination corpus: a JSON array, JSONL, or one document per line."""
    import json

    if not os.path.exists(path):
        raise SystemExit(f"error: corpus not found: '{path}'")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()

    records: List[Any]
    stripped = text.strip()
    if stripped.startswith("["):
        try:
            records = json.loads(stripped)
        except json.JSONDecodeError:
            records = []
    else:
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append(line)

    out = []
    for record in records:
        if isinstance(record, dict):
            out.append(
                str(
                    record.get("query")
                    or record.get("task")
                    or record.get("text")
                    or record.get("prompt")
                    or ""
                )
            )
        else:
            out.append(str(record))
    return [t for t in out if t]


def _cmd_pack_contamination(args: argparse.Namespace) -> int:
    from .contamination import contamination_report, held_out_pack
    from .scenarios import load_scenarios, save_scenarios

    if not os.path.exists(args.pack):
        raise SystemExit(f"error: pack not found: '{args.pack}'")
    scenarios = load_scenarios(args.pack)
    corpus = _load_corpus(args.corpus) if args.corpus else None
    report = contamination_report(scenarios, corpus=corpus, threshold=args.threshold)
    print(report.summary_md())
    if args.held_out:
        save_scenarios(held_out_pack(scenarios), args.held_out)
        print(f"\nwrote held-out siblings -> {args.held_out}")
    return 1 if report.flagged else 0


def _cmd_pack_verify_run(args: argparse.Namespace) -> int:
    import json

    from .provenance import verify_run

    if not os.path.exists(args.manifest):
        raise SystemExit(f"error: manifest not found: '{args.manifest}'")
    with open(args.manifest, encoding="utf-8") as fh:
        data = json.load(fh)
    # accept a raw manifest or a `bench --json` report that embeds one
    if isinstance(data, dict) and "run_hash" not in data and isinstance(data.get("manifest"), dict):
        manifest = data["manifest"]
    else:
        manifest = data
    if not isinstance(manifest, dict) or "run_hash" not in manifest:
        raise SystemExit("error: no run manifest in that file")

    pack_ref = args.pack or manifest.get("pack_id")
    if not pack_ref:
        raise SystemExit("error: no pack — pass --pack or use a manifest carrying a pack_id")
    scenarios, _ = _load_pack(pack_ref, args.hub)
    policy = _resolve_policy(args)

    result = verify_run(manifest, scenarios, policy, tolerance=args.tolerance)
    print(f"pack intact: {result['pack_intact']}")
    print(
        f"pass_rate:   expected {result['expected_pass_rate']}, got "
        f"{result['actual_pass_rate']} (delta {result['pass_rate_delta']})"
    )
    print(f"run_hash:    expected {result['expected_hash']}, got {result['actual_hash']}")
    if not result["pack_intact"]:
        print("NOT REPRODUCED — the pack changed since the run (fingerprint mismatch)")
    elif result["reproduced"]:
        print("VERIFIED — exact reproduction")
    elif result["within_tolerance"]:
        print(f"VERIFIED — within tolerance {args.tolerance}")
    else:
        print("NOT REPRODUCED — the policy did not reproduce the claimed result")
    ok = result["pack_intact"] and (result["reproduced"] or result["within_tolerance"])
    return 0 if ok else 1


def _cmd_pack(args: argparse.Namespace) -> int:
    if args.pack_command == "new":
        return _cmd_pack_new(args)
    if args.pack_command == "validate":
        return _cmd_pack_validate(args)
    if args.pack_command == "teach":
        return _cmd_pack_teach(args)
    if args.pack_command == "audit":
        return _cmd_pack_audit(args)
    if args.pack_command == "export":
        return _cmd_pack_export(args)
    if args.pack_command == "contamination":
        return _cmd_pack_contamination(args)
    if args.pack_command == "verify-run":
        return _cmd_pack_verify_run(args)
    print(
        "usage: agentsynth pack "
        "{new,validate,teach,audit,export,contamination,verify-run} ..."
    )
    return 1


def _get_json(url: str):
    import json
    from urllib import error, request

    from . import __version__

    req = request.Request(url, headers={"User-Agent": f"agentsynth/{__version__}"})
    try:
        with request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except error.HTTPError as exc:
        raise SystemExit(f"error: GET {url} -> HTTP {exc.code}")
    except Exception as exc:
        raise SystemExit(f"error: GET {url} failed: {exc}")


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
    if args.command == "pack":
        return _cmd_pack(args)

    # Unreachable given argparse validation.
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
