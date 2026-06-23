"""AgentSynth: generate synthetic agent trajectories and score them with an
LLM-as-Judge loop. Runs offline against a mock by default.

    >>> from agentsynth import AgentTrajectoryGenerator, TrajectoryEvaluator
    >>> gen = AgentTrajectoryGenerator()
    >>> traj = gen.generate("What's the weather in Paris?")
    >>> ev = TrajectoryEvaluator().evaluate(traj)
    >>> ev.overall, ev.passed
"""

from __future__ import annotations

__version__ = "0.7.4"

from .adapters import action_from_openai_tool_call, to_openai_tools
from .benchmarks import (
    BUILTIN_CASES,
    BenchmarkCase,
    BenchmarkReport,
    agentsynth_model,
    compare_models,
    prompted_model,
    report_table_md,
    run_benchmark,
)
from .contamination import (
    ContaminationReport,
    canary_for,
    contamination_report,
    held_out_pack,
)
from .dedup import DedupResult, decontaminate, dedup_trajectories
from .environments import (
    BrowserEnvironment,
    CompositeEnvironment,
    Environment,
    MCPEnvironment,
    PythonSandbox,
    RestEnvironment,
    SQLEnvironment,
)
from .evaluator import TrajectoryEvaluator
from .evolve import evolve_queries
from .exporters import (
    load_jsonl,
    save_dataset,
    to_adp,
    to_jsonl,
    to_sharegpt,
)
from .generator import AgentTrajectoryGenerator
from .hub import dataset_card, prepare_dataset_dir, push_dataset
from .importers import (
    import_traces,
    load_traces_jsonl,
    redact_text,
    redact_trajectory,
    trajectory_from_messages,
)
from .metrics import compute_dataset_metrics, diversity_score, run_report_md
from .mining import (
    FailureReport,
    mine_failures,
    mine_judge_failures,
    recipe_from_failures,
    targeted_queries,
)
from .pack_export import (
    export_pack,
    reward_from_messages,
    scenario_reward,
)
from .pipelines import Recipe, RunResult, load_recipe, make_environment, run_recipe
from .preferences import (
    PreferencePair,
    build_preference_pairs,
    load_dpo_jsonl,
    to_dpo_jsonl,
)
from .reliability import ReliabilityReport, reliability_report, wilson_interval
from .rl import AgentGym, make_reward_fn
from .robustness import (
    RobustnessReport,
    audit_pack,
    ipt_report,
    perturb_scenario,
)
from .scale import BudgetExceeded, CachingLLMClient, CostMeter, run_resumable
from .scenarios import (
    AnswerContains,
    CalledTool,
    HttpCheck,
    Scenario,
    SqlCheck,
    load_scenarios,
    run_scenario_suite,
    save_scenarios,
)
from .schemas import (
    DEFAULT_RUBRIC_WEIGHTS,
    RUBRIC_DIMENSIONS,
    EvalResult,
    RubricScores,
    ToolSpec,
    Trajectory,
    TrajectoryStep,
)
from .synth import pack_from_demonstrations, scenario_from_demonstration
from .tasks import SEED_TASKS, SeedTask, sample_tasks
from .training import build_dpo_dataset, build_sft_dataset, to_dpo_records, to_sft_records
from .utils import (
    DEFAULT_TOOL_CATALOG,
    LLMClient,
    PythonREPL,
    default_tool_catalog,
    parse_tool_catalog,
)
from .verification import (
    RUBRIC_PRESETS,
    EnsembleEvaluator,
    ExecutionVerifier,
    ExpectedAnswerVerifier,
    LearnedVerifier,
    SafetyVerifier,
    ToolArgVerifier,
    VerificationResult,
    Verifier,
    batch_verify,
    get_rubric,
    rubric_names,
    train_learned_verifier,
    verify_trajectory,
)

__all__ = [
    "__version__",
    # schemas
    "Trajectory",
    "TrajectoryStep",
    "ToolSpec",
    "RubricScores",
    "EvalResult",
    "RUBRIC_DIMENSIONS",
    "DEFAULT_RUBRIC_WEIGHTS",
    # utils
    "parse_tool_catalog",
    "default_tool_catalog",
    "DEFAULT_TOOL_CATALOG",
    "PythonREPL",
    "LLMClient",
    # core
    "AgentTrajectoryGenerator",
    "TrajectoryEvaluator",
    # metrics
    "compute_dataset_metrics",
    "diversity_score",
    # exporters
    "to_jsonl",
    "to_sharegpt",
    "to_adp",
    "save_dataset",
    "load_jsonl",
    # environments
    "Environment",
    "CompositeEnvironment",
    "SQLEnvironment",
    "PythonSandbox",
    "MCPEnvironment",
    "BrowserEnvironment",
    "RestEnvironment",
    # rl
    "AgentGym",
    "make_reward_fn",
    # adapters
    "to_openai_tools",
    "action_from_openai_tool_call",
    # scale
    "CachingLLMClient",
    "CostMeter",
    "BudgetExceeded",
    "run_resumable",
    # scenarios
    "Scenario",
    "SqlCheck",
    "HttpCheck",
    "CalledTool",
    "AnswerContains",
    "run_scenario_suite",
    "load_scenarios",
    "save_scenarios",
    # robustness (reward-hacking audit)
    "audit_pack",
    "RobustnessReport",
    "perturb_scenario",
    "ipt_report",
    # synth (auto-generate verifiers from a demonstration)
    "scenario_from_demonstration",
    "pack_from_demonstrations",
    # pack export (OpenEnv / Prime Intellect verifiers)
    "scenario_reward",
    "reward_from_messages",
    "export_pack",
    # reliability (beyond pass@1)
    "reliability_report",
    "ReliabilityReport",
    "wilson_interval",
    # contamination audit
    "contamination_report",
    "ContaminationReport",
    "canary_for",
    "held_out_pack",
    # importers
    "import_traces",
    "load_traces_jsonl",
    "redact_text",
    "redact_trajectory",
    "trajectory_from_messages",
    # flywheel
    "evolve_queries",
    "run_report_md",
    "FailureReport",
    "mine_failures",
    "mine_judge_failures",
    "targeted_queries",
    "recipe_from_failures",
    # tasks
    "SeedTask",
    "SEED_TASKS",
    "sample_tasks",
    # pipelines
    "Recipe",
    "RunResult",
    "run_recipe",
    "load_recipe",
    "make_environment",
    # verification
    "Verifier",
    "VerificationResult",
    "verify_trajectory",
    "batch_verify",
    "ExecutionVerifier",
    "ToolArgVerifier",
    "SafetyVerifier",
    "ExpectedAnswerVerifier",
    "EnsembleEvaluator",
    "LearnedVerifier",
    "train_learned_verifier",
    "RUBRIC_PRESETS",
    "get_rubric",
    "rubric_names",
    # preferences (DPO)
    "PreferencePair",
    "build_preference_pairs",
    "to_dpo_jsonl",
    "load_dpo_jsonl",
    # dedup / decontamination
    "dedup_trajectories",
    "decontaminate",
    "DedupResult",
    # training data prep
    "to_sft_records",
    "to_dpo_records",
    "build_sft_dataset",
    "build_dpo_dataset",
    # benchmark
    "BenchmarkCase",
    "BenchmarkReport",
    "BUILTIN_CASES",
    "run_benchmark",
    "compare_models",
    "agentsynth_model",
    "prompted_model",
    "report_table_md",
    # hub
    "dataset_card",
    "prepare_dataset_dir",
    "push_dataset",
]
