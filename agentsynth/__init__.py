"""AgentSynth: generate synthetic agent trajectories and score them with an
LLM-as-Judge loop. Runs offline against a mock by default.

    >>> from agentsynth import AgentTrajectoryGenerator, TrajectoryEvaluator
    >>> gen = AgentTrajectoryGenerator()
    >>> traj = gen.generate("What's the weather in Paris?")
    >>> ev = TrajectoryEvaluator().evaluate(traj)
    >>> ev.overall, ev.passed
"""

from __future__ import annotations

__version__ = "0.4.0"

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
from .exporters import (
    load_jsonl,
    save_dataset,
    to_adp,
    to_jsonl,
    to_sharegpt,
)
from .generator import AgentTrajectoryGenerator
from .hub import dataset_card, prepare_dataset_dir, push_dataset
from .metrics import compute_dataset_metrics, diversity_score
from .pipelines import Recipe, RunResult, load_recipe, make_environment, run_recipe
from .preferences import (
    PreferencePair,
    build_preference_pairs,
    load_dpo_jsonl,
    to_dpo_jsonl,
)
from .rl import AgentGym, make_reward_fn
from .schemas import (
    DEFAULT_RUBRIC_WEIGHTS,
    RUBRIC_DIMENSIONS,
    EvalResult,
    RubricScores,
    ToolSpec,
    Trajectory,
    TrajectoryStep,
)
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
