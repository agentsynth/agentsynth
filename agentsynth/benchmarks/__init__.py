"""A self-contained function-calling benchmark, plus BFCL and τ-bench adapters."""

from __future__ import annotations

from .bfcl import SAMPLE_BFCL, bfcl_case, load_bfcl, sample_cases
from .tau_bench import run_tau_bench, tau_bench_available
from .tool_calling import (
    BUILTIN_CASES,
    BenchmarkCase,
    BenchmarkReport,
    CaseResult,
    agentsynth_model,
    compare_models,
    prompted_model,
    report_table_md,
    run_benchmark,
)

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
    # BFCL
    "load_bfcl",
    "bfcl_case",
    "sample_cases",
    "SAMPLE_BFCL",
    # tau-bench
    "run_tau_bench",
    "tau_bench_available",
]
