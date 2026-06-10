"""Score a model on a real slice of the Berkeley Function-Calling Leaderboard (BFCL).

    python examples/benchmark_bfcl.py

Runs offline on the 25-case BFCL `simple_python` slice bundled with the package
(see agentsynth/benchmarks/data/NOTICE.md). The model here is AgentSynth's own mock
generator — swap in a fine-tuned model with `prompted_model(complete_fn)` as the
"after" and use `compare_models(...)` for a before/after table, or point
`load_bfcl(questions, answers)` at the official BFCL files for the full suite.
"""

import os

os.environ.setdefault("AGENTSYNTH_FORCE_MOCK", "1")

from agentsynth import AgentTrajectoryGenerator
from agentsynth.benchmarks import agentsynth_model, load_sample_bfcl, run_benchmark


def main() -> None:
    model = agentsynth_model(AgentTrajectoryGenerator(use_mock=True))
    for split in ("simple", "multiple"):
        cases = load_sample_bfcl(split=split)
        report = run_benchmark(model, cases=cases)
        print(f"BFCL {split} slice — {report.n} real cases")
        print(f"  tool accuracy: {report.tool_accuracy:.1%}")
        print(f"  arg accuracy:  {report.arg_accuracy:.1%}")
        print(f"  overall score: {report.score:.1%}\n")
    print(
        "The 'multiple' split offers 2-3 candidate functions per case, so it tests\n"
        "tool *selection*, not just formatting. Plug a fine-tuned model in via\n"
        "prompted_model(...) and compare_models(...) for a before/after table;\n"
        "point load_bfcl() at the official BFCL files for the full suite."
    )


if __name__ == "__main__":
    main()
