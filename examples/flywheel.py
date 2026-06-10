"""One full turn of the flywheel: benchmark -> mine failures -> regenerate.

    python examples/flywheel.py

Benchmarks a (deliberately weak) model, mines what it got wrong, and turns the
failure report into a verified generation run aimed at exactly those gaps — the
data you'd fine-tune on next. Fully offline.
"""

import os

os.environ.setdefault("AGENTSYNTH_FORCE_MOCK", "1")

from agentsynth import mine_failures, recipe_from_failures, run_recipe
from agentsynth.benchmarks import BUILTIN_CASES, run_benchmark


def weak_model(query, tools):
    """Stands in for your current fine-tune: it loves get_weather a little too much."""
    return "get_weather", {}


def main() -> None:
    # 1. Evaluate the model you have.
    report = run_benchmark(weak_model, BUILTIN_CASES)
    print(f"benchmark: tool acc {report.tool_accuracy:.0%}, overall {report.score:.0%}\n")

    # 2. Mine what failed and why.
    mined = mine_failures(report, BUILTIN_CASES)
    print(mined.summary_md())

    # 3. Aim the next generation run at the gaps (verified by default).
    recipe = recipe_from_failures(mined, k=8, evaluate=True)
    result = run_recipe(recipe)
    print(
        f"\npatch dataset: {len(result.trajectories)} trajectories targeting the failures "
        f"(verified rate {result.metrics.get('verified_rate')})"
    )
    print("sample targeted query:", result.trajectories[0].query)
    # 4. Fine-tune on the patch data, re-benchmark, repeat.


if __name__ == "__main__":
    main()
