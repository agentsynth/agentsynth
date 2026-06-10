"""Distill the LLM judge into a cheap classifier and report the agreement.

    pip install "agentsynth-ai[learned]"
    python examples/learned_verifier.py

Generates a batch, judges it, trains a LearnedVerifier on the judge's labels, and
prints how often the classifier agrees with the judge on held-out trajectories —
the point being that a screening pass costs microseconds instead of an LLM call.
Runs offline; with real judge labels (provider key set) the same code applies.
"""

import os

os.environ.setdefault("AGENTSYNTH_FORCE_MOCK", "1")

import statistics

from agentsynth import (
    AgentTrajectoryGenerator,
    TrajectoryEvaluator,
    train_learned_verifier,
    verify_trajectory,
)


def main() -> None:
    gen = AgentTrajectoryGenerator(use_mock="auto")
    trajectories = gen.generate_batch(
        "analyze the quarter's sales and report by region", num_trajectories=120, vary_modes=True
    )
    results = TrajectoryEvaluator(use_mock="auto").evaluate_batch(trajectories)

    # Mock judge scores cluster high, so split labels at the median; with a real
    # LLM judge, the default (its pass/fail flag) is the natural label.
    median = statistics.median(r.overall for r in results)
    verifier, report = train_learned_verifier(trajectories, results, threshold=median)

    print(f"trained on {report['train_n']}, held out {report['test_n']}")
    print(f"agreement with the judge: {report['agreement']:.1%}")
    print(f"precision {report['precision']:.1%} / recall {report['recall']:.1%}\n")

    sample = trajectories[0]
    print(f"p(judge passes {sample.id}) = {verifier.predict_proba(sample):.3f}")
    outcome = verify_trajectory(sample, verifiers=[verifier])
    print("as a Verifier check:", outcome.checks[0].detail)


if __name__ == "__main__":
    main()
