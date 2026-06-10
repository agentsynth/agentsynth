"""Distill the LLM judge into a small, cheap classifier.

The LLM-as-judge is the quality signal, but calling it on every trajectory is what
makes generation expensive at scale. A LearnedVerifier trains on judge labels and
predicts pass/fail from cheap, deterministic trajectory features — microseconds per
trajectory instead of an LLM call. Distill once, screen cheaply, and reserve the
real judge for the borderline band.

    result = run_recipe(Recipe(num_trajectories=500, verify=True))
    judged = TrajectoryEvaluator().evaluate_batch(result.trajectories)
    verifier, report = train_learned_verifier(result.trajectories, judged)
    print(report["agreement"])          # held-out agreement with the LLM judge
    verify_trajectory(traj, verifiers=[verifier])

scikit-learn does the fitting: `pip install "agentsynth-ai[learned]"`. The fitted
verifier is a plain picklable object and plugs into the same `Verifier` interface
as everything else.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..schemas import EvalResult, Trajectory
from .base import CheckResult, Verifier

_ERROR_RE = re.compile(r"error|traceback|exception|denied|refused", re.IGNORECASE)

# Order matters: it's the feature-vector layout the classifier is trained on.
FEATURE_NAMES = [
    "num_steps",
    "num_tool_calls",
    "num_observations",
    "num_thoughts",
    "distinct_tools",
    "repeated_signature",
    "valid_arg_ratio",
    "error_observation_ratio",
    "avg_observation_chars",
    "has_final_answer",
    "final_answer_words",
    "thought_before_first_call",
    "query_words",
    "mode_single",
    "mode_multi",
    "mode_code",
    "has_verification",
    "verification_verified",
    "verification_score",
]


def _valid_arg_ratio(traj: Trajectory) -> float:
    """Fraction of tool calls that name a real tool and fill its required args."""
    calls = traj.tool_calls()
    if not calls:
        return 1.0
    spec_by_name = {t.name: t for t in traj.tools}
    ok = 0
    for step in calls:
        spec = spec_by_name.get(step.tool_name) if step.tool_name else None
        if spec is None:
            continue
        args = step.tool_args or {}
        if all(r in args and args[r] is not None for r in spec.required_args()):
            ok += 1
    return ok / len(calls)


def extract_features(trajectory: Trajectory) -> List[float]:
    """A deterministic, cheap feature vector for one trajectory (see FEATURE_NAMES)."""
    steps = trajectory.steps
    calls = trajectory.tool_calls()
    observations = [s for s in steps if s.step_type == "observation"]
    thoughts = [s for s in steps if s.step_type in ("thought", "plan", "critique")]

    obs_texts = [s.observation or "" for s in observations]
    error_obs = sum(1 for t in obs_texts if _ERROR_RE.search(t))
    signatures = [f"{s.tool_name}:{sorted((s.tool_args or {}).items())!r}" for s in calls]

    first_call_idx = next((i for i, s in enumerate(steps) if s.step_type == "tool_call"), None)
    thought_first = (
        1.0
        if first_call_idx is not None
        and any(s.step_type in ("thought", "plan") for s in steps[:first_call_idx])
        else 0.0
    )

    verification = trajectory.verification or {}
    return [
        float(len(steps)),
        float(len(calls)),
        float(len(observations)),
        float(len(thoughts)),
        float(len(set(s.tool_name for s in calls if s.tool_name))),
        float(len(signatures) - len(set(signatures))),
        _valid_arg_ratio(trajectory),
        error_obs / len(observations) if observations else 0.0,
        sum(len(t) for t in obs_texts) / len(obs_texts) if obs_texts else 0.0,
        1.0 if (trajectory.final_answer or "").strip() else 0.0,
        float(len((trajectory.final_answer or "").split())),
        thought_first,
        float(len((trajectory.query or "").split())),
        1.0 if trajectory.mode == "single_agent" else 0.0,
        1.0 if trajectory.mode == "multi_agent" else 0.0,
        1.0 if trajectory.mode == "code_execution" else 0.0,
        1.0 if trajectory.verification is not None else 0.0,
        1.0 if verification.get("verified") else 0.0,
        float(verification.get("score") or 0.0),
    ]


class LearnedVerifier(Verifier):
    """A judge-distilled classifier behind the standard Verifier interface.

    Advisory by default (`required=False`): it contributes to the verification
    score without hard-failing a trajectory, since it's a screen, not a proof.
    """

    name = "learned_judge"
    required = False

    def __init__(self, model: Any, threshold: float = 0.5) -> None:
        self.model = model
        self.threshold = threshold

    def predict_proba(self, trajectory: Trajectory) -> float:
        """P(the LLM judge would pass this trajectory)."""
        return float(self.model.predict_proba([extract_features(trajectory)])[0][1])

    def check(self, trajectory: Trajectory) -> CheckResult:
        proba = self.predict_proba(trajectory)
        passed = proba >= self.threshold
        return CheckResult(name=self.name, passed=passed, detail=f"p(pass)={proba:.3f}")


def train_learned_verifier(
    trajectories: Sequence[Trajectory],
    eval_results: Sequence[EvalResult],
    threshold: Optional[float] = None,
    test_size: float = 0.25,
    seed: int = 7,
) -> Tuple[LearnedVerifier, Dict[str, Any]]:
    """Fit a LearnedVerifier on judge labels and report held-out agreement.

    Labels come from each eval result's `passed` flag, or `overall >= threshold`
    when `threshold` is given. Returns `(verifier, report)` where the report has
    `agreement` (held-out accuracy vs the judge), `precision`/`recall` for the
    pass class, and the split sizes. Raises ValueError when the labels are all
    one class — vary the rubric or threshold so there is something to learn.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            'the learned verifier needs scikit-learn: pip install "agentsynth-ai[learned]"'
        ) from exc

    by_id = {r.trajectory_id: r for r in eval_results}
    X: List[List[float]] = []
    y: List[int] = []
    for traj in trajectories:
        result = by_id.get(traj.id)
        if result is None:
            continue
        X.append(extract_features(traj))
        if threshold is None:
            y.append(1 if result.passed else 0)
        else:
            y.append(1 if result.overall >= threshold else 0)

    if len(set(y)) < 2:
        raise ValueError(
            "all labels are identical, nothing to learn — vary the rubric or pass a "
            "threshold near the score median"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    true_pass = sum(1 for p, t in zip(predictions, y_test) if p == 1 and t == 1)
    agreement = sum(1 for p, t in zip(predictions, y_test) if p == t) / len(y_test)
    precision = true_pass / max(1, sum(1 for p in predictions if p == 1))
    recall = true_pass / max(1, sum(1 for t in y_test if t == 1))

    report = {
        "n": len(y),
        "train_n": len(y_train),
        "test_n": len(y_test),
        "pass_rate": round(sum(y) / len(y), 4),
        "agreement": round(agreement, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "features": list(FEATURE_NAMES),
    }
    return LearnedVerifier(model), report


__all__ = [
    "FEATURE_NAMES",
    "extract_features",
    "LearnedVerifier",
    "train_learned_verifier",
]
