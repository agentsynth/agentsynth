"""Dataset-level quality metrics and Plotly figures for AgentSynth.

Plotting helpers import Plotly (and pandas) lazily, so importing this module
stays cheap. Everything here copes with empty input and returns zeros rather
than blowing up on degenerate data.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from .schemas import RUBRIC_DIMENSIONS, EvalResult, Trajectory

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _mean(values: Sequence[float]) -> Optional[float]:
    """Mean rounded to 4 dp, or None for an empty sequence."""
    vals = [float(v) for v in values]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def diversity_score(trajectories: Sequence[Trajectory]) -> float:
    """Diversity of a trajectory set, in [0, 1]; 0.0 when empty.

    Averages three signals, each normalised on its own: structural (unique tool
    signatures over count), domain (unique domains over count), and lexical (the
    type-token ratio across all query tokens).
    """
    trajs = list(trajectories or [])
    n = len(trajs)
    if n == 0:
        return 0.0

    unique_sigs = {t.tool_signature() for t in trajs}
    structural = _clamp01(len(unique_sigs) / n)

    unique_domains = {t.domain for t in trajs if t.domain}
    domain = _clamp01(len(unique_domains) / n)

    all_tokens: List[str] = []
    for t in trajs:
        all_tokens.extend(_tokenize(t.query or ""))
    if all_tokens:
        lexical = _clamp01(len(set(all_tokens)) / len(all_tokens))
    else:
        lexical = 0.0

    return round((structural + domain + lexical) / 3.0, 4)


def _empty_metrics() -> Dict[str, Any]:
    return {
        "num_trajectories": 0,
        "num_evaluated": 0,
        "pass_rate": None,
        "avg_overall": None,
        "avg_scores": {},
        "avg_steps": 0.0,
        "avg_tool_calls": 0.0,
        "success_rate": 0.0,
        "unique_tools": 0,
        "tool_usage": {},
        "tool_coverage": 0.0,
        "mode_distribution": {},
        "domain_distribution": {},
        "step_type_distribution": {},
        "diversity_score": 0.0,
    }


def compute_dataset_metrics(
    trajectories: Sequence[Trajectory],
    eval_results: Optional[Sequence[EvalResult]] = None,
) -> Dict[str, Any]:
    """Flat dict of dataset-level metrics.

    Without `eval_results` the judge-derived keys (`pass_rate`, `avg_overall`,
    `avg_scores`) come back None/empty. Empty input gives a zero-filled dict.
    """
    trajs = list(trajectories or [])
    if not trajs:
        return _empty_metrics()

    n = len(trajs)

    step_counts = [t.num_steps() for t in trajs]
    tool_call_counts = [len(t.tool_calls()) for t in trajs]
    success_count = sum(1 for t in trajs if t.success)

    tool_usage: Dict[str, int] = {}
    for t in trajs:
        for name in t.tool_names_used():
            tool_usage[name] = tool_usage.get(name, 0) + 1
    # Count desc, then name asc, so the order is stable across runs.
    tool_usage = dict(sorted(tool_usage.items(), key=lambda kv: (-kv[1], kv[0])))
    unique_tools = len(tool_usage)

    # Coverage is tools actually called over tools offered across all catalogs.
    available_tools = set()
    for t in trajs:
        for spec in t.tools:
            if spec.name:
                available_tools.add(spec.name)
    used_tools = set(tool_usage.keys())
    if available_tools:
        coverage = _clamp01(len(used_tools & available_tools) / len(available_tools))
    else:
        coverage = 0.0

    mode_distribution: Dict[str, int] = {}
    for t in trajs:
        mode_distribution[t.mode] = mode_distribution.get(t.mode, 0) + 1

    domain_distribution: Dict[str, int] = {}
    for t in trajs:
        key = t.domain if t.domain else "unspecified"
        domain_distribution[key] = domain_distribution.get(key, 0) + 1

    step_type_distribution: Dict[str, int] = {}
    for t in trajs:
        for step in t.steps:
            step_type_distribution[step.step_type] = (
                step_type_distribution.get(step.step_type, 0) + 1
            )

    evals = list(eval_results or [])
    num_evaluated = len(evals)
    if evals:
        pass_rate = round(sum(1 for e in evals if e.passed) / num_evaluated, 4)
        avg_overall = _mean([e.overall for e in evals])
        avg_scores: Dict[str, Optional[float]] = {}
        for dim in RUBRIC_DIMENSIONS:
            avg_scores[dim] = _mean([float(getattr(e.scores, dim)) for e in evals])
    else:
        pass_rate = None
        avg_overall = None
        avg_scores = {}

    return {
        "num_trajectories": n,
        "num_evaluated": num_evaluated,
        "pass_rate": pass_rate,
        "avg_overall": avg_overall,
        "avg_scores": avg_scores,
        "avg_steps": _mean(step_counts) or 0.0,
        "avg_tool_calls": _mean(tool_call_counts) or 0.0,
        "success_rate": round(success_count / n, 4),
        "unique_tools": unique_tools,
        "tool_usage": tool_usage,
        "tool_coverage": round(coverage, 4),
        "mode_distribution": mode_distribution,
        "domain_distribution": domain_distribution,
        "step_type_distribution": step_type_distribution,
        "diversity_score": diversity_score(trajs),
    }


_TEMPLATE = "plotly_white"


def _empty_figure(title: str):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(template=_TEMPLATE, title=title)
    return fig


def plot_rubric_radar(eval_results: Optional[Sequence[EvalResult]]):
    """Scatterpolar radar of the mean of each rubric dimension."""
    import plotly.graph_objects as go

    evals = list(eval_results or [])
    if not evals:
        return _empty_figure("No data yet")

    means = [
        _mean([float(getattr(e.scores, dim)) for e in evals]) or 0.0 for dim in RUBRIC_DIMENSIONS
    ]
    labels = [dim.replace("_", " ").title() for dim in RUBRIC_DIMENSIONS]

    # Repeat the first point to close the polygon.
    r = list(means) + [means[0]]
    theta = list(labels) + [labels[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=r,
            theta=theta,
            fill="toself",
            name="mean score",
            line=dict(color="#4C78A8"),
        )
    )
    fig.update_layout(
        template=_TEMPLATE,
        title="Mean rubric scores",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
    )
    return fig


def plot_score_distribution(eval_results: Optional[Sequence[EvalResult]]):
    """Histogram of overall scores."""
    import plotly.graph_objects as go

    evals = list(eval_results or [])
    if not evals:
        return _empty_figure("No data yet")

    overalls = [float(e.overall) for e in evals]
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=overalls,
            xbins=dict(start=0.0, end=1.0, size=0.1),
            marker=dict(color="#4C78A8"),
            name="overall",
        )
    )
    fig.update_layout(
        template=_TEMPLATE,
        title="Overall score distribution",
        xaxis_title="overall score",
        yaxis_title="count",
        bargap=0.05,
    )
    fig.update_xaxes(range=[0, 1])
    return fig


def plot_pass_gauge(eval_results: Optional[Sequence[EvalResult]]):
    """Gauge showing pass@1 as a percentage."""
    import plotly.graph_objects as go

    evals = list(eval_results or [])
    if not evals:
        return _empty_figure("No data yet")

    pass_pct = 100.0 * sum(1 for e in evals if e.passed) / len(evals)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(pass_pct, 1),
            number={"suffix": "%"},
            title={"text": "pass@1"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4C78A8"},
                "steps": [
                    {"range": [0, 50], "color": "#F2DEDE"},
                    {"range": [50, 80], "color": "#FCF8E3"},
                    {"range": [80, 100], "color": "#DFF0D8"},
                ],
            },
        )
    )
    fig.update_layout(template=_TEMPLATE, title="Pass rate")
    return fig


def plot_tool_usage(trajectories: Optional[Sequence[Trajectory]]):
    """Horizontal bar chart of tool-usage counts, top 15 tools."""
    import plotly.graph_objects as go

    trajs = list(trajectories or [])
    if not trajs:
        return _empty_figure("No data yet")

    usage: Dict[str, int] = {}
    for t in trajs:
        for name in t.tool_names_used():
            usage[name] = usage.get(name, 0) + 1
    if not usage:
        return _empty_figure("No tool calls yet")

    items = sorted(usage.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
    # Smallest on top, largest on the bottom reads better on a horizontal bar.
    items = list(reversed(items))
    names = [name for name, _ in items]
    counts = [count for _, count in items]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=counts,
            y=names,
            orientation="h",
            marker=dict(color="#4C78A8"),
        )
    )
    fig.update_layout(
        template=_TEMPLATE,
        title="Tool usage",
        xaxis_title="calls",
        yaxis_title="tool",
    )
    return fig


def plot_step_distribution(trajectories: Optional[Sequence[Trajectory]]):
    """Histogram of trajectory length in steps."""
    import plotly.graph_objects as go

    trajs = list(trajectories or [])
    if not trajs:
        return _empty_figure("No data yet")

    lengths = [t.num_steps() for t in trajs]
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=lengths,
            marker=dict(color="#4C78A8"),
            name="steps",
        )
    )
    fig.update_layout(
        template=_TEMPLATE,
        title="Trajectory length distribution",
        xaxis_title="steps per trajectory",
        yaxis_title="count",
        bargap=0.05,
    )
    return fig


def _fmt(value: Any, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if pct:
            return f"{value * 100:.1f}%"
        return f"{value:.3f}"
    return str(value)


def metrics_summary_md(metrics: Dict[str, Any]) -> str:
    """Render a metrics dict as GitHub-flavoured markdown.

    A headline table plus a per-dimension averages table. Safe on the dict from
    compute_dataset_metrics, empty case included.
    """
    metrics = metrics or {}

    headline = [
        ("Trajectories", _fmt(metrics.get("num_trajectories"))),
        ("Evaluated", _fmt(metrics.get("num_evaluated"))),
        ("Pass@1", _fmt(metrics.get("pass_rate"), pct=True)),
        ("Avg overall", _fmt(metrics.get("avg_overall"))),
        ("Success rate", _fmt(metrics.get("success_rate"), pct=True)),
        ("Avg steps", _fmt(metrics.get("avg_steps"))),
        ("Avg tool calls", _fmt(metrics.get("avg_tool_calls"))),
        ("Unique tools", _fmt(metrics.get("unique_tools"))),
        ("Tool coverage", _fmt(metrics.get("tool_coverage"), pct=True)),
        ("Diversity", _fmt(metrics.get("diversity_score"))),
    ]

    lines: List[str] = []
    lines.append("### Dataset metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    for label, value in headline:
        lines.append(f"| {label} | {value} |")

    avg_scores = metrics.get("avg_scores") or {}
    if avg_scores:
        lines.append("")
        lines.append("#### Average rubric scores")
        lines.append("")
        lines.append("| Dimension | Mean |")
        lines.append("| --- | --- |")
        for dim in RUBRIC_DIMENSIONS:
            label = dim.replace("_", " ").title()
            lines.append(f"| {label} | {_fmt(avg_scores.get(dim))} |")

    return "\n".join(lines)


def filter_dataset(
    trajectories: Sequence[Trajectory],
    eval_results: Optional[Sequence[EvalResult]] = None,
    min_score: float = 0.6,
    verified_only: bool = False,
    modes: Optional[Sequence[str]] = None,
) -> Tuple[List[Trajectory], Dict[str, Any]]:
    """Filter a batch of trajectories, returning the kept subset and a drop report.

    Drops trajectories if:
    - Their overall score in `eval_results` is below `min_score`.
    - `verified_only` is True and the trajectory failed verification.
    - `modes` is provided and the trajectory's mode is not in the list.
    """
    from typing import Tuple
    
    trajs = list(trajectories or [])
    
    eval_map = {}
    if eval_results:
        for res in eval_results:
            eval_map[res.trajectory_id] = res

    kept: List[Trajectory] = []
    report = {
        "initial": len(trajs),
        "kept": 0,
        "dropped_score": 0,
        "dropped_verification": 0,
        "dropped_mode": 0,
        "dropped_total": 0,
    }

    mode_set = set(modes) if modes is not None else None

    for t in trajs:
        if mode_set is not None and t.mode not in mode_set:
            report["dropped_mode"] += 1
            report["dropped_total"] += 1
            continue
            
        if verified_only:
            ver_dict = t.verification or {}
            if not ver_dict.get("verified", False):
                report["dropped_verification"] += 1
                report["dropped_total"] += 1
                continue
                
        if eval_results is not None:
            res = eval_map.get(t.id)
            if res is None or res.overall < min_score:
                report["dropped_score"] += 1
                report["dropped_total"] += 1
                continue
                
        kept.append(t)
        
    report["kept"] = len(kept)
    return kept, report


__all__ = [
    "compute_dataset_metrics",
    "diversity_score",
    "filter_dataset",
    "plot_rubric_radar",
    "plot_score_distribution",
    "plot_pass_gauge",
    "plot_tool_usage",
    "plot_step_distribution",
    "metrics_summary_md",
]
