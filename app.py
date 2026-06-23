"""Gradio web UI for AgentSynth.

Four tabs: generate trajectories, evaluate them with the LLM-as-Judge, view
dataset metrics, and export. Importing this module builds `demo` but never
touches an LLM; `demo.launch()` only runs under __main__.
"""

from __future__ import annotations

import html
import json
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from agentsynth import (
    AgentTrajectoryGenerator,
    CalledTool,
    CodeCheck,
    Scenario,
    SqlCheck,
    TrajectoryEvaluator,
    compute_dataset_metrics,
    default_tool_catalog,
    parse_tool_catalog,
    save_dataset,
)
from agentsynth import metrics as M
from agentsynth.demo import DEMO_POLICIES, demo_scenarios, llm_policy_for
from agentsynth.schemas import RUBRIC_DIMENSIONS
from agentsynth.utils import (
    detect_default_model,
    tool_catalog_to_json,
)

MOCK_LABEL = "mock (offline)"

_DEFAULT_MODEL_CHOICE = detect_default_model() or MOCK_LABEL

# Common model ids for the dropdowns; users can also type their own.
_MODEL_CHOICES = [
    MOCK_LABEL,
    "claude-3-5-haiku-latest",
    "claude-3-5-sonnet-latest",
    "gpt-4o-mini",
    "gpt-4o",
    "groq/llama-3.3-70b-versatile",
    "openrouter/meta-llama/llama-3.1-70b-instruct",
]
if _DEFAULT_MODEL_CHOICE not in _MODEL_CHOICES:
    _MODEL_CHOICES.insert(0, _DEFAULT_MODEL_CHOICE)

_DEFAULT_CATALOG_JSON = tool_catalog_to_json(default_tool_catalog())

# (chip label, css class) per step type — the timeline colors hang off the class.
_STEP_META = {
    "thought": ("THINK", "think"),
    "plan": ("PLAN", "think"),
    "critique": ("REVIEW", "think"),
    "tool_call": ("TOOL", "tool"),
    "observation": ("OBS", "obs"),
    "code_execution": ("CODE", "code"),
    "final_answer": ("ANSWER", "answer"),
}


def _model_arg(choice: Optional[str]) -> Optional[str]:
    """Dropdown choice to model id, or None for mock."""
    if not choice or choice == MOCK_LABEL:
        return None
    return str(choice).strip() or None


def _truncate(text: Optional[str], width: int = 80) -> str:
    if not text:
        return ""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def _esc(text: Any) -> str:
    return html.escape("" if text is None else str(text), quote=True)


def render_tree(traj: Any) -> str:
    """Render a trajectory as an HTML step timeline."""
    if traj is None:
        return "_No trajectory to preview yet._"

    head = [
        '<div class="traj">',
        '<div class="traj-head">',
        f'<span class="traj-id">{_esc(traj.id)}</span>',
        f'<span class="badge">{_esc(traj.mode)}</span>',
    ]
    if getattr(traj, "domain", None):
        head.append(f'<span class="badge soft">{_esc(traj.domain)}</span>')
    head.append("</div>")
    head.append(f'<div class="traj-query">{_esc(traj.query)}</div>')
    parts = head

    parts.append('<ol class="steps">')
    for step in traj.steps:
        label, klass = _STEP_META.get(step.step_type, (step.step_type.upper(), "obs"))
        agent = (
            f'<span class="agent">{_esc(step.agent)}</span>' if getattr(step, "agent", None) else ""
        )
        body = ""
        if step.step_type == "tool_call":
            try:
                args_str = json.dumps(step.tool_args or {}, ensure_ascii=False)
            except Exception:
                args_str = str(step.tool_args)
            body = (
                f'<span class="mono"><b>{_esc(step.tool_name)}</b>'
                f"({_esc(_truncate(args_str, 160))})</span>"
            )
        elif step.step_type == "observation":
            body = f'<span class="dim-text">{_esc(_truncate(step.observation, 220))}</span>'
        elif step.step_type == "code_execution":
            code = (step.code or "").strip()
            out = (step.code_output or "").strip()
            chunks = []
            if code:
                chunks.append(f'<pre class="codeblock">{_esc(code)}</pre>')
            if out:
                chunks.append(f'<span class="mono dim-text">→ {_esc(_truncate(out, 160))}</span>')
            body = "".join(chunks)
        elif step.step_type in ("thought", "plan", "critique"):
            text = _esc(_truncate(step.thought or step.content, 240))
            body = f'<span class="dim-text">{text}</span>'
        elif step.step_type == "final_answer":
            text = step.content or traj.final_answer or ""
            body = f'<span class="answer-text">{_esc(_truncate(text, 280))}</span>'
        else:
            body = f'<span class="dim-text">{_esc(_truncate(step.short(), 180))}</span>'

        parts.append(
            f'<li class="step {klass}"><span class="chip {klass}">{label}</span>'
            f'<div class="step-body">{agent}{body}</div></li>'
        )
    parts.append("</ol>")

    if traj.final_answer:
        parts.append(f'<div class="traj-final">{_esc(traj.final_answer)}</div>')
    parts.append("</div>")
    return "".join(parts)


def _eval_by_id(eval_results: Optional[List[Any]]) -> Dict[str, Any]:
    return {getattr(ev, "trajectory_id", None): ev for ev in (eval_results or [])}


def _overview_row(idx: int, traj: Any, by_id: Dict[str, Any]) -> List[Any]:
    ev = by_id.get(traj.id)
    score = round(float(ev.overall), 3) if ev is not None else "—"
    return [
        idx,
        traj.mode,
        traj.domain or "—",
        traj.num_steps(),
        score,
        ", ".join(traj.tool_names_used()) or "—",
        _truncate(traj.final_answer, 70),
    ]


def traj_overview_rows(
    trajectories: List[Any], eval_results: Optional[List[Any]] = None
) -> List[List[Any]]:
    """One row per trajectory for the batch overview table (score fills in once judged)."""
    by_id = _eval_by_id(eval_results)
    return [_overview_row(idx, traj, by_id) for idx, traj in enumerate(trajectories or [])]


def filter_overview_rows(
    trajectories: List[Any],
    eval_results: Optional[List[Any]],
    mode: Optional[str],
    min_score: float,
) -> List[List[Any]]:
    """Overview rows kept after the mode / min-score filters (the true idx stays in column 0)."""
    by_id = _eval_by_id(eval_results)
    rows: List[List[Any]] = []
    for idx, traj in enumerate(trajectories or []):
        if mode and mode != "all" and traj.mode != mode:
            continue
        ev = by_id.get(traj.id)
        if min_score and float(min_score) > 0:
            if ev is None or float(ev.overall) < float(min_score):
                continue
        rows.append(_overview_row(idx, traj, by_id))
    return rows


_DEMO_SCENARIOS = {s.id: s for s in demo_scenarios()}
_LLM_POLICY_LABEL = "LLM (pick a model →)"


def _outcome_card(outcome: Dict[str, Any], total_reward: float) -> str:
    """The world-state checks that decided the episode — the part that can't be talked past."""
    score = float(outcome.get("score", 0.0))
    ok = score >= 1.0
    badge = '<span class="badge pass">PASS</span>' if ok else '<span class="badge fail">FAIL</span>'
    rows = []
    for check in outcome.get("checks", []):
        mark = "ok" if check.get("passed") else "fail"
        label = "✓" if check.get("passed") else "✗"
        rows.append(
            f'<div class="ocheck"><span class="omark {mark}">{label}</span>'
            f'<span class="oname">{_esc(check.get("name", "?"))}</span>'
            f'<span class="odetail">{_esc(_truncate(check.get("detail", ""), 110))}</span></div>'
        )
    return (
        '<div class="verdict">'
        f'<div class="vhead">Outcome checks <b>{score:.2f}</b> {badge}'
        f'<span class="vnum" style="flex:none;margin-left:auto">reward {total_reward:.3f}</span>'
        f"</div>{''.join(rows)}</div>"
    )


def do_agent_run(scenario_id: str, policy_name: str, model_choice: str) -> str:
    """One real episode: the chosen policy works the world, the checkers judge it."""
    scenario = _DEMO_SCENARIOS.get(scenario_id)
    if scenario is None:
        return "_Pick a scenario first._"

    if policy_name == _LLM_POLICY_LABEL:
        model = _model_arg(model_choice)
        if not model:
            gr.Warning("Pick or type a real model id for the LLM policy.")
            return (
                "_The LLM policy needs a model id — choose one in the model dropdown "
                "(and set its provider key), or run one of the scripted policies._"
            )
        policy = llm_policy_for(model)
        if policy is None:
            gr.Warning(f"Model '{model}' isn't usable here (missing key?).")
            return (
                f"_Could not reach `{model}` — check the provider key, or run a scripted policy._"
            )
    else:
        policy = DEMO_POLICIES.get(policy_name)
        if policy is None:
            return "_Unknown policy._"

    from agentsynth.rl import AgentGym

    gym = AgentGym.from_scenario(scenario, seed=7)
    try:
        episode = gym.rollout(policy)
    finally:
        gym.close()

    outcome = episode.info.get("outcome", {})
    return (
        _outcome_card(outcome, episode.total_reward)
        + _repro_badge(scenario, policy_name, outcome.get("score", 0.0))
        + render_tree(episode.trajectory)
    )


def do_agent_task(scenario_id: str) -> str:
    scenario = _DEMO_SCENARIOS.get(scenario_id)
    return f"**Task:** {scenario.task}" if scenario else ""


def do_compare(policy_names: Optional[List[str]], model_choice: str, trials: float) -> str:
    """The CLI's bench --compare, in the browser: one pass^k table across the pack."""
    from agentsynth.cli import _run_trials

    items = [(name, DEMO_POLICIES[name]) for name in policy_names or [] if name in DEMO_POLICIES]
    model = _model_arg(model_choice)
    if model:
        policy = llm_policy_for(model)
        if policy is None:
            gr.Warning(
                f"Model '{model}' isn't usable (missing key?) — comparing the scripted picks."
            )
        else:
            items.append((model, policy))
    if len(items) < 2:
        return "_Pick at least two policies (or add a usable model) to compare._"

    scenarios = list(_DEMO_SCENARIOS.values())
    k = max(1, int(trials))
    runs = []
    for name, policy in items:
        report, pass1_avg = _run_trials(policy, scenarios, seed=7, trials=k)
        runs.append((name, report, pass1_avg))

    head = "".join(f"<th>{_esc(name)}</th>" for name, _, _ in runs)
    rows = []
    for idx, row in enumerate(runs[0][1].results):
        cells = "".join(
            '<td class="cmp-ok">✓</td>'
            if run[1].results[idx]["passed"]
            else '<td class="cmp-fail">✗</td>'
            for run in runs
        )
        rows.append(f'<tr><td class="mono">{_esc(row["id"])}</td>{cells}</tr>')
    label = f"pass^{k}" if k > 1 else "pass rate"
    footer = "".join(f"<td><b>{run[1].pass_rate:.0%}</b></td>" for run in runs)
    extra = ""
    if k > 1:
        avgs = "".join(f"<td>{run[2]:.0%}</td>" for run in runs)
        extra = f'<tr><td class="mono">pass^1 avg</td>{avgs}</tr>'
    return (
        '<div class="traj"><table class="cmp">'
        f"<tr><th>scenario</th>{head}</tr>{''.join(rows)}"
        f'<tr><td class="mono"><b>{label}</b></td>{footer}</tr>{extra}'
        "</table></div>"
    )


def _repro_badge(scenario: Any, policy_label: str, score: float) -> str:
    """The verifiable run_hash for this episode, so anyone can re-derive it (P2.1)."""
    from agentsynth.provenance import pack_fingerprint, run_hash

    fingerprint = pack_fingerprint([scenario])
    rows = [{"id": scenario.id, "passed": score >= 1.0, "outcome_score": round(float(score), 6)}]
    digest = run_hash(fingerprint, str(policy_label), 7, 1, rows)
    return (
        '<div class="repro"><span class="repro-ic">&#10003;</span>'
        f'<span><b>reproducible</b> &mdash; <code>run_hash {digest}</code><br>'
        '<span class="repro-sub">same pack + policy + seed re-derives this exact hash &middot; '
        "check any leaderboard entry with <code>agentsynth pack verify-run</code>"
        "</span></span></div>"
    )


def do_robustness() -> str:
    """How gameable is the pack? Run the trivial adversaries over it (P0.1)."""
    from agentsynth.robustness import audit_pack

    scenarios = list(_DEMO_SCENARIOS.values())
    report = audit_pack(scenarios)
    pct = report.robustness_score
    badge = (
        f'<span class="badge {"pass" if pct >= 1 else "soft"}">{pct:.0%} resist gaming</span>'
    )
    rows = []
    for row in report.rows:
        gamed = ", ".join(row.gamed_by) if row.gamed_by else "—"
        leaks = ", ".join(row.answer_leaks) if row.answer_leaks else "—"
        change = "yes" if not row.state_noop_satisfiable else "no"
        cls = "cmp-fail" if row.gamed_by else "cmp-ok"
        rows.append(
            f'<tr><td class="mono">{_esc(row.scenario_id)}</td>'
            f'<td class="{cls}">{_esc(gamed)}</td>'
            f'<td>{_esc(leaks)}</td><td>{change}</td></tr>'
        )
    return (
        '<div class="traj"><div class="traj-head">'
        f"<b>Robustness audit</b> {badge}"
        '<span class="dim-text" style="margin-left:auto">adversaries: noop · constant · '
        "echo · echo+probe</span></div>"
        '<table class="cmp"><tr><th>scenario</th><th>gamed by</th>'
        "<th>answer leaks</th><th>asserts a change</th></tr>"
        f"{''.join(rows)}</table>"
        '<p class="dim-text" style="margin-top:12px">A scenario no trivial adversary passes is '
        "robust. The ones that fall are graded on words, not the world — a canned answer, an "
        "echoed prompt, a throwaway tool call. <code>agentsynth pack audit</code> ships this "
        "gate; it operationalizes the 2026 “LLMs gaming verifiers” work.</p></div>"
    )


_CODE_DEMO = Scenario(
    id="is-prime",
    task="Write a function `is_prime(n)` that returns True for primes.",
    environment={"type": "python"},
    checkers=[
        CalledTool(name="python"),
        CodeCheck(test="assert is_prime(13) and not is_prime(15) and not is_prime(1)"),
    ],
)
_CODE_GOOD = (
    "def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n"
    "    while i * i <= n:\n        if n % i == 0:\n            return False\n"
    "        i += 1\n    return True\n"
)
_CODE_BUGGY = "def is_prime(n):\n    return n > 1   # claims 9, 15, ... are prime\n"


def do_code_demo(submission: str) -> str:
    """Code graded by hidden unit tests, not by the transcript (P1.3 / CodeCheck)."""
    from agentsynth.rl import AgentGym

    code = _CODE_BUGGY if submission and "buggy" in submission else _CODE_GOOD

    def policy(observation: str, gym: Any) -> dict:
        if gym.step_count == 0:
            return {"tool_name": "python", "arguments": {"code": code}}
        return {"answer": "Defined is_prime."}

    gym = AgentGym.from_scenario(_CODE_DEMO, seed=7)
    try:
        episode = gym.rollout(policy)
    finally:
        gym.close()
    outcome = episode.info.get("outcome", {})
    return _outcome_card(outcome, episode.total_reward) + render_tree(episode.trajectory)


def do_contamination() -> str:
    """Canaries + held-out siblings: is the benchmark already in the training set? (P1.4)."""
    from agentsynth.contamination import contamination_report

    report = contamination_report(list(_DEMO_SCENARIOS.values()))
    rows = "".join(
        f'<tr><td class="mono">{_esc(r.id)}</td><td class="mono">{_esc(r.canary)}</td></tr>'
        for r in report.rows[:6]
    )
    return (
        '<div class="traj"><div class="traj-head"><b>Contamination canaries</b>'
        '<span class="badge soft">embed &amp; grep</span></div>'
        '<p class="dim-text">A unique token per scenario. Embed it in the pack, then search a '
        "model's outputs or a training corpus for it — a hit means the pack was memorized, not "
        "solved.</p>"
        f'<table class="cmp"><tr><th>scenario</th><th>canary</th></tr>{rows}</table>'
        '<p class="dim-text" style="margin-top:12px">For a contamination-resistant score, '
        "<code>agentsynth pack contamination --held-out</code> rewrites the labels into "
        "isomorphic siblings a memorizing model can't match.</p></div>"
    )


_CONVO_DEMO = Scenario(
    id="refund-then-cancel",
    task="Refund order 7.",
    metadata={"user_turns": ["Thanks — actually, please also cancel order 8."]},
    environment={
        "type": "sql",
        "schema": "CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT)",
        "table": "orders",
        "rows": [[7, "paid"], [8, "paid"]],
    },
    checkers=[
        SqlCheck(query="SELECT status FROM orders WHERE id=7", equals=[["refunded"]]),
        SqlCheck(query="SELECT status FROM orders WHERE id=8", equals=[["cancelled"]]),
    ],
)


def do_conversation() -> str:
    """A multi-turn user-simulator conversation, graded on the end state (P2.3 / usersim)."""
    from agentsynth.usersim import run_conversation

    def policy(observation: str, ctx: Any) -> dict:
        if ctx.step_count == 0:
            sql = (
                "UPDATE orders SET status='refunded' WHERE id=7"
                if ctx.turn == 0
                else "UPDATE orders SET status='cancelled' WHERE id=8"
            )
            return {"tool_name": "sql_query", "arguments": {"query": sql}}
        return {"answer": "Done — anything else?"}

    result = run_conversation(policy, _CONVO_DEMO)
    badge = (
        '<span class="badge pass">PASS</span>'
        if result.passed
        else '<span class="badge fail">FAIL</span>'
    )
    turns = "".join(
        f'<div class="ocheck"><span class="oname">turn {i + 1}</span>'
        f'<span class="odetail"><b>user:</b> {_esc(t.user)}<br><b>agent:</b> {_esc(t.agent)} '
        f'<span class="dim-text">({t.tool_calls} tool call'
        f'{"" if t.tool_calls == 1 else "s"})</span></span></div>'
        for i, t in enumerate(result.turns)
    )
    return (
        '<div class="traj"><div class="traj-head"><b>Conversation</b> '
        f'{badge}<span class="dim-text" style="margin-left:auto">{result.n_turns} turns, '
        "graded on the end state</span></div>"
        f"{turns}"
        '<p class="dim-text" style="margin-top:10px">One persistent world across the whole '
        "exchange — fix turn one but break it on turn three and the run still fails. τ²-bench "
        "style, via <code>run_conversation</code>.</p></div>"
    )


def _default_agent_view() -> str:
    """A ready-made expert episode so the lead tab shows the thesis before any click."""
    try:
        first = next(iter(sorted(_DEMO_SCENARIOS)))
        return do_agent_run(first, next(iter(DEMO_POLICIES)), MOCK_LABEL)
    except Exception:
        return "_Pick a scenario and policy, then run the episode._"


def _verdict_card(ev: Any) -> str:
    """The judge verdict as a scorecard: overall, pass badge, one bar per dimension."""
    badge = (
        '<span class="badge pass">PASS</span>'
        if ev.passed
        else '<span class="badge fail">FAIL</span>'
    )
    flat = ev.flat()
    bars = []
    for dim in RUBRIC_DIMENSIONS:
        score = float(flat.get(dim, 0.0))
        bars.append(
            f'<div class="vdim"><span class="vname">{_esc(dim)}</span>'
            f'<span class="vtrack"><span class="vfill" style="width:{score:.0%}"></span></span>'
            f'<span class="vnum">{score:.2f}</span></div>'
        )
    explanation = ""
    if getattr(ev, "explanation", None):
        explanation = f'<div class="vwhy">{_esc(_truncate(ev.explanation, 260))}</div>'
    return (
        '<div class="verdict">'
        f'<div class="vhead">Judge verdict <b>{ev.overall:.3f}</b> {badge}</div>'
        f'<div class="vdims">{"".join(bars)}</div>{explanation}</div>'
    )


def render_trajectory_detail(traj: Any, eval_results: Optional[List[Any]] = None) -> str:
    """The full step timeline for one trajectory, verdict card on top once it's scored."""
    if traj is None:
        return "_Click a row in the batch overview to inspect a trajectory._"
    tree = render_tree(traj)
    ev = _eval_by_id(eval_results).get(traj.id)
    if ev is None:
        return tree
    return _verdict_card(ev) + tree


def eval_rows(eval_results: List[Any]) -> List[List[Any]]:
    """Dataframe rows of (trajectory_id, overall, passed, + the 6 dims)."""
    rows: List[List[Any]] = []
    for ev in eval_results or []:
        flat = ev.flat()
        row: List[Any] = [
            flat.get("trajectory_id", ""),
            round(float(flat.get("overall", 0.0)), 3),
            bool(flat.get("passed", False)),
        ]
        for dim in RUBRIC_DIMENSIONS:
            row.append(round(float(flat.get(dim, 0.0)), 3))
        rows.append(row)
    return rows


_EVAL_HEADERS = ["trajectory_id", "overall", "passed"] + list(RUBRIC_DIMENSIONS)
_OVERVIEW_HEADERS = ["#", "mode", "domain", "steps", "score", "tools", "answer"]


_PLOT_COLORWAY = ["#4f46e5", "#818cf8", "#0f9d58", "#f59e0b", "#ef4444", "#64748b"]
_PLOT_FONT = "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif"


def _brand(fig, dark: bool = False):
    """One look for every chart, matched to the page theme the client reported."""
    ink = "#e6e9f0" if dark else "#11141a"
    bg = "#14161c" if dark else "#ffffff"
    grid = "#272b36" if dark else "#e7e9ee"
    accent = "#6366f1" if dark else "#4f46e5"
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        colorway=_PLOT_COLORWAY,
        font=dict(family=_PLOT_FONT, size=13, color=ink),
        title_font=dict(family=_PLOT_FONT, size=15, color=ink),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        margin=dict(l=44, r=24, t=52, b=44),
    )
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid)
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid)
    if getattr(fig.layout, "polar", None) and fig.layout.polar.radialaxis:
        fig.update_polars(
            bgcolor=bg,
            radialaxis=dict(gridcolor=grid, linecolor=grid),
            angularaxis=dict(gridcolor=grid, linecolor=grid),
        )
    for trace in fig.data:
        kind = getattr(trace, "type", "")
        if kind in ("bar", "histogram"):
            trace.update(marker_color=accent, marker_line_width=0)
        elif kind == "scatterpolar":
            trace.update(line_color=accent, fillcolor="rgba(99,102,241,0.25)")
        elif kind == "box":
            trace.update(line_color=accent)
        elif kind == "indicator":
            trace.update(
                gauge_bar_color=accent,
                gauge_bordercolor=grid,
                number_font_color=ink,
            )
    return fig


def _empty_fig(title: str = "No data yet", dark: bool = False):
    """Blank Plotly figure to show before any data exists."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(title=title)
    return _brand(fig, dark)


def _kpi_cards(metrics: Dict[str, Any]) -> str:
    """The dataset numbers as a row of stat cards instead of a skinny table."""

    def pct(v: Any) -> str:
        return "—" if v is None else f"{float(v):.0%}"

    def num(v: Any, nd: int = 2) -> str:
        return "—" if v is None else f"{float(v):.{nd}f}"

    cards = [
        ("Trajectories", str(metrics.get("num_trajectories", 0))),
        ("Evaluated", str(metrics.get("num_evaluated", 0))),
        ("Pass@1", pct(metrics.get("pass_rate"))),
        ("Avg overall", num(metrics.get("avg_overall"), 3)),
        ("Avg steps", num(metrics.get("avg_steps"), 1)),
        ("Avg tool calls", num(metrics.get("avg_tool_calls"), 1)),
        ("Unique tools", str(metrics.get("unique_tools", 0))),
        ("Tool coverage", pct(metrics.get("tool_coverage", 0.0))),
        ("Diversity", num(metrics.get("diversity_score", 0.0), 2)),
    ]
    items = "".join(
        f'<div class="kpi"><div class="kpi-v">{_esc(v)}</div>'
        f'<div class="kpi-l">{_esc(k)}</div></div>'
        for k, v in cards
    )
    return f'<div class="kpis">{items}</div>'


def _fig_dim_spread(eval_results: Optional[List[Any]], dark: bool = False):
    """Score distribution per rubric dimension — the variance the means hide."""
    import plotly.graph_objects as go

    fig = go.Figure()
    if eval_results:
        for dim in RUBRIC_DIMENSIONS:
            vals = [float(ev.flat().get(dim, 0.0)) for ev in eval_results]
            fig.add_trace(
                go.Box(
                    y=vals,
                    name=dim.replace("_", " "),
                    boxmean=True,
                    line_color="#4f46e5",
                    fillcolor="rgba(79,70,229,0.15)",
                    marker=dict(color="#4f46e5", size=4, opacity=0.6),
                )
            )
        fig.update_layout(
            title="Rubric spread per dimension", showlegend=False, yaxis_range=[-0.05, 1.05]
        )
    else:
        fig.update_layout(title="Rubric spread — run the judge first")
    return _brand(fig, dark)


def _fig_score_vs_steps(
    trajectories: Optional[List[Any]], eval_results: Optional[List[Any]], dark: bool = False
):
    """Overall score against trajectory length — does rambling cost quality?"""
    import plotly.graph_objects as go

    fig = go.Figure()
    by_id = _eval_by_id(eval_results)
    points = [
        (t.num_steps(), float(by_id[t.id].overall), bool(by_id[t.id].passed), t.id)
        for t in trajectories or []
        if t.id in by_id
    ]
    if points:
        for passed, color, label in ((True, "#0f9d58", "pass"), (False, "#ef4444", "fail")):
            group = [p for p in points if p[2] is passed]
            if group:
                fig.add_trace(
                    go.Scatter(
                        x=[p[0] for p in group],
                        y=[p[1] for p in group],
                        mode="markers",
                        name=label,
                        text=[p[3] for p in group],
                        marker=dict(size=9, color=color, opacity=0.75),
                    )
                )
        fig.update_layout(
            title="Score vs trajectory length",
            xaxis_title="steps",
            yaxis_title="overall",
            yaxis_range=[-0.05, 1.05],
        )
    else:
        fig.update_layout(title="Score vs length — run the judge first")
    return _brand(fig, dark)


def _five_empty_figs(dark: bool = False) -> Tuple[Any, Any, Any, Any, Any]:
    return (
        _empty_fig("Rubric radar — no data", dark),
        _empty_fig("Score distribution — no data", dark),
        _empty_fig("Rubric spread — no data", dark),
        _empty_fig("Score vs length — no data", dark),
        _empty_fig("Tool usage — no data", dark),
    )


def do_generate(
    query: str,
    tools_json: str,
    mode: str,
    num_trajectories: float,
    temperature: float,
    max_steps: float,
    vary_modes: bool,
    model_choice: str,
    progress=gr.Progress(),
):
    """Generate a batch of trajectories; update the Generate tab and State."""
    empty_overview = gr.update(value=[], headers=_OVERVIEW_HEADERS)
    try:
        query = (query or "").strip()
        if not query:
            gr.Warning("Please enter a user query first.")
            return ("⚠️ Please enter a user query first.", "", empty_overview, [])

        tools = parse_tool_catalog(tools_json)
        if not tools:
            gr.Warning("Could not parse the tool catalog — falling back to defaults.")
            tools = default_tool_catalog()

        n = max(1, int(num_trajectories))
        model = _model_arg(model_choice)

        gen = AgentTrajectoryGenerator(
            model=model,
            temperature=float(temperature),
            max_steps=int(max_steps),
            use_mock="auto",
        )

        def _progress(frac: float, desc: str = "") -> None:
            try:
                progress(frac, desc=desc)
            except Exception:
                pass

        trajectories = gen.generate_batch(
            query,
            tools=tools,
            mode=mode,
            num_trajectories=n,
            progress=_progress,
            vary_modes=bool(vary_modes),
        )

        backend = getattr(gen, "client", None)
        backend_model = getattr(backend, "model", None)
        using_llm = bool(getattr(gen, "use_llm", False))
        backend_label = f"LLM `{backend_model}`" if using_llm and backend_model else "offline mock"
        warning = getattr(gen, "warning", None)

        status = (
            f"✅ Generated **{len(trajectories)}** trajectories ({backend_label}) from your query."
        )
        if warning:
            status += f"\n\n⚠️ {warning}"

        tree = render_tree(trajectories[0]) if trajectories else "_No trajectories._"
        overview = gr.update(value=traj_overview_rows(trajectories), headers=_OVERVIEW_HEADERS)
        return (status, tree, overview, trajectories)

    except Exception as exc:
        gr.Warning(f"Generation failed: {exc}")
        return (f"❌ Generation failed: {exc}", "", empty_overview, [])


def do_select_trajectory(
    trajectories: Optional[List[Any]],
    eval_results: Optional[List[Any]],
    overview: Any,
    evt: gr.SelectData,
) -> str:
    """Render the trajectory for the clicked overview row — its true index is column 0."""
    if not trajectories:
        return "_Generate a batch first, then click a row to inspect it._"
    try:
        row = evt.index[0] if evt is not None and evt.index else 0
        if overview is not None and hasattr(overview, "iloc"):
            idx = int(overview.iloc[row, 0])
        elif overview is not None and len(overview):
            idx = int(overview[row][0])
        else:
            idx = int(row)
    except Exception:
        idx = 0
    if idx < 0 or idx >= len(trajectories):
        return "_That row is out of range — re-run the filter and try again._"
    return render_trajectory_detail(trajectories[idx], eval_results)


def do_filter_overview(
    trajectories: Optional[List[Any]],
    eval_results: Optional[List[Any]],
    mode: str,
    min_score: float,
):
    """Rebuild the overview table from the mode / min-score filters."""
    rows = filter_overview_rows(trajectories, eval_results, mode, min_score)
    return gr.update(value=rows, headers=_OVERVIEW_HEADERS)


def do_evaluate(
    trajectories: Optional[List[Any]],
    model_choice: str,
    pass_threshold: float,
    progress=gr.Progress(),
):
    """Run the LLM-as-Judge over the trajectories held in State."""
    empty_table = gr.update(value=[], headers=_EVAL_HEADERS)
    if not trajectories:
        msg = "Generate trajectories first (Generate tab), then run the judge here."
        return (empty_table, msg, [], gr.update())

    try:
        model = _model_arg(model_choice)
        evaluator = TrajectoryEvaluator(
            model=model,
            use_mock="auto",
            pass_threshold=float(pass_threshold),
        )

        def _progress(frac: float, desc: str = "") -> None:
            try:
                progress(frac, desc=desc)
            except Exception:
                pass

        results = evaluator.evaluate_batch(trajectories, progress=_progress)

        table = gr.update(value=eval_rows(results), headers=_EVAL_HEADERS)

        n_pass = sum(1 for r in results if r.passed)
        total = len(results) or 1
        pass_at_1 = n_pass / total
        judge_model = results[0].judge_model if results else "mock"

        lines = [
            f"### Judge results  ·  judge: `{judge_model}`",
            "",
            f"- **pass@1:** {pass_at_1:.0%}  ({n_pass}/{len(results)} passed, "
            f"threshold = {float(pass_threshold):.2f})",
            "",
            "**Sample explanations:**",
        ]
        for r in results[:3]:
            verdict = "✅ pass" if r.passed else "❌ fail"
            lines.append(
                f"- `{r.trajectory_id}` — overall **{r.overall:.3f}** ({verdict}): "
                f"{_truncate(r.explanation, 220)}"
            )
        overview = gr.update(
            value=traj_overview_rows(trajectories, results), headers=_OVERVIEW_HEADERS
        )
        return (table, "\n".join(lines), results, overview)

    except Exception as exc:
        gr.Warning(f"Evaluation failed: {exc}")
        return (empty_table, f"❌ Evaluation failed: {exc}", [], gr.update())


def do_metrics(
    trajectories: Optional[List[Any]],
    eval_results: Optional[List[Any]],
    dark: bool = False,
):
    """Compute dataset metrics and render the summary plus five plots."""
    if not trajectories:
        msg = "Generate trajectories first to see dataset metrics."
        return (msg, *_five_empty_figs(dark))

    try:
        metrics = compute_dataset_metrics(trajectories, eval_results or None)
        summary = _kpi_cards(metrics)

        radar = _brand(M.plot_rubric_radar(eval_results or None), dark)
        dist = _brand(M.plot_score_distribution(eval_results or None), dark)
        spread = _fig_dim_spread(eval_results or None, dark)
        scatter = _fig_score_vs_steps(trajectories, eval_results or None, dark)
        tools = _brand(M.plot_tool_usage(trajectories), dark)
        return (summary, radar, dist, spread, scatter, tools)

    except Exception as exc:
        gr.Warning(f"Metrics failed: {exc}")
        return (f"❌ Metrics failed: {exc}", *_five_empty_figs(dark))


def do_export(
    trajectories: Optional[List[Any]],
    fmt: str,
):
    """Write the dataset to a tempfile; return a download and a text preview."""
    if not trajectories:
        return (
            gr.update(value=None, visible=False),
            "Generate trajectories first, then build a dataset file here.",
        )

    fmt = (fmt or "jsonl").strip().lower()
    suffix = {"jsonl": ".jsonl", "sharegpt": ".json", "adp": ".json"}.get(fmt, ".jsonl")

    try:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, prefix="agentsynth_", delete=False, encoding="utf-8"
        )
        path = tmp.name
        tmp.close()

        save_dataset(trajectories, path, fmt=fmt)

        if fmt == "jsonl":
            with open(path, "r", encoding="utf-8") as fh:
                head = [next_line.rstrip("\n") for _, next_line in zip(range(25), fh)]
            preview = "\n".join(head)
        else:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            preview = json.dumps(data[:3], indent=2, ensure_ascii=False)

        if not preview:
            preview = "(empty)"

        return (
            gr.update(value=path, visible=True),
            preview,
        )

    except Exception as exc:
        gr.Warning(f"Export failed: {exc}")
        return (gr.update(value=None, visible=False), f"❌ Export failed: {exc}")


def do_preview(trajectories: Optional[List[Any]]):
    """First few trajectories as plain dicts for the JSON viewer."""
    if not trajectories:
        return {"message": "Generate some trajectories first to preview the dataset."}
    preview = []
    for traj in trajectories[:5]:
        try:
            preview.append(traj.model_dump(mode="json"))
        except Exception:
            preview.append(json.loads(traj.model_dump_json()))
    return {"count": len(trajectories), "showing": len(preview), "trajectories": preview}


_INTRO_MD = """\
Watch agents work **outcome-checked** worlds — a run passes only when the database,
API, or sandbox ends in the goal state, so an agent can't talk its way to a score.
Compare policies on a pack, then generate, judge, and export trajectories for
fine-tuning. **Offline by default** — no keys needed; set a provider key
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …) as an environment variable / Space
secret to switch on a real LLM.
"""

_FUNNEL_MD = (
    "**Put your own model on the [live leaderboard](https://agentsynth.tech/leaderboard):** "
    "`pip install agentsynth-ai && agentsynth bench --pack core_v2 --model <id> --submit`"
)

_RUBRIC_MD = """\
Every trajectory gets a score in **[0, 1]** per dimension, combined into a
weighted `overall`; it **passes** when `overall ≥ pass_threshold`.

| Dimension | What it measures |
| --- | --- |
| **task_completion** | Did the final answer actually solve the user's query? |
| **tool_correctness** | Were valid tools called with all required, well-formed args? |
| **faithfulness** | Is the answer grounded in the observations / code output (no hallucination)? |
| **reasoning_coherence** | Did the agent reason before acting, with a clean step order? |
| **efficiency** | Did it reach the answer without redundant or repeated tool calls? |
| **safety** | Absence of dangerous content (destructive commands, leaked secrets, etc.). |
"""

_EXPORT_MD = """\
- **jsonl** — one HF / TRL / Unsloth-friendly record per line (`messages`,
  `tools`, `steps`, `final_answer`, …). The canonical, round-trippable format.
- **sharegpt** — `{"conversations": [...], "tools": [...]}` records.
- **adp** — Agent Data Protocol (`adp/0.1`) style records.
"""

_HEADER_HTML = """\
<div id="as-header">
  <span class="brand">Agent<b>Synth</b><span class="tag">playground</span></span>
  <span class="links">
    <a href="https://agentsynth.tech" target="_blank" rel="noopener">agentsynth.tech</a>
    <a href="https://agentsynth.tech/leaderboard" target="_blank" rel="noopener">Leaderboard</a>
    <a href="https://agentsynth.github.io/agentsynth/" target="_blank" rel="noopener">Docs</a>
    <a href="https://github.com/agentsynth/agentsynth" target="_blank" rel="noopener">GitHub</a>
  </span>
</div>
"""

_CSS = """
.gradio-container{max-width:1200px !important;margin:0 auto !important}
footer{display:none !important}

/* mobile: keep everything inside the viewport. Gradio's flex layers default to
   min-width:auto, so the 6-tab nav and the inputs impose a wider-than-screen
   min-width and the whole page overflows. Let every layer shrink (the tabs scroll),
   stack the control rows, wrap long SQL/code, and clip any residual bleed. */
@media (max-width:700px){
  .gradio-container{max-width:100% !important;
    padding-left:8px !important;padding-right:8px !important}
  body{overflow-x:hidden}
  main.fillable,main.app,.gradio-container .contain,.gradio-container .column,
  .gradio-container .block,.gradio-container .form,.gradio-container .wrap{
    min-width:0 !important;max-width:100% !important}
  .tabs,.tab-wrapper,.tab-container,[role="tablist"]{
    min-width:0 !important;overflow-x:auto !important}
  input,textarea,select{min-width:0 !important;max-width:100% !important}
  .as-controls{flex-direction:column !important;flex-wrap:wrap !important}
  .as-controls > *{width:100% !important;min-width:0 !important}
  .traj,.verdict,.step-body,.mono{min-width:0 !important;word-break:break-word}
  .codeblock,pre{white-space:pre-wrap !important;word-break:break-word}
  #as-header{padding:12px 2px 10px}
  #as-header .links a{margin-left:12px;font-size:13px}
}

/* the JSON editor stretches to the bottom of the settings column and scrolls inside */
#tool-catalog{flex:1 1 0;min-height:240px;display:flex;flex-direction:column}
#tool-catalog > .wrap:last-child{flex:1;min-height:0;display:flex;flex-direction:column}
#tool-catalog .codemirror-wrapper{flex:1;min-height:0}
#tool-catalog .cm-editor{height:100%;max-height:none}
#tool-catalog .cm-scroller{overflow:auto}
#traj-detail{max-height:760px;overflow-y:auto}

#kpi-summary{width:100%}
.kpis{display:grid;gap:12px;margin:6px 0 4px;
  grid-template-columns:repeat(auto-fit,minmax(110px,1fr))}
.kpi{background:#fff;border:1px solid #e7e9ee;border-radius:12px;padding:14px 16px}
.kpi-v{font-size:22px;font-weight:800;letter-spacing:-.02em;color:#11141a}
.kpi-l{font-size:12px;color:#5b6471;margin-top:2px}

#as-header{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;
  gap:8px;padding:14px 4px 12px;margin-bottom:4px;
  border-bottom:1px solid var(--border-color-primary)}
#as-header .brand{font-size:19px;font-weight:700;letter-spacing:-.01em}
#as-header .brand b{color:#4f46e5}
#as-header .tag{margin-left:10px;font-size:12px;font-weight:600;color:#4f46e5;
  background:#eef0fe;padding:3px 10px;border-radius:999px;vertical-align:2px}
#as-header .links a{margin-left:18px;font-size:14px;color:#5b6471;text-decoration:none}
#as-header .links a:hover{color:#4f46e5}

.traj{background:#fff;color:#11141a;border:1px solid #e7e9ee;border-radius:12px;padding:16px 18px}
.traj-head{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px}
.traj-id{font:12px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:#5b6471;
  background:#f4f5f8;border:1px solid #e7e9ee;padding:2px 8px;border-radius:6px}
.badge{font-size:11px;font-weight:700;letter-spacing:.04em;color:#4f46e5;
  background:#eef0fe;padding:3px 9px;border-radius:999px;text-transform:uppercase}
.badge.soft{color:#5b6471;background:#f4f5f8}
.badge.pass{color:#0f9d58;background:#e7f6ee}
.badge.fail{color:#dc2626;background:#fdecec}
.traj-query{font-size:15px;font-weight:600;margin:2px 0 14px;color:#11141a}
.steps{list-style:none;margin:0;padding:0;position:relative}
.steps:before{content:"";position:absolute;left:31px;top:10px;bottom:10px;width:2px;background:#eef0fe}
.step{display:flex;gap:12px;align-items:flex-start;padding:7px 0;position:relative}
.chip{flex:0 0 64px;text-align:center;font-size:10px;font-weight:800;letter-spacing:.06em;
  padding:4px 0;border-radius:6px;position:relative;z-index:1}
.chip.tool{background:#4f46e5;color:#fff}
.chip.obs{background:#f4f5f8;color:#5b6471;border:1px solid #e7e9ee}
.chip.code{background:#0e1117;color:#a6e3a1}
.chip.think{background:#eef0fe;color:#4f46e5}
.chip.answer{background:#0f9d58;color:#fff}
.step-body{flex:1;min-width:0;font-size:14px;line-height:1.55;padding-top:2px;color:#11141a}
.step-body .agent{display:inline-block;margin-right:8px;
  font:11px ui-monospace,SFMono-Regular,monospace;
  color:#4f46e5;background:#eef0fe;border-radius:5px;padding:1px 7px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px;
  color:#11141a;word-break:break-word}
.mono b{color:#11141a}
.dim-text{color:#5b6471}
.answer-text{color:#11141a;font-weight:600}
.codeblock{background:#0e1117;color:#e6edf3;border-radius:8px;padding:10px 12px;margin:4px 0 6px;
  font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  overflow-x:auto;white-space:pre}
.traj-final{margin-top:12px;padding:10px 14px;border-left:3px solid #0f9d58;background:#f3faf6;
  border-radius:0 8px 8px 0;font-size:14px;color:#11141a}

.verdict{background:#fff;color:#11141a;border:1px solid #e7e9ee;border-radius:12px;
  padding:14px 18px;margin-bottom:12px}
.vhead{font-size:14px;color:#5b6471;display:flex;align-items:center;gap:10px;margin-bottom:10px}
.vhead b{font-size:20px;color:#11141a}
.vdims{display:grid;grid-template-columns:1fr 1fr;gap:6px 22px}
.vdim{display:flex;align-items:center;gap:10px;font-size:12.5px}
.vname{flex:0 0 150px;color:#5b6471;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.vtrack{flex:1;height:6px;background:#eef0fe;border-radius:999px;overflow:hidden}
.vfill{display:block;height:100%;background:#4f46e5;border-radius:999px}
.vnum{flex:0 0 34px;text-align:right;font:12px ui-monospace,SFMono-Regular,monospace;color:#11141a}
.vwhy{margin-top:10px;font-size:13px;color:#5b6471}
@media(max-width:760px){.vdims{grid-template-columns:1fr}.vname{flex-basis:120px}}

.traj table.cmp{width:100%;border-collapse:collapse}
.cmp td{padding:8px 6px;border-bottom:1px solid #e7e9ee;text-align:center;
  font-size:13.5px;color:#11141a}
.cmp th{padding:8px 6px;border-bottom:1px solid #e7e9ee;text-align:center;font-size:11px;
  text-transform:uppercase;letter-spacing:.05em;color:#5b6471}
.cmp td:first-child{text-align:left}
.cmp th:first-child{text-align:left}
.cmp-ok{color:#0f9d58;font-weight:800}
.cmp-fail{color:#dc2626;font-weight:800}

.ocheck{display:flex;align-items:flex-start;gap:10px;padding:6px 0;font-size:13.5px}
.omark{flex:0 0 20px;text-align:center;font-weight:800;border-radius:5px}
.omark.ok{color:#0f9d58}
.omark.fail{color:#dc2626}
.oname{flex:0 0 150px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:12.5px;color:#11141a}
.odetail{flex:1;color:#5b6471}

.repro{display:flex;align-items:flex-start;gap:9px;margin:-2px 0 14px;padding:11px 14px;
  background:#f3faf6;border:1px solid #cdeada;border-radius:10px;font-size:13px;color:#11141a}
.repro-ic{color:#0f9d58;font-weight:800;font-size:15px;line-height:1.35}
.repro code{font:12px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#e7f6ee;
  border-radius:5px;padding:1px 6px;color:#0b7a44}
.repro-sub{color:#5b6471;font-size:12px}

/* dark theme: the cards go dark too instead of floating as white islands.
   One selector per rule — gradio's css scoper mangles comma lists under .dark. */
.dark .traj{background:#14161c;border-color:#272b36}
.dark .verdict{background:#14161c;border-color:#272b36}
.dark .traj-query{color:#e6e9f0}
.dark .step-body{color:#e6e9f0}
.dark .answer-text{color:#e6e9f0}
.dark .mono{color:#e6e9f0}
.dark .mono b{color:#e6e9f0}
.dark .vhead{color:#9aa3b2}
.dark .vhead b{color:#e6e9f0}
.dark .vnum{color:#e6e9f0}
.dark .dim-text{color:#9aa3b2}
.dark .vname{color:#9aa3b2}
.dark .vwhy{color:#9aa3b2}
.dark .traj-id{background:#1b1e27;border-color:#272b36;color:#9aa3b2}
.dark .badge{background:rgba(99,102,241,.18);color:#a5b4fc}
.dark .badge.soft{background:#1b1e27;color:#9aa3b2}
.dark .badge.pass{background:rgba(16,185,129,.15);color:#34d399}
.dark .badge.fail{background:rgba(239,68,68,.15);color:#f87171}
.dark .chip.obs{background:#1b1e27;color:#9aa3b2;border-color:#272b36}
.dark .chip.think{background:rgba(99,102,241,.18);color:#a5b4fc}
.dark .steps:before{background:#272b36}
.dark .traj-final{background:rgba(16,185,129,.08);color:#e6e9f0}
.dark .repro{background:rgba(16,185,129,.08);border-color:#1f6f4a;color:#e6e9f0}
.dark .repro code{background:rgba(16,185,129,.16);color:#34d399}
.dark .repro-sub{color:#9aa3b2}
.dark .vtrack{background:#272b36}
.dark #as-header .links a{color:#9aa3b2}
.dark #as-header .links a:hover{color:#a5b4fc}
.dark #as-header .tag{background:rgba(99,102,241,.18)}
.dark .kpi{background:#14161c;border-color:#272b36}
.dark .kpi-v{color:#e6e9f0}
.dark .kpi-l{color:#9aa3b2}
.dark .oname{color:#e6e9f0}
.dark .odetail{color:#9aa3b2}
.dark .cmp td{color:#e6e9f0;border-color:#272b36}
.dark .cmp th{color:#9aa3b2;border-color:#272b36}
"""

_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.gray,
    font=["-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "Helvetica", "sans-serif"],
    font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
)

# Gradio 6 moved theme/css from Blocks() to launch(); support both.
_GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6
_STYLE_KW = {"theme": _THEME, "css": _CSS}
_BLOCKS_KW = {} if _GRADIO_6 else dict(_STYLE_KW)


with gr.Blocks(title="AgentSynth — playground", **_BLOCKS_KW) as demo:
    gr.HTML(_HEADER_HTML)
    gr.Markdown(_INTRO_MD)

    # State holds the live Python objects, not serialized copies.
    traj_state = gr.State([])  # List[Trajectory]
    eval_state = gr.State([])  # List[EvalResult]

    # The page reports its theme on load so server-rendered charts can match it.
    dark_flag = gr.Checkbox(value=False, visible=False)
    demo.load(
        None,
        inputs=None,
        outputs=[dark_flag],
        js="() => document.body.classList.contains('dark')",
    )

    with gr.Tab("Agent runs"):
        gr.Markdown(
            "**Watch an agent work an outcome-checked world.** A run passes only when "
            "the world ends in the goal state — the lazy talker scores zero, the expert "
            "earns it. Scripted policies run offline; pick the LLM policy plus a model "
            "id to watch a real model try."
        )
        with gr.Row(elem_classes=["as-controls"]):
            agent_scenario = gr.Dropdown(
                choices=sorted(_DEMO_SCENARIOS),
                value=next(iter(sorted(_DEMO_SCENARIOS)), None),
                label="Scenario (core_v2)",
                scale=2,
            )
            agent_policy = gr.Dropdown(
                choices=list(DEMO_POLICIES) + [_LLM_POLICY_LABEL],
                value=next(iter(DEMO_POLICIES)),
                label="Policy",
                scale=2,
            )
            agent_model = gr.Dropdown(
                choices=_MODEL_CHOICES,
                value=MOCK_LABEL,
                label="Model (LLM policy only)",
                allow_custom_value=True,
                scale=2,
            )
        agent_task = gr.Markdown(do_agent_task(next(iter(sorted(_DEMO_SCENARIOS)), "")))
        agent_btn = gr.Button("Run the episode", variant="primary", elem_id="agent-btn")
        agent_view = gr.Markdown(_default_agent_view())
        gr.Markdown(_FUNNEL_MD)

        agent_scenario.change(do_agent_task, inputs=[agent_scenario], outputs=[agent_task])
        agent_btn.click(
            do_agent_run,
            inputs=[agent_scenario, agent_policy, agent_model],
            outputs=[agent_view],
        )

    with gr.Tab("Compare"):
        gr.Markdown(
            "Line policies up against the whole pack — the CLI's `bench --compare`, in the "
            "browser. Two trials by default, so flaky wins don't count. `bench --trials` adds "
            "the full reliability picture: the pass^1→pass^k decay curve, a Wilson confidence "
            "interval, and which scenarios are flaky rather than cleanly passing."
        )
        with gr.Row(elem_classes=["as-controls"]):
            cmp_policies = gr.CheckboxGroup(
                choices=list(DEMO_POLICIES),
                value=list(DEMO_POLICIES),
                label="Scripted policies",
                scale=3,
            )
            cmp_model = gr.Dropdown(
                choices=_MODEL_CHOICES,
                value=MOCK_LABEL,
                label="Add an LLM (optional)",
                allow_custom_value=True,
                scale=2,
            )
            cmp_trials = gr.Slider(1, 3, value=2, step=1, label="Trials (pass^k)", scale=1)
        cmp_btn = gr.Button("Run the comparison", variant="primary", elem_id="cmp-btn")
        cmp_view = gr.Markdown("_Pick at least two policies, then run._")
        gr.Markdown(_FUNNEL_MD)

        cmp_btn.click(do_compare, inputs=[cmp_policies, cmp_model, cmp_trials], outputs=[cmp_view])

    with gr.Tab("Robustness"):
        gr.Markdown(
            "**Can this benchmark be trusted?** Two failure modes: a model that *games* the "
            "checkers without solving the task, and a pack that *leaked* into training. Audit "
            "both — `agentsynth pack audit` and `pack contamination`, in the browser."
        )
        with gr.Row(elem_classes=["as-controls"]):
            rob_btn = gr.Button("Audit for gaming", variant="primary", elem_id="rob-btn")
            contam_btn = gr.Button("Contamination canaries", elem_id="contam-btn")
        rob_view = gr.Markdown("_Run the adversaries over the demo pack to see its robustness._")
        gr.Markdown(_FUNNEL_MD)
        rob_btn.click(do_robustness, inputs=None, outputs=[rob_view])
        contam_btn.click(do_contamination, inputs=None, outputs=[rob_view])

    with gr.Tab("Code"):
        gr.Markdown(
            "**Code graded by hidden tests** — the agent writes Python in the sandbox, then a "
            "test it never sees runs against it. It passes only if the code works, not if the "
            "transcript claims it does. This is the `CodeCheck` checker (pack `code_v1`)."
        )
        code_choice = gr.Radio(
            ["correct solution", "buggy solution"],
            value="correct solution",
            label="What does the agent submit?",
        )
        code_btn = gr.Button("Run the hidden tests", variant="primary")
        code_view = gr.Markdown(do_code_demo("correct"))
        code_btn.click(do_code_demo, inputs=[code_choice], outputs=[code_view])

    with gr.Tab("Conversation"):
        gr.Markdown(
            "**A multi-turn conversation, graded on the end state** (τ²-bench style). The user "
            "asks across turns; the world persists; the checkers run once at the end — so an "
            "agent that fixes turn one but forgets turn two fails. This is `run_conversation`."
        )
        convo_btn = gr.Button("Run the conversation", variant="primary")
        convo_view = gr.Markdown(do_conversation())
        convo_btn.click(do_conversation, inputs=None, outputs=[convo_view])

    with gr.Tab("Generate"):
        with gr.Row(equal_height=True):
            with gr.Column(scale=7):
                gen_query = gr.Textbox(
                    label="User query",
                    lines=2,
                    value="What's the weather in Paris and what's 18% tip on a $54 bill?",
                )
                gen_tools = gr.Code(
                    label="Tool catalog (JSON)",
                    language="json",
                    value=_DEFAULT_CATALOG_JSON,
                    max_lines=18,
                    elem_id="tool-catalog",
                )
            with gr.Column(scale=5):
                gen_mode = gr.Dropdown(
                    ["single_agent", "multi_agent", "code_execution"],
                    value="single_agent",
                    label="Mode",
                )
                gen_model = gr.Dropdown(
                    choices=_MODEL_CHOICES,
                    value=_DEFAULT_MODEL_CHOICE,
                    label="Generator model",
                    allow_custom_value=True,
                    info="Offline mock by default — pick or type a model id for a real LLM.",
                )
                with gr.Accordion("Generation settings", open=True):
                    gen_num = gr.Slider(
                        1, 1000, value=10, step=1, label="Trajectories", info="Batch size."
                    )
                    gen_temp = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
                    gen_max_steps = gr.Slider(1, 12, value=6, step=1, label="Max steps")
                    gen_vary = gr.Checkbox(
                        label="Vary modes across the batch (diversity)", value=False
                    )

        gen_btn = gr.Button("Generate batch", variant="primary", size="lg", elem_id="gen-btn")

        gr.Examples(
            label="Try one",
            examples=[
                [
                    "What's the weather in Paris and what's 18% tip on a $54 bill?",
                    "single_agent",
                ],
                [
                    "Analyze last quarter's sales from the database and "
                    "email a summary to the team.",
                    "multi_agent",
                ],
                [
                    "Compute the mean and standard deviation of 12, 19, 7, 22, 31.",
                    "code_execution",
                ],
                [
                    "Read report.csv and tell me the total revenue.",
                    "single_agent",
                ],
            ],
            inputs=[gen_query, gen_mode],
        )

        gen_status = gr.Markdown("Ready. Configure your query and click **Generate batch**.")

        gr.Markdown("#### Batch explorer")
        with gr.Row(equal_height=False):
            with gr.Column(scale=6):
                with gr.Row():
                    ov_mode = gr.Dropdown(
                        ["all", "single_agent", "multi_agent", "code_execution"],
                        value="all",
                        label="Mode",
                        scale=1,
                    )
                    ov_min_score = gr.Slider(
                        0.0, 1.0, value=0.0, step=0.05, label="Min score", scale=2
                    )
                gen_overview = gr.Dataframe(
                    headers=_OVERVIEW_HEADERS,
                    label="Click a row to inspect it",
                    wrap=True,
                    interactive=False,
                    max_height=760,
                    column_widths=["6%", "15%", "12%", "9%", "10%", "22%", "26%"],
                )
            with gr.Column(scale=6):
                gen_tree = gr.Markdown(
                    value="_Generate a batch, then click a row in the overview to inspect it._",
                    label="Selected trajectory",
                    elem_id="traj-detail",
                )

        gen_btn.click(
            do_generate,
            inputs=[
                gen_query,
                gen_tools,
                gen_mode,
                gen_num,
                gen_temp,
                gen_max_steps,
                gen_vary,
                gen_model,
            ],
            outputs=[gen_status, gen_tree, gen_overview, traj_state],
        )

        gen_overview.select(
            do_select_trajectory,
            inputs=[traj_state, eval_state, gen_overview],
            outputs=[gen_tree],
        )
        # filters apply as they change — no apply button to hunt for
        ov_mode.change(
            do_filter_overview,
            inputs=[traj_state, eval_state, ov_mode, ov_min_score],
            outputs=[gen_overview],
        )
        ov_min_score.release(
            do_filter_overview,
            inputs=[traj_state, eval_state, ov_mode, ov_min_score],
            outputs=[gen_overview],
        )

    with gr.Tab("Evaluate"):
        with gr.Row():
            eval_model = gr.Dropdown(
                choices=_MODEL_CHOICES,
                value=_DEFAULT_MODEL_CHOICE,
                label="Judge model",
                allow_custom_value=True,
                info="Offline heuristic judge by default.",
                scale=2,
            )
            eval_threshold = gr.Slider(
                0.0, 1.0, value=0.6, step=0.01, label="Pass threshold", scale=1
            )
        with gr.Accordion("What the judge scores", open=False):
            gr.Markdown(_RUBRIC_MD)
        eval_btn = gr.Button("Run the judge", variant="primary", size="lg", elem_id="eval-btn")

        eval_summary = gr.Markdown(
            "Generate trajectories on the **Generate** tab, then run the judge here."
        )
        eval_table = gr.Dataframe(
            headers=_EVAL_HEADERS,
            label="Per-trajectory scores",
            wrap=True,
            interactive=False,
        )

        eval_btn.click(
            do_evaluate,
            inputs=[traj_state, eval_model, eval_threshold],
            outputs=[eval_table, eval_summary, eval_state, gen_overview],
        )

    with gr.Tab("Metrics"):
        with gr.Row():
            metrics_btn = gr.Button(
                "Refresh metrics", variant="primary", elem_id="metrics-btn", scale=0
            )
        metrics_summary = gr.Markdown(
            "Click **Refresh metrics** after generating (and optionally evaluating).",
            elem_id="kpi-summary",
        )
        with gr.Row():
            with gr.Column():
                plot_radar = gr.Plot(label="Rubric radar (mean per dimension)")
            with gr.Column():
                plot_spread = gr.Plot(label="Rubric spread per dimension")
        with gr.Row():
            with gr.Column():
                plot_dist = gr.Plot(label="Overall score distribution")
            with gr.Column():
                plot_scatter = gr.Plot(label="Score vs trajectory length")
        with gr.Row():
            plot_tools = gr.Plot(label="Tool usage (top 15)")

        metrics_btn.click(
            do_metrics,
            inputs=[traj_state, eval_state, dark_flag],
            outputs=[
                metrics_summary,
                plot_radar,
                plot_dist,
                plot_spread,
                plot_scatter,
                plot_tools,
            ],
        )

    with gr.Tab("Export"):
        gr.Markdown(_EXPORT_MD)
        with gr.Row(equal_height=True):
            export_fmt = gr.Dropdown(
                ["jsonl", "sharegpt", "adp"], value="jsonl", label="Format", scale=3
            )
            export_btn = gr.Button("Build dataset file", variant="primary", scale=1)

        export_file = gr.File(label="Download", visible=False)
        export_preview = gr.Code(label="Preview (first records)", language="json")

        with gr.Accordion("Raw trajectory objects", open=False):
            preview_btn = gr.Button("Load preview")
            preview_json = gr.JSON(label="First five trajectories")

        export_btn.click(
            do_export,
            inputs=[traj_state, export_fmt],
            outputs=[export_file, export_preview],
        )
        preview_btn.click(
            do_preview,
            inputs=[traj_state],
            outputs=[preview_json],
        )


if __name__ == "__main__":
    demo.launch(**(_STYLE_KW if _GRADIO_6 else {}))
