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
    TrajectoryEvaluator,
    compute_dataset_metrics,
    default_tool_catalog,
    parse_tool_catalog,
    save_dataset,
)
from agentsynth import metrics as M
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
_OVERVIEW_HEADERS = ["idx", "mode", "domain", "steps", "score", "tools_used", "final_answer"]


_PLOT_COLORWAY = ["#4f46e5", "#818cf8", "#0f9d58", "#f59e0b", "#ef4444", "#64748b"]
_PLOT_FONT = "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif"


def _brand(fig):
    """One look for every chart: white, indigo, system font, tight margins."""
    fig.update_layout(
        template="plotly_white",
        colorway=_PLOT_COLORWAY,
        font=dict(family=_PLOT_FONT, size=13, color="#11141a"),
        title_font=dict(family=_PLOT_FONT, size=15, color="#11141a"),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=44, r=24, t=52, b=44),
    )
    for trace in fig.data:
        kind = getattr(trace, "type", "")
        if kind in ("bar", "histogram"):
            trace.update(marker_color="#4f46e5", marker_line_width=0)
        elif kind == "scatterpolar":
            trace.update(line_color="#4f46e5", fillcolor="rgba(79,70,229,0.22)")
        elif kind == "indicator":
            trace.update(
                gauge_bar_color="#4f46e5",
                gauge_bordercolor="#e7e9ee",
                number_font_color="#11141a",
            )
    return fig


def _empty_fig(title: str = "No data yet"):
    """Blank Plotly figure to show before any data exists."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(title=title)
    return _brand(fig)


def _five_empty_figs() -> Tuple[Any, Any, Any, Any, Any]:
    return (
        _empty_fig("Rubric radar — no data"),
        _empty_fig("Score distribution — no data"),
        _empty_fig("Pass@1 — no data"),
        _empty_fig("Tool usage — no data"),
        _empty_fig("Step distribution — no data"),
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
):
    """Compute dataset metrics and render the summary plus five plots."""
    if not trajectories:
        msg = "Generate trajectories first to see dataset metrics."
        return (msg, *_five_empty_figs())

    try:
        metrics = compute_dataset_metrics(trajectories, eval_results or None)
        summary = M.metrics_summary_md(metrics)

        radar = _brand(M.plot_rubric_radar(eval_results or None))
        dist = _brand(M.plot_score_distribution(eval_results or None))
        gauge = _brand(M.plot_pass_gauge(eval_results or None))
        tools = _brand(M.plot_tool_usage(trajectories))
        steps = _brand(M.plot_step_distribution(trajectories))
        return (summary, radar, dist, gauge, tools, steps)

    except Exception as exc:
        gr.Warning(f"Metrics failed: {exc}")
        return (f"❌ Metrics failed: {exc}", *_five_empty_figs())


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
Generate synthetic multi-step agent trajectories, judge them on a six-dimension
rubric, inspect the dataset, and export it for fine-tuning. **Offline by
default** — no keys needed; set a provider key (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, …) as a Space secret to switch on a real LLM.
"""

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

#tool-catalog .cm-editor{max-height:380px}
#tool-catalog .cm-scroller{overflow:auto}

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

    with gr.Tab("Generate"):
        with gr.Row(equal_height=False):
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
                gen_btn = gr.Button(
                    "Generate batch", variant="primary", size="lg", elem_id="gen-btn"
                )

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
                        scale=2,
                    )
                    ov_min_score = gr.Slider(
                        0.0, 1.0, value=0.0, step=0.05, label="Min judge score", scale=3
                    )
                    ov_filter_btn = gr.Button("Filter", scale=1)
                gen_overview = gr.Dataframe(
                    headers=_OVERVIEW_HEADERS,
                    label="Click a row to inspect it",
                    wrap=True,
                    interactive=False,
                    max_height=460,
                    column_widths=["7%", "16%", "12%", "9%", "10%", "22%", "24%"],
                )
            with gr.Column(scale=6):
                gen_tree = gr.Markdown(
                    value="_Generate a batch, then click a row in the overview to inspect it._",
                    label="Selected trajectory",
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
        ov_filter_btn.click(
            do_filter_overview,
            inputs=[traj_state, eval_state, ov_mode, ov_min_score],
            outputs=[gen_overview],
        )

    with gr.Tab("Evaluate"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                eval_model = gr.Dropdown(
                    choices=_MODEL_CHOICES,
                    value=_DEFAULT_MODEL_CHOICE,
                    label="Judge model",
                    allow_custom_value=True,
                    info="Offline heuristic judge by default.",
                )
                eval_threshold = gr.Slider(0.0, 1.0, value=0.6, step=0.01, label="Pass threshold")
                eval_btn = gr.Button(
                    "Run the judge", variant="primary", size="lg", elem_id="eval-btn"
                )
            with gr.Column(scale=7):
                with gr.Accordion("What the judge scores", open=False):
                    gr.Markdown(_RUBRIC_MD)

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
            "Click **Refresh metrics** after generating (and optionally evaluating)."
        )
        with gr.Row():
            with gr.Column():
                plot_radar = gr.Plot(label="Rubric radar (mean per dimension)")
            with gr.Column():
                plot_gauge = gr.Plot(label="Pass@1 gauge")
        with gr.Row():
            with gr.Column():
                plot_dist = gr.Plot(label="Overall score distribution")
            with gr.Column():
                plot_steps = gr.Plot(label="Steps per trajectory")
        with gr.Row():
            plot_tools = gr.Plot(label="Tool usage (top 15)")

        metrics_btn.click(
            do_metrics,
            inputs=[traj_state, eval_state],
            outputs=[
                metrics_summary,
                plot_radar,
                plot_dist,
                plot_gauge,
                plot_tools,
                plot_steps,
            ],
        )

    with gr.Tab("Export"):
        gr.Markdown(_EXPORT_MD)
        with gr.Row():
            export_fmt = gr.Dropdown(
                ["jsonl", "sharegpt", "adp"], value="jsonl", label="Format", scale=2
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
