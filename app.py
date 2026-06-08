"""Gradio web UI for AgentSynth.

Four tabs: generate trajectories, evaluate them with the LLM-as-Judge, view
dataset metrics, and export. Importing this module builds `demo` but never
touches an LLM; `demo.launch()` only runs under __main__.
"""

from __future__ import annotations

import json
import tempfile
from typing import Any, List, Optional, Tuple

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

_STEP_EMOJI = {
    "thought": "💭",
    "plan": "🗺️",
    "tool_call": "🛠️",
    "observation": "👁️",
    "code_execution": "🐍",
    "critique": "🧐",
    "final_answer": "✅",
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


def render_tree(traj: Any) -> str:
    """Render a trajectory as an indented emoji tree (Markdown)."""
    if traj is None:
        return "_No trajectory to preview yet._"

    lines: List[str] = []
    lines.append(f"### 🌳 Trajectory `{traj.id}`  ·  mode: **{traj.mode}**")
    if getattr(traj, "domain", None):
        lines.append(f"_domain: {traj.domain}_")
    lines.append("")
    lines.append(f"**❓ Query:** {traj.query}")
    lines.append("")

    for i, step in enumerate(traj.steps):
        emoji = _STEP_EMOJI.get(step.step_type, "•")
        agent = f" `[{step.agent}]`" if getattr(step, "agent", None) else ""
        header = f"- {emoji} **{step.step_type}**{agent}"

        if step.step_type == "tool_call":
            args = step.tool_args or {}
            try:
                args_str = json.dumps(args, ensure_ascii=False)
            except Exception:
                args_str = str(args)
            lines.append(f"{header} — `{step.tool_name}({args_str})`")
        elif step.step_type == "observation":
            lines.append(header)
            lines.append(f"    - 📄 {_truncate(step.observation, 160)}")
        elif step.step_type == "code_execution":
            lines.append(header)
            code = (step.code or "").strip()
            if code:
                lines.append("")
                lines.append("    ```python")
                for code_line in code.splitlines():
                    lines.append(f"    {code_line}")
                lines.append("    ```")
            out = (step.code_output or "").strip()
            if out:
                lines.append(f"    - 📤 output: `{_truncate(out, 160)}`")
        elif step.step_type in ("thought", "plan", "critique"):
            text = step.thought or step.content or ""
            lines.append(f"{header} — {_truncate(text, 200)}")
        elif step.step_type == "final_answer":
            text = step.content or traj.final_answer or ""
            lines.append(f"{header} — {_truncate(text, 240)}")
        else:
            lines.append(f"{header} — {_truncate(step.short(), 160)}")

    lines.append("")
    if traj.final_answer:
        lines.append(f"**🏁 Final answer:** {traj.final_answer}")
    return "\n".join(lines)


def traj_overview_rows(trajectories: List[Any]) -> List[List[Any]]:
    """One row per trajectory for the batch overview table."""
    rows: List[List[Any]] = []
    for idx, traj in enumerate(trajectories or []):
        tools_used = ", ".join(traj.tool_names_used()) or "—"
        rows.append(
            [
                idx,
                traj.mode,
                traj.num_steps(),
                tools_used,
                _truncate(traj.final_answer, 80),
            ]
        )
    return rows


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
_OVERVIEW_HEADERS = ["idx", "mode", "num_steps", "tools_used", "final_answer"]


def _empty_fig(title: str = "No data yet"):
    """Blank Plotly figure to show before any data exists."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(template="plotly_white", title=title)
    return fig


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


def do_evaluate(
    trajectories: Optional[List[Any]],
    model_choice: str,
    pass_threshold: float,
    progress=gr.Progress(),
):
    """Run the LLM-as-Judge over the trajectories held in State."""
    empty_table = gr.update(value=[], headers=_EVAL_HEADERS)
    if not trajectories:
        msg = "🪄 Generate some trajectories first (Generate tab), then run the judge here."
        return (empty_table, msg, [])

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
            f"### 🧑‍⚖️ LLM-as-Judge results  ·  judge: `{judge_model}`",
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
        return (table, "\n".join(lines), results)

    except Exception as exc:
        gr.Warning(f"Evaluation failed: {exc}")
        return (empty_table, f"❌ Evaluation failed: {exc}", [])


def do_metrics(
    trajectories: Optional[List[Any]],
    eval_results: Optional[List[Any]],
):
    """Compute dataset metrics and render the summary plus five plots."""
    if not trajectories:
        msg = "🪄 Generate some trajectories first to see dataset metrics."
        return (msg, *_five_empty_figs())

    try:
        metrics = compute_dataset_metrics(trajectories, eval_results or None)
        summary = M.metrics_summary_md(metrics)

        radar = M.plot_rubric_radar(eval_results or None)
        dist = M.plot_score_distribution(eval_results or None)
        gauge = M.plot_pass_gauge(eval_results or None)
        tools = M.plot_tool_usage(trajectories)
        steps = M.plot_step_distribution(trajectories)
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
            "🪄 Generate some trajectories first, then build a dataset file here.",
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


_HEADER_MD = """\
# 🤖 AgentSynth
### Synthetic Agentic Trajectories Generator + LLM-as-Judge Eval Loop

Generate high-quality multi-step agent trajectories (tool-use, code-execution,
multi-agent), score them with a 6-dimension LLM-as-Judge rubric, inspect
dataset metrics, and export to JSONL / ShareGPT / ADP for fine-tuning.

> 🟢 **Runs fully offline** in deterministic *mock* mode — no API keys, no
> network required. Set a provider key (e.g. `ANTHROPIC_API_KEY`,
> `OPENAI_API_KEY`) as a **Space secret** to switch on a real LLM generator and
> LLM-as-Judge.
"""

_RUBRIC_MD = """\
### 🧑‍⚖️ The 6 rubric dimensions

The judge scores every trajectory on each dimension in **[0, 1]**, then
combines them into a weighted `overall` score:

| Dimension | What it measures |
| --- | --- |
| **task_completion** | Did the final answer actually solve the user's query? |
| **tool_correctness** | Were valid tools called with all required, well-formed args? |
| **faithfulness** | Is the answer grounded in the observations / code output (no hallucination)? |
| **reasoning_coherence** | Did the agent reason before acting, with a clean step order? |
| **efficiency** | Did it reach the answer without redundant or repeated tool calls? |
| **safety** | Absence of dangerous content (destructive commands, leaked secrets, etc.). |

A trajectory **passes** when `overall ≥ pass_threshold`.
"""

_EXPORT_MD = """\
### 📦 Export formats

- **jsonl** — one HF / TRL / Unsloth-friendly record per line (`messages`,
  `tools`, `steps`, `final_answer`, …). The canonical, round-trippable format.
- **sharegpt** — `{"conversations": [...], "tools": [...]}` records.
- **adp** — Agent Data Protocol (`adp/0.1`) style records.
"""


with gr.Blocks(theme=gr.themes.Soft(), title="AgentSynth") as demo:
    gr.Markdown(_HEADER_MD)

    # State holds the live Python objects, not serialized copies.
    traj_state = gr.State([])  # List[Trajectory]
    eval_state = gr.State([])  # List[EvalResult]

    with gr.Tab("🧪 Generate"):
        with gr.Row():
            with gr.Column(scale=3):
                gen_query = gr.Textbox(
                    label="User query",
                    lines=2,
                    value="What's the weather in Paris and what's 18% tip on a $54 bill?",
                )
                gen_tools = gr.Code(
                    label="Tool catalog (JSON)",
                    language="json",
                    value=_DEFAULT_CATALOG_JSON,
                )
            with gr.Column(scale=2):
                gen_mode = gr.Dropdown(
                    ["single_agent", "multi_agent", "code_execution"],
                    value="single_agent",
                    label="Mode",
                )
                gen_num = gr.Slider(1, 1000, value=10, step=1, label="num_trajectories")
                gen_temp = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="temperature")
                gen_max_steps = gr.Slider(1, 12, value=6, step=1, label="max_steps")
                gen_vary = gr.Checkbox(label="Vary modes across the batch (diversity)", value=False)
                gen_model = gr.Dropdown(
                    choices=_MODEL_CHOICES,
                    value=_DEFAULT_MODEL_CHOICE,
                    label="Generator model (optional)",
                    allow_custom_value=True,
                    info="Defaults to offline mock. Pick or type a model id to use a real LLM.",
                )

        gen_btn = gr.Button("Generate", variant="primary")

        gr.Examples(
            label="Example (query, mode) pairs",
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

        gen_status = gr.Markdown("Ready. Configure your query and click **Generate**.")
        gen_tree = gr.Markdown(label="First trajectory preview")
        gen_overview = gr.Dataframe(
            headers=_OVERVIEW_HEADERS,
            label="Batch overview",
            wrap=True,
            interactive=False,
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

    with gr.Tab("🧑‍⚖️ Evaluate"):
        gr.Markdown(_RUBRIC_MD)
        with gr.Row():
            eval_model = gr.Dropdown(
                choices=_MODEL_CHOICES,
                value=_DEFAULT_MODEL_CHOICE,
                label="Judge model (optional)",
                allow_custom_value=True,
                info="Defaults to offline mock heuristic judge.",
            )
            eval_threshold = gr.Slider(0.0, 1.0, value=0.6, step=0.01, label="pass_threshold")
        eval_btn = gr.Button("Run LLM-as-Judge eval", variant="primary")

        eval_table = gr.Dataframe(
            headers=_EVAL_HEADERS,
            label="Per-trajectory scores",
            wrap=True,
            interactive=False,
        )
        eval_summary = gr.Markdown(
            "Generate trajectories on the **Generate** tab, then run the judge here."
        )

        eval_btn.click(
            do_evaluate,
            inputs=[traj_state, eval_model, eval_threshold],
            outputs=[eval_table, eval_summary, eval_state],
        )

    with gr.Tab("📊 Metrics"):
        metrics_btn = gr.Button("Refresh metrics", variant="primary")
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

    with gr.Tab("📦 Dataset / Export"):
        gr.Markdown(_EXPORT_MD)
        with gr.Row():
            export_fmt = gr.Dropdown(
                ["jsonl", "sharegpt", "adp"], value="jsonl", label="Export format"
            )
            export_btn = gr.Button("Build dataset file", variant="primary")

        export_file = gr.File(label="Download dataset", visible=False)
        export_preview = gr.Code(label="Preview (first records)", language="json")

        gr.Markdown("#### 🔍 Current dataset (first few trajectories)")
        preview_btn = gr.Button("Preview trajectories")
        preview_json = gr.JSON(label="Trajectory objects")

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
    demo.launch()
