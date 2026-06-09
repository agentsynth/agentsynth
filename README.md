# AgentSynth

> Synthetic agentic trajectories with a built-in LLM-as-Judge eval loop. Generate tool-use, code-execution, and multi-agent training data offline, then score it.

<p align="center">
  <a href="https://github.com/agentsynth/agentsynth/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/agentsynth/agentsynth/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/agentsynth-ai/"><img alt="PyPI" src="https://img.shields.io/pypi/v/agentsynth-ai.svg"></a>
  <a href="https://codecov.io/gh/agentsynth/agentsynth"><img alt="coverage" src="https://codecov.io/gh/agentsynth/agentsynth/branch/main/graph/badge.svg"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python 3.9+" src="https://img.shields.io/badge/python-3.9%2B-blue.svg"></a>
  <a href="https://github.com/agentsynth/agentsynth/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green.svg"></a>
  <a href="https://huggingface.co/spaces/agentsynth/agentsynth"><img alt="Hugging Face Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow.svg"></a>
</p>

<p align="center">
  <a href="https://agentsynth.github.io/agentsynth/">Docs</a> ·
  <a href="docs/VISION.md">Vision</a> ·
  <a href="docs/ARCHITECTURE.md">Architecture</a> ·
  <a href="ROADMAP.md">Roadmap</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

## What it is

AgentSynth generates multi-turn agent trajectories — tool-use, grounded code-execution, and multi-agent collaboration traces — and scores each one with an LLM-as-Judge eval loop. The output is training data for fine-tuning agentic LLMs, built without harvesting real conversations.

What it's good for:

- Bootstrapping an agentic dataset when you have no production traffic, or can't use what you have.
- Running entirely offline. Mock generation and evaluation are deterministic and need no API keys or network access.
- Swapping in a real LLM when you want richer generation and a sharper judge. Claude, Grok, Groq, and OpenAI are all supported through [LiteLLM](https://github.com/BerriAI/litellm).
- Filtering before you train. An 8-metric rubric scores every trajectory, so you can keep the high-signal subset and drop the rest.
- Exporting straight into a training pipeline: JSONL, ShareGPT, and ADP formats load into Hugging Face / TRL / Unsloth / Axolotl without conversion.

Runs are reproducible. Any randomness in the mock paths comes from a stable hash seed, so identical inputs produce identical trajectories.

---

## Live demo

Try it in the browser: [AgentSynth on Hugging Face Spaces](https://huggingface.co/spaces/agentsynth/agentsynth).

Generate a trajectory, watch the judge score it across the rubric dimensions, then export the dataset — all from the Gradio UI.

---

## Features

Core capabilities:

- Synthetic trajectory generation in single-agent, multi-agent, and code-execution modes.
- An LLM-as-Judge eval loop built on a weighted 6-dimension rubric, with a deterministic mock fallback.
- Dataset metrics: aggregate pass@1, per-dimension averages, and trajectory diversity.
- Export to JSONL, ShareGPT, or ADP in a single call.

The eval loop scores six per-trajectory dimensions — task completion, tool correctness, trajectory faithfulness, reasoning coherence / plan adherence, efficiency, and safety — plus two dataset-level metrics, overall pass@1 and diversity.

---

## Install

```bash
# Core library (offline mock generation + eval, exporters, metrics)
pip install agentsynth-ai

# With the Gradio web UI
pip install "agentsynth-ai[app]"

# For running the Hugging Face Space (pins everything the app needs)
pip install -r requirements.txt
```

The core library targets Python 3.9+. The Gradio app wants 3.10+, so use that interpreter if you plan to run the UI locally or on Spaces.

Calling a real LLM also needs `pip install litellm` (already in the `[app]` extra) and the relevant provider key. See [Using a real LLM-as-Judge](#using-a-real-llm-as-judge).

---

## Quickstart

```python
from agentsynth import AgentTrajectoryGenerator, TrajectoryEvaluator, to_jsonl

# 1. Create a generator — offline deterministic mock mode by default.
gen = AgentTrajectoryGenerator()

# 2. Generate a multi-step agent trajectory for a query.
traj = gen.generate("What's the weather in Paris, and is it warmer than Berlin?")

print(f"{traj.num_steps()} steps, tools used: {traj.tool_names_used()}")
print("final answer:", traj.final_answer)

# 3. Evaluate it with the built-in LLM-as-Judge (also mock by default).
result = TrajectoryEvaluator().evaluate(traj)
print(f"overall = {result.overall:.3f}  passed = {result.passed}")
print(result.scores.as_dict())   # all 6 rubric dimensions in [0, 1]

# 4. Export a training-ready dataset.
to_jsonl([traj], "agent_data.jsonl")
```

No keys, no network. Set `AGENTSYNTH_FORCE_MOCK=1` to pin offline behavior even when provider keys are present.

---

## Worked examples

### 1) Single-agent tool use with a custom tool catalog

Pass your own tools through `parse_tool_catalog`. It accepts any JSON-Schema function-calling shape, including raw OpenAI `tools` blocks:

```python
from agentsynth import AgentTrajectoryGenerator, TrajectoryEvaluator, parse_tool_catalog

# A custom catalog — a list of tool dicts (OpenAI/Anthropic style also accepted).
my_tools = parse_tool_catalog([
    {
        "name": "stock_price",
        "description": "Look up the latest stock price for a ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "e.g. 'AAPL'"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "currency_convert",
        "description": "Convert an amount from one currency to another.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "from_ccy": {"type": "string"},
                "to_ccy": {"type": "string"},
            },
            "required": ["amount", "from_ccy", "to_ccy"],
        },
    },
])

gen = AgentTrajectoryGenerator(tools=my_tools)
traj = gen.generate(
    "How much is 100 shares of AAPL worth in euros?",
    mode="single_agent",
    domain="finance",
)

for step in traj.steps:
    print(step.short())

result = TrajectoryEvaluator().evaluate(traj)
print(f"tool_correctness = {result.scores.tool_correctness:.2f}")
```

### 2) Code-execution trace (grounded REPL output)

In `code_execution` mode, the emitted Python actually runs through the sandboxed `PythonREPL`. That means `code_output` is captured stdout, not something the model made up:

```python
from agentsynth import AgentTrajectoryGenerator

gen = AgentTrajectoryGenerator()
traj = gen.generate(
    "Compute the mean and standard deviation of [4, 8, 15, 16, 23, 42].",
    mode="code_execution",
    domain="data_analysis",
)

for step in traj.steps:
    if step.step_type == "code_execution":
        print("CODE:\n", step.code)
        print("OUTPUT (grounded, from the REPL):\n", step.code_output)

print("ANSWER:", traj.final_answer)
```

You can drive the same REPL directly to ground your own snippets:

```python
from agentsynth import PythonREPL

repl = PythonREPL()
print(repl.run("import statistics\nstatistics.pstdev([4, 8, 15, 16, 23, 42])"))
# -> 12.315302134607444   (real stdout; only whitelisted numeric/data imports allowed)
```

### 3) Multi-agent batch + dataset metrics

Generate a batch, set `vary_modes=True` to mix single-agent, multi-agent, and code-execution traces, then evaluate and aggregate:

```python
from agentsynth import (
    AgentTrajectoryGenerator,
    TrajectoryEvaluator,
    compute_dataset_metrics,
    save_dataset,
)

queries = [
    "Plan a 3-day trip to Tokyo on a $1500 budget.",
    "Summarize last quarter's sales from the analytics DB and email the team.",
    "Find the 10th Fibonacci number and explain the recurrence.",
    "What's the weather in Reykjavik and should I pack a coat?",
]

gen = AgentTrajectoryGenerator()
trajectories = gen.generate_batch(queries, vary_modes=True)   # mixes modes per query

evaluator = TrajectoryEvaluator()
results = [evaluator.evaluate(t) for t in trajectories]

# Aggregate quality metrics across the dataset (pass@1, per-dim averages, diversity).
metrics = compute_dataset_metrics(trajectories, results)
print(metrics)

# Ship it (format inferred from the extension).
save_dataset(trajectories, "dataset.jsonl")
```

### 4) Grounded execution with environments and recipes

Attach an environment and the tool calls run for real — `sql_query` hits an
in-memory SQLite database, `python` runs in an isolated subprocess — so the
observations are actual output, not templated text:

```python
from agentsynth import AgentTrajectoryGenerator
from agentsynth.environments import SQLEnvironment

gen = AgentTrajectoryGenerator(environment=SQLEnvironment())
traj = gen.generate("Which product sold the most units?")

for step in traj.steps:
    if step.step_type == "observation":
        print(step.observation)   # a real query result, e.g. "Widget | 2931 ... (3 rows)"
```

A `Recipe` wraps a whole run — generate (optionally concurrent), evaluate,
compute metrics, export — and loads from YAML:

```python
from agentsynth import Recipe, run_recipe

result = run_recipe(Recipe(
    query="Which region had the highest revenue, and how did products compare?",
    num_trajectories=12,
    vary_modes=True,
    environment="sql+python",   # real SQLite + subprocess Python
    export_format="jsonl",
    export_path="dataset.jsonl",
    max_workers=4,
))
print(result.metrics["pass_rate"], "->", result.output_path)
```

Or run a recipe file: `run_recipe(load_recipe("recipes/analytics_sql.yaml"))`.

### 5) Verify trajectories and build DPO pairs

Verification re-runs what it can instead of trusting the model — a `code_execution`
step only passes if its code reproduces the recorded output:

```python
from agentsynth import AgentTrajectoryGenerator, verify_trajectory

traj = AgentTrajectoryGenerator().generate("compute the mean of 4, 8, 15, 16, 23, 42",
                                           mode="code_execution")
result = verify_trajectory(traj)        # tool args + execution + safety checks
print(result.verified, result.detail)   # True 'tool_args: ok; execution: ok; safety: ok'
```

Turn scored trajectories into preference pairs for DPO:

```python
from agentsynth import (
    AgentTrajectoryGenerator, TrajectoryEvaluator, build_preference_pairs, to_dpo_jsonl,
)

pairs = build_preference_pairs(
    AgentTrajectoryGenerator(), TrajectoryEvaluator(),
    "analyze sales by region and email a summary", k=8,
)
to_dpo_jsonl(pairs, "prefs.jsonl")   # {"prompt", "chosen", "rejected", "margin"} per line
```

Recipes can do it all at once — `Recipe(..., verify=True, dedup=True, rubric="strict")`
adds verification, near-duplicate removal, and a stricter judge to the run.

### 6) Generate against a real MCP server

Point AgentSynth at any [Model Context Protocol](https://modelcontextprotocol.io)
server and its tools become a live environment — calls run against the server, so the
observations are real:

```python
import sys
from agentsynth import AgentTrajectoryGenerator
from agentsynth.environments import MCPEnvironment

# A local stdio server here; pass url=... for an HTTP/SSE server instead.
env = MCPEnvironment(command=sys.executable, args=["examples/mcp_server.py"])
gen = AgentTrajectoryGenerator(environment=env)

traj = gen.generate("reverse some text and count its words")
print(traj.tool_names_used())   # tools discovered from the MCP server
env.close()
```

Needs `pip install "agentsynth-ai[mcp]"` (Python 3.10+).

---

## Run the app locally

```bash
pip install "agentsynth-ai[app]"
python app.py
```

Open the printed local URL (usually `http://127.0.0.1:7860`). The UI generates trajectories, shows them step by step, runs the judge, renders the metrics dashboard, and downloads the dataset in any supported format. No keys required.

---

## Fine-tune and benchmark

The point of the data is to make a model better. AgentSynth ships the harness to prove
it: dataset prep, fine-tune scripts (TRL SFT + DPO, Unsloth-friendly), a built-in
function-calling benchmark, and a one-command reproduction. The fine-tune needs a GPU;
everything else runs on CPU.

```bash
# generate data, score a model, dry-run the trainer — all offline
python scripts/make_dataset.py --n 500 --vary-modes --verify --dedup --out data
python scripts/run_benchmark.py --model mock
python scripts/train_sft.py --data data/train.jsonl --dry-run
```

`run_benchmark.py --before <base> --after <finetuned>` prints the before/after table.
Full walkthrough in [docs/BENCHMARK.md](docs/BENCHMARK.md).

---

## Using a real LLM-as-Judge

Generation and evaluation both default to deterministic mock mode. Set any of the provider keys below and AgentSynth upgrades to a real model, auto-detected through [LiteLLM](https://github.com/BerriAI/litellm). It picks a fast, cheap default for whichever key it finds first.

| Provider   | Env var             | Default model used                  |
|------------|---------------------|-------------------------------------|
| Anthropic  | `ANTHROPIC_API_KEY` | `claude-3-5-haiku-latest`           |
| xAI (Grok) | `XAI_API_KEY`       | `xai/grok-2-latest`                 |
| Groq       | `GROQ_API_KEY`      | `groq/llama-3.3-70b-versatile`      |
| OpenAI     | `OPENAI_API_KEY`    | `gpt-4o-mini`                       |

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python app.py            # generation + judge now use Claude
```

```python
from agentsynth import AgentTrajectoryGenerator, TrajectoryEvaluator

# Or pin a model explicitly:
gen = AgentTrajectoryGenerator(model="claude-3-5-haiku-latest")
ev = TrajectoryEvaluator(model="gpt-4o-mini")
```

If LiteLLM isn't installed, no key is set, or a request fails, AgentSynth falls back to mock instead of crashing. Set `AGENTSYNTH_FORCE_MOCK=1` to force offline mode regardless of which keys are present.

---

## Dataset formats

AgentSynth exports three trainer-friendly shapes, all compatible with Hugging Face Datasets, TRL, Unsloth, and Axolotl.

### JSONL

One JSON object per line, holding the full structured trajectory — steps, tools, scores, metadata. Good for archival and custom loaders.

```json
{"id": "a1b2c3d4e5f6", "query": "What's the weather in Paris?", "mode": "single_agent", "domain": "general", "tools": [{"name": "get_weather", "description": "Get the current weather for a given city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}], "steps": [{"step_type": "thought", "thought": "I should look up the weather for Paris."}, {"step_type": "tool_call", "tool_name": "get_weather", "tool_args": {"city": "Paris"}}, {"step_type": "observation", "observation": "Paris: 18C, partly cloudy."}, {"step_type": "final_answer", "content": "It's 18C and partly cloudy in Paris."}], "final_answer": "It's 18C and partly cloudy in Paris.", "success": true, "generator_model": "mock"}
```

### ShareGPT

The familiar `{"conversations": [{"from": "human"/"gpt"/"tool", "value": ...}]}` chat format that Axolotl and Unsloth chat-SFT recipes read natively. Built from each trajectory's `to_messages()` rendering, with user / assistant / tool roles and assistant tool-calls preserved.

### ADP (Agent Data Protocol)

A normalized agent-centric schema that keeps thoughts, tool calls, observations, and code-execution as first-class typed steps. Reach for this when you're training a tool-using or multi-agent policy and want the full trajectory structure rather than a flattened chat log.

```python
from agentsynth import to_jsonl, to_sharegpt, to_adp

to_jsonl(trajectories, "data.jsonl")        # structured trajectories
to_sharegpt(trajectories, "data_sg.json")   # chat SFT
to_adp(trajectories, "data_adp.json")       # agent-protocol records
```

---

## Quality metrics

`TrajectoryEvaluator` scores every trajectory. It produces six rubric dimensions per trajectory, each in `[0, 1]`, and combines them into a weighted overall score. `compute_dataset_metrics` adds two more at the dataset level.

| Metric | Scope | What it measures |
|--------|-------|------------------|
| **Task Completion** | per-traj | Did the trajectory actually solve the user's query? *(weight 0.30)* |
| **Tool Correctness** | per-traj | Were the right tools called with valid, well-typed arguments? *(weight 0.20)* |
| **Trajectory Faithfulness** | per-traj | Is the final answer grounded in the observations / tool outputs (no hallucination)? *(weight 0.15)* |
| **Reasoning Coherence / Plan Adherence** | per-traj | Do the steps follow a logical plan, and does execution match it? *(weight 0.15)* |
| **Efficiency** | per-traj | Was the goal reached without redundant or wasted steps? *(weight 0.10)* |
| **Safety** | per-traj | Does the trajectory avoid unsafe tool use or harmful content? *(weight 0.10)* |
| **Overall pass@1** | dataset | Fraction of trajectories whose weighted overall clears the pass threshold. |
| **Diversity** | dataset | How varied the dataset is across tool-usage signatures, modes, and domains. |

The six per-trajectory weights live in `DEFAULT_RUBRIC_WEIGHTS` and sum to `1.0`. Pass your own to `RubricScores.weighted_overall(weights=...)` to re-balance.

```python
from agentsynth import TrajectoryEvaluator, diversity_score

result = TrajectoryEvaluator().evaluate(traj)
print(result.flat())                 # trajectory_id, overall, passed, judge_model + 6 dims
print(result.explanation)            # human-readable judge rationale

print("dataset diversity:", diversity_score(trajectories))
```

---

## Project structure

```text
AgentSynth/
├── agentsynth/
│   ├── schemas.py          # Pydantic models (Trajectory, ToolSpec, EvalResult, …)
│   ├── utils.py            # tool-catalog parsing, PythonREPL, LLMClient (LiteLLM)
│   ├── generator.py        # AgentTrajectoryGenerator (mock + LLM-backed)
│   ├── evaluator.py        # TrajectoryEvaluator — LLM-as-Judge eval loop
│   ├── metrics.py          # dataset metrics + Plotly dashboards
│   ├── exporters.py        # JSONL / ShareGPT / ADP / Parquet
│   ├── preferences.py      # DPO preference pairs
│   ├── dedup.py            # near-duplicate removal + decontamination
│   ├── hub.py              # push datasets to the Hugging Face Hub
│   ├── cli.py              # the `agentsynth` CLI
│   ├── environments/       # SQL, Python, MCP, composite — run tool calls for real
│   ├── tasks/              # seed-task taxonomy
│   ├── pipelines/          # Recipe + run_recipe (generate → verify → export)
│   ├── verification/       # verifiers, judge ensemble, rubric presets
│   ├── benchmarks/         # function-calling benchmark + before/after reporting
│   └── training/           # SFT / DPO dataset builders
├── app.py                  # Gradio web UI (Hugging Face Space entrypoint)
├── scripts/                # make_dataset / train_sft / train_dpo / run_benchmark
├── examples/               # sample datasets + a demo MCP server
├── docs/                   # VISION, ARCHITECTURE, BENCHMARK, MANIFESTO
├── tests/                  # pytest suite
├── pyproject.toml          # packaging / metadata
└── README.md
```

---

## Deploy to Hugging Face Spaces

1. Create a Space and pick the Gradio SDK.
2. Push this repo to the Space. The entrypoint is `app.py`.
3. `requirements.txt` is auto-detected and installed, so there's no extra build config.
4. Optional: to enable a real LLM judge, add a provider key (for example `ANTHROPIC_API_KEY`) under Settings → Repository secrets. Without one, the Space stays in deterministic mock mode.
5. CPU Basic hardware is enough. Generation and the mock judge need no GPU.

---

## Roadmap

- [ ] More agent personas & domain-specific tool packs.
- [ ] Configurable rubric presets (strict / lenient / safety-focused).
- [ ] Difficulty-aware curriculum sampling for batches.
- [ ] Direct `datasets.Dataset` / `push_to_hub` export helper.
- [ ] Pairwise / preference (DPO-style) trajectory generation.
- [ ] Streaming generation progress in the Gradio UI.

---

## Contributing

Contributions are welcome.

1. Fork the repo and create a feature branch.
2. Keep changes Python 3.9-compatible and add or extend tests under `tests/`.
3. Run the suite: `pytest`.
4. Open a PR with a clear description.

Bug reports, new tool catalogs, and additional export formats all make good first contributions.

---

## License

MIT. See [`LICENSE`](LICENSE) for details.

---

## Citation

If you use AgentSynth in your research or product, please cite it:

```bibtex
@software{agentsynth2026,
  title        = {AgentSynth: Synthetic Agentic Trajectories Generator with an LLM-as-Judge Evaluation Loop},
  author       = {Your Name and Contributors},
  year         = {2026},
  url          = {https://github.com/agentsynth/agentsynth},
  note         = {Open-source library for generating and evaluating synthetic agent trajectories}
}
```

---

<sub>Suggested GitHub topics: `synthetic-data` · `agentic-ai` · `llm-finetuning` · `trajectories` · `tool-use` · `llm-as-judge`</sub>
