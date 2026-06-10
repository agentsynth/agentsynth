# Reproducing the benchmark

This is the recipe behind a before/after table: generate a verified dataset with
AgentSynth, fine-tune a small model on it, and score the base model against the
fine-tuned one on the function-calling benchmark.

Everything except the fine-tune runs on CPU. The fine-tune needs a GPU — a free
Colab T4 handles an 8B 4-bit model with LoRA.

**Fastest path:** the Colab notebook [`notebooks/agentsynth_finetune.ipynb`](https://github.com/agentsynth/agentsynth/blob/main/notebooks/agentsynth_finetune.ipynb)
runs the whole flow (generate → SFT → DPO → benchmark) on a free T4. The CLI steps
below are the equivalent.

## Reference run (free Colab T4, 2026-06-10)

`unsloth/Llama-3.2-1B` — a **base** model with no instruction tuning and no
function-calling ability — fine-tuned (4-bit LoRA, 150 steps ≈ 5 minutes) on **275
verified trajectories** distilled from a mock AgentSynth run (318 trajectories,
95.9% verified). Both runs use the same answer-priming, so the comparison is fair.

| Built-in suite (12 cases, pick among 8 tools) | Before | After | Δ |
| --- | --- | --- | --- |
| Tool accuracy | 0.0% | **58.3%** | +58.3 |
| Arg accuracy | 0.0% | **58.3%** | +58.3 |
| Overall | 0.0% | **58.3%** | +58.3 |

| BFCL `multiple` slice (25 real cases, 2-3 candidate functions) | Before | After | Δ |
| --- | --- | --- | --- |
| Tool accuracy | 24.0% | **48.0%** | +24.0 |
| Arg accuracy | 8.0% | **28.0%** | +20.0 |
| Overall | 16.0% | **38.0%** | +22.0 |

The `multiple` split is the meaningful external test — every case offers several
candidate functions the model has never seen, so the number reflects real tool
selection, and it doubles. (The `simple_python` slice moves 60% → 68% tool accuracy,
but with a single candidate function per case it mostly checks formatting under
answer-priming, so we don't lead with it.)

The training data here is the deterministic mock generator. Swapping in a real LLM
generator (set a provider key) produces richer trajectories — especially better
argument values — and is the expected path to stronger numbers.

The source dataset is public: [agentsynth/agentsynth-trajectories](https://huggingface.co/datasets/agentsynth/agentsynth-trajectories).

## 0. Offline smoke (no GPU, no keys)

Confirms the whole pipeline wires up before you spend GPU time:

```bash
python scripts/make_dataset.py --n 20 --vary-modes --verify --dedup --out /tmp/ds
python scripts/run_benchmark.py --model mock
python scripts/train_sft.py --data /tmp/ds/train.jsonl --dry-run
```

## 1. Generate a verified dataset

```bash
# mock by default; set e.g. ANTHROPIC_API_KEY for a real LLM generator
python scripts/make_dataset.py --n 1000 --vary-modes --verify --dedup --rubric strict --out data
```

This writes `data/train.jsonl` (+ a ShareGPT copy and a dataset card).

## 2. Build the SFT and DPO splits

```python
from agentsynth import (
    AgentTrajectoryGenerator, TrajectoryEvaluator, load_jsonl,
    build_sft_dataset, build_preference_pairs, build_dpo_dataset,
)

trajectories = load_jsonl("data/train.jsonl")
build_sft_dataset(trajectories, "data/sft.jsonl")

gen, judge = AgentTrajectoryGenerator(), TrajectoryEvaluator()
pairs = build_preference_pairs(gen, judge, [t.query for t in trajectories][:200], k=6)
build_dpo_dataset(pairs, "data/dpo.jsonl")
```

## 3. Fine-tune (GPU)

```bash
pip install "agentsynth-ai[train]"          # plus `unsloth` for the fast 4-bit path
python scripts/train_sft.py --data data/sft.jsonl --model unsloth/llama-3.1-8b-bnb-4bit --out out/sft
python scripts/train_dpo.py --data data/dpo.jsonl --model out/sft --out out/dpo
```

## 4. Benchmark before vs after

```bash
python scripts/run_benchmark.py --before <base-model> --after <finetuned-model>
```

It prints a markdown table — tool accuracy, arg accuracy, and overall score, with
before / after / Δ. A positive Δ is the headline result.

## Notes

- The built-in benchmark is small, so the harness also runs recognized suites. A real
  25-case slice of the Berkeley Function-Calling Leaderboard (`simple_python`) ships with
  the package: `load_sample_bfcl()` returns cases you can pass straight to `run_benchmark`
  — no download, fully offline (see [`examples/benchmark_bfcl.py`](https://github.com/agentsynth/agentsynth/blob/main/examples/benchmark_bfcl.py)).
  For the full suite, download the official BFCL files and call
  `load_bfcl("BFCL_v4_simple_python.json", "possible_answer/BFCL_v4_simple_python.json")`.
  `run_tau_bench(model=...)` bridges to the official τ-bench harness (multi-turn, so it
  delegates to the `tau-bench` package; see [`examples/tau_bench_demo.py`](https://github.com/agentsynth/agentsynth/blob/main/examples/tau_bench_demo.py)).
  Any model is a `model_fn(query, tools) -> (tool_name, tool_args)`; `prompted_model`
  wraps any text model, and `run_benchmark.py --model <litellm-id>` uses native tool calls.
- Keep the benchmark queries out of the training set — generate training data with
  different seeds, and run `decontaminate` against the benchmark queries to be sure.
