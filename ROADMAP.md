# Roadmap

This is where AgentSynth is headed. It's a direction, not a contract — order and
scope will shift as we learn. If something here matters to you, open an issue or a
Discussion; the items marked **good first issue** are a nice way in.

The north star: **the number of high-quality, verified agent trajectories people
generate with this, and the number of models that get measurably better because of
them.** Everything below serves that.

## Where we are (v0.1.0)

The foundation is in place and works offline end to end:

- [x] Three generation modes: single-agent tool use, multi-agent, grounded code execution
- [x] LLM-as-Judge eval loop over a six-dimension rubric, with a deterministic fallback
- [x] Dataset metrics + Plotly dashboards
- [x] JSONL / ShareGPT / ADP / Parquet export, JSONL round-trips
- [x] Gradio app, CLI, offline test suite, CI

Today generation is mock-first. The next phases turn it into a real data engine.

## Real generation

Move past simulated observations to executed ones. The first cut shipped in
v0.1.x: a `sql_query` tool that runs against a real in-memory SQLite, a `python`
tool that runs in an isolated subprocess, a seed-task taxonomy, and YAML recipes.

- [x] `environments/`: pluggable execution backends — SQLite (`sql_query`) and an
      isolated Python subprocess (`python`) are in
- [ ] More environments: a hardened sandbox (gVisor/e2b-style), a browser env, a virtual filesystem
- [x] A seed-task taxonomy spanning domains and modes
- [ ] Self-Instruct / Evol-Instruct expansion of the taxonomy for diversity and difficulty
- [ ] Persona and environment injection to fight mode collapse
- [x] Concurrent batches via a recipe runner (`max_workers`)
- [ ] Cost tracking, caching, and resumable runs
- [x] Declarative YAML run recipes
- [ ] Local-model backend (vLLM / Ollama) for cheap bulk generation
- [ ] **good first issue:** more built-in tool catalogs and seed tasks (finance, devops, support, research)

## Verification — the part that makes the data worth training on

The first cut shipped in v0.2.x: re-run verification, a judge ensemble, rubric
presets, DPO pairs, and dedup/decontamination.

- [x] Execution-based verification: re-run `code_execution` steps and confirm the
      recorded output reproduces; plus tool-arg and safety checks
- [ ] More verifiers: API ground-truth, unit-test harnesses for code tasks
- [x] Judge ensembles with an agreement signal, and rubric presets (balanced / strict / lenient / safety_first)
- [ ] Judge calibration against human labels
- [x] Preference pairs (chosen / rejected) for DPO, with TRL-compatible export
- [x] Near-duplicate removal (Jaccard shingles) and benchmark decontamination
- [ ] Stronger dedup (MinHash / embeddings) and end-to-end provenance tracking

## Proof and distribution

The harness shipped in v0.2.x: dataset prep, a function-calling benchmark with
before/after reporting, fine-tune scripts, Hub publishing, and a one-command repro
(`docs/BENCHMARK.md`). The remaining piece is running it on a GPU and publishing the
result.

- [x] Trainer-ready dataset prep (`build_sft_dataset` / `build_dpo_dataset`)
- [x] A built-in function-calling benchmark (`agentsynth.benchmarks`) with before/after tables
- [x] Fine-tune scripts (TRL SFT + DPO, Unsloth-friendly) and a one-command repro guide
- [x] HF Hub dataset push + auto dataset card (`push_dataset`, `dataset_card`)
- [x] A Colab notebook (`notebooks/agentsynth_finetune.ipynb`): generate → SFT → DPO → benchmark on a free T4
- [x] A BFCL adapter (`load_bfcl`) and a τ-bench bridge (`run_tau_bench`)
- [ ] Run it for real: a flagship 10k+ verified dataset on the Hub and a published
      before/after table on BFCL / τ-bench (needs a GPU run)
- [ ] API-Bank and more benchmark adapters
- [x] A docs site (mkdocs-material + mkdocstrings API reference), deployed to GitHub Pages
- [ ] More tutorials and how-to guides on the docs site

## Ecosystem

- [x] **MCP-native generation** — point AgentSynth at any MCP server (stdio or HTTP)
      and its tools become a live environment (`MCPEnvironment`)
- [ ] Connectors: OpenAPI / REST specs, LangChain and LlamaIndex tool definitions
- [ ] A plugin interface for custom generators, judges, and environments

## How to help

Pick up a **good first issue**, improve a tool catalog, file a bug with a clean repro,
or write a tutorial. If you want to take on something bigger from this list, say so on
the issue first so we don't duplicate work.
