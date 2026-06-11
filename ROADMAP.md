# Roadmap

This is where AgentSynth is headed. It's a direction, not a contract — order and
scope will shift as we learn. If something here matters to you, open an issue or a
Discussion; the items marked **good first issue** are a nice way in.

The north star: **the number of high-quality, verified agent trajectories people
generate with this, and the number of models that get measurably better because of
them.** Everything below serves that.

## Where we are (v0.3.0)

The engine works offline end to end, and the proof is public:

- [x] Three generation modes: single-agent tool use, multi-agent, grounded code execution
- [x] LLM-as-Judge eval loop over a six-dimension rubric, with a deterministic fallback
- [x] Real environments: SQLite, Python sandbox, any MCP server, a headless browser
- [x] Verification (re-run code, tool args, safety) + a learned verifier distilled from the judge
- [x] A published dataset ([agentsynth-trajectories](https://huggingface.co/datasets/agentsynth/agentsynth-trajectories))
      and a reproducible before/after fine-tune ([docs/BENCHMARK.md](docs/BENCHMARK.md))
- [x] Dataset metrics + dashboards, batch explorer in the app, JSONL/ShareGPT/ADP/Parquet export
- [x] Gradio app + HF Space, CLI, offline test suite, CI + coverage

Mock generation is the default; a provider key switches on real-LLM generation.

## Real generation

Move past simulated observations to executed ones. The first cut shipped in
v0.1.x: a `sql_query` tool that runs against a real in-memory SQLite, a `python`
tool that runs in an isolated subprocess, a seed-task taxonomy, and YAML recipes.

- [x] `environments/`: pluggable execution backends — SQLite (`sql_query`) and an
      isolated Python subprocess (`python`) are in
- [x] A browser environment (headless Chromium via Playwright)
- [x] A Docker sandbox for untrusted generated code (`DockerSandbox`)
- [x] A REST environment: any OpenAPI spec becomes runnable tools (`RestEnvironment`)
- [ ] More environments: a hardened sandbox (gVisor/e2b-style), a virtual filesystem
- [x] A seed-task taxonomy spanning domains and modes
- [x] Scenarios with outcome checkers: seedable worlds + end-state assertions, packs in YAML
- [x] Failure mining: categorize benchmark/judge misses and aim the next run at them
      (`mine_failures` → `recipe_from_failures` — the flywheel's last leg)
- [x] Query evolution: template or LLM-paraphrase expansion (`evolve_queries`)
- [ ] Persona and environment injection to fight mode collapse
- [x] Concurrent batches via a recipe runner (`max_workers`)
- [x] Cost tracking, caching, budget caps, and resumable runs (`agentsynth.scale`)
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
- [x] A learned verifier: distill the judge into a cheap classifier (`train_learned_verifier`)
- [ ] Judge calibration against human labels
- [x] Preference pairs (chosen / rejected) for DPO, with TRL-compatible export
- [x] Near-duplicate removal (Jaccard shingles) and benchmark decontamination
- [x] MinHash/LSH dedup for the 100k scale
- [ ] Embedding dedup and end-to-end provenance tracking

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
- [x] A BFCL adapter (`load_bfcl`) + bundled real slices (simple & multiple splits) and a τ-bench bridge (`run_tau_bench`)
- [x] Run it for real: a public dataset on the Hub and a published before/after table
      (0% → 58.3% tool selection from a base 1B — see docs/BENCHMARK.md); scaling the
      dataset to 10k+ with real-LLM generation is next
- [ ] API-Bank and more benchmark adapters
- [x] A docs site (mkdocs-material + mkdocstrings API reference), deployed to GitHub Pages
- [ ] More tutorials and how-to guides on the docs site

## Ecosystem

- [x] **MCP-native generation** — point AgentSynth at any MCP server (stdio or HTTP)
      and its tools become a live environment (`MCPEnvironment`)
- [x] OpenAPI / REST connector (`RestEnvironment`)
- [x] **RL-native**: gym-style episodes with verified rewards over any environment,
      TRL-compatible reward functions, and an OpenEnv bridge (`agentsynth.rl`)
- [x] Trace importers: OpenAI / Anthropic / OpenTelemetry GenAI logs become verifiable trajectories
- [ ] Connectors: LangChain and LlamaIndex tool definitions
- [ ] A plugin interface for custom generators, judges, and environments

## How to help

Pick up a **good first issue**, improve a tool catalog, file a bug with a clean repro,
or write a tutorial. If you want to take on something bigger from this list, say so on
the issue first so we don't duplicate work.
