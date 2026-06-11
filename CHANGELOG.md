# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.2] - 2026-06-12

### Added

- `agentsynth bench --pack` takes a pack name or URL on top of a file path. Names
  resolve against the local `packs/` directory first and fall back to the hub, so
  a pip-only install can bench `core_v1` with nothing cloned. New `--hub` option
  (default `https://api.agentsynth.tech`); a bare `--submit` posts the run there.

### Fixed

- PyYAML ships with the base install. Scenario packs, recipes, and OpenAPI specs
  parse YAML, and a plain `pip install agentsynth-ai` crashed on all three
  (`ModuleNotFoundError: yaml`) — it was only pulled in by extras.

## [0.6.1] - 2026-06-11

### Fixed

- `agentsynth bench --submit` now sends an `agentsynth/<version>` user-agent. The
  default urllib user-agent is blocked by some WAFs (Cloudflare returns HTTP 403,
  error 1010), which kept submissions from reaching a hub behind a proxy.

## [0.6.0] - 2026-06-11

### Added

- Industrial-scale generation (`agentsynth.scale`): `CachingLLMClient` (disk cache,
  retries with backoff, token/cost meter, hard `budget_usd` cap, rate limiting) and
  `run_resumable` (incremental JSONL + state file — crashed runs continue where they
  stopped, `max_items` enables chunked/cron runs). `dedup_trajectories` gains
  `method="minhash"` LSH for linear-time near-duplicate removal at the 100k scale.
- Scenarios (`agentsynth.scenarios`): a serializable bundle of environment config
  (with per-episode seed state), task, and **outcome checkers** that assert on the
  world's end state — `SqlCheck`, `HttpCheck`, `CalledTool`, `AnswerContains`.
  `AgentGym.from_scenario` makes the outcome the dominant terminal reward
  (0.6/0.2/0.2), `run_scenario_suite` turns a pack into an outcome benchmark, and
  packs round-trip through YAML/JSON. `SQLEnvironment` gains `read_only=False` (for
  scenario-owned worlds) and a `.rows()` helper for checks.

## [0.5.0] - 2026-06-11

### Added

- Reward integrity for the learned verifier: `train_learned_verifier(calibrate=True)`
  runs sigmoid calibration and every report now carries a `brier` score, and
  `route_by_confidence` splits a batch into auto-fail / needs-judge / auto-pass bands
  so the LLM judge is only spent on the borderline cases.
- Trace importers (`agentsynth.importers`): convert real agent logs — OpenAI-style
  `tool_calls` messages and Anthropic `tool_use`/`tool_result` blocks — into
  `Trajectory` objects (`import_traces`, `load_traces_jsonl`, format auto-detected),
  so judging, verification, dedup, failure mining, and SFT/DPO export all apply to
  production traffic.
- Failure mining (`agentsynth.mining`): `mine_failures` categorizes benchmark misses
  (no call / wrong tool / bad arguments), `mine_judge_failures` flags rubric dimensions
  below a threshold, and `recipe_from_failures` turns the report into a verified
  generation run aimed at exactly those gaps — the mine-failures leg of the flywheel.

## [0.4.0] - 2026-06-11

### Added

- An RL layer (`agentsynth.rl`): `AgentGym` runs gym-style episodes over any
  environment — tool calls execute for real and the terminal reward comes from
  verification + the judge; `make_reward_fn` plugs the same checks straight into TRL's
  `GRPOTrainer` as a reward function; `to_openenv` bridges a gym onto the OpenEnv
  standard (`agentsynth-ai[rl]`, Python 3.10+); `episodes_to_grpo_jsonl` exports
  episodes for offline methods.
- `RestEnvironment` (`environments/`): turn any OpenAPI spec (dict, JSON/YAML string,
  file path, or URL) into runnable tools — operations become tools, path/query/JSON-body
  parameters are flattened and routed back to the wire, local `$ref`s resolve, and the
  observations are real HTTP responses. Pure stdlib, no extra dependency; pass
  `methods=("get",)` to expose only reads.

## [0.3.0] - 2026-06-10

### Added

- A learned verifier (`train_learned_verifier` / `LearnedVerifier`): distill the LLM
  judge into a small classifier that screens trajectories in microseconds and reports
  its held-out agreement with the judge. Behind the `learned` extra (scikit-learn).
- The bundled BFCL sample gained the `multiple` split —
  `load_sample_bfcl(split="multiple")` gives 25 real cases with 2-3 candidate
  functions each, so the benchmark exercises tool *selection*, not just formatting.
- Trajectories now carry their verification verdict into the exported JSONL (and back
  on load) when a run uses `verify=True`. Contributed by @Ishant5436 (#21).

## [0.2.1] - 2026-06-09

### Added

- A batch explorer in the Gradio app — click a row in the overview to inspect that
  trajectory's steps and judge verdict, with mode / min-score filters.
- `BrowserEnvironment` (`environments/`): a headless-Chromium backend (Playwright) for
  grounded web tool-use — `browser_navigate`, `browser_read`, `browser_links`,
  `browser_find`, `browser_click`. Install with `agentsynth-ai[browser]` plus a one-time
  `playwright install chromium` (Python 3.10+).
- A bundled, real 25-case BFCL slice (`load_sample_bfcl()`) so the benchmark runs on a
  recognized suite offline, plus runnable `examples/benchmark_bfcl.py` and
  `examples/tau_bench_demo.py`.

### Changed

- The PyPI distribution name is now **`agentsynth-ai`** (the name `agentsynth` was already
  taken). The import package and the CLI are unchanged — still `import agentsynth`.

## [0.2.0] - 2026-06-08

### Added

- Execution environments (`environments/`): `SQLEnvironment` (in-memory SQLite),
  `PythonSandbox` (isolated subprocess), `MCPEnvironment` (any Model Context Protocol
  server, via `agentsynth-ai[mcp]`), `CompositeEnvironment`. Pass `environment=` to the
  generator for real, grounded observations.
- Seed-task taxonomy (`tasks/`) and a deterministic `sample_tasks`.
- Declarative `Recipe` (loadable from YAML) and a concurrent `run_recipe` (`pipelines/`).
- Verification (`verification/`): `verify_trajectory` with execution / tool-arg / safety
  checks, an `EnsembleEvaluator`, and rubric presets (balanced / strict / lenient /
  safety_first).
- Preference pairs for DPO (`build_preference_pairs`, `to_dpo_jsonl`).
- Near-duplicate removal and benchmark decontamination (`dedup_trajectories`, `decontaminate`).
- Training-data prep (`training/`: `build_sft_dataset`, `build_dpo_dataset`), a built-in
  function-calling benchmark (`benchmarks/`: `run_benchmark`, `compare_models`,
  `report_table_md`), and HF Hub publishing (`hub.py`: `push_dataset`, `dataset_card`).
- Fine-tune + benchmark scripts under `scripts/` (SFT, DPO, dataset, benchmark) and a
  reproduction guide (`docs/BENCHMARK.md`).
- BFCL adapter (`load_bfcl`) and a τ-bench bridge (`run_tau_bench`), plus a
  `prompted_model` helper to benchmark any instruction-following model.
- A Colab notebook (`notebooks/agentsynth_finetune.ipynb`) for the full
  generate → SFT → DPO → benchmark flow on a free GPU.
- A documentation site (mkdocs-material + mkdocstrings API reference under `docs/`),
  deployed to GitHub Pages via `.github/workflows/docs.yml`.

### Changed

- Trajectory ids are now derived from the generation seed, so identical inputs
  reproduce the same trajectory (and the same eval scores). JSONL export is
  byte-for-byte reproducible.

## [0.1.0] - 2026-06-08

First public release.

### Added

- `AgentTrajectoryGenerator` — synthetic trajectories in `single_agent`,
  `multi_agent`, and `code_execution` modes. Offline deterministic mock by
  default, with an optional LiteLLM backend (Anthropic / xAI / Groq / OpenAI).
- Grounded code execution: `code_execution` steps run through `PythonREPL`, so the
  recorded output is real stdout.
- `TrajectoryEvaluator` — an LLM-as-Judge loop over a six-dimension rubric (task
  completion, tool correctness, faithfulness, reasoning coherence, efficiency,
  safety) with a deterministic structural fallback.
- Dataset metrics (`compute_dataset_metrics`, `diversity_score`) and Plotly
  dashboards.
- Exporters for JSONL (round-trippable via `load_jsonl`), ShareGPT, ADP, and
  Parquet.
- A four-tab Gradio app (`app.py`) and an `agentsynth` CLI.
- Example datasets under `examples/` and an offline test suite.

[Unreleased]: https://github.com/agentsynth/agentsynth/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/agentsynth/agentsynth/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/agentsynth/agentsynth/releases/tag/v0.1.0
