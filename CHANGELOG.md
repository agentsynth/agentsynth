# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `BrowserEnvironment` (`environments/`): a headless-Chromium backend (Playwright) for
  grounded web tool-use — `browser_navigate`, `browser_read`, `browser_links`,
  `browser_find`, `browser_click`. Install with `agentsynth-ai[browser]` plus a one-time
  `playwright install chromium` (Python 3.10+).

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
