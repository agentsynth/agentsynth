# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Reward-hacking / verifier-robustness audit (`agentsynth.robustness`, and
  `agentsynth pack audit`). It measures how gameable a pack's checkers are by running
  trivial adversaries that need no knowledge of the task — a canned answer, an echoed
  prompt, a throwaway tool call — and flags scenarios graded on words rather than a
  state change, plus answer targets that leaked into the prompt. `pack validate` now
  prints a one-line robustness summary. For the generalizing case, `perturb_scenario`
  builds an isomorphic sibling (rename the labels, keep the structure) and `ipt_report`
  confirms a real solver still passes while a replayed transcript no longer does. This
  operationalizes the failure modes from the 2026 "LLMs gaming verifiers" work
  (arXiv:2604.15149). The `core_v2` audit scores 86% — the two refusal scenarios leak
  their keywords, which the report calls out.
- Auto-generated verifiers from a demonstration (`agentsynth.synth`, and
  `agentsynth pack new --from-demo`). Hand it a seed world and the actions that solve a
  task; it runs them, diffs the end state, and writes a scenario whose checkers assert
  exactly what changed — the robust kind, keyed on the primary key, with the row count
  and an untouched witness row pinned against over-mutation. The generated pack validates
  (the actions are the oracle) and audits 100% (every check is on the world). This is the
  cheap-authoring side of the "verifier problem": demonstrate once, the verifier writes
  itself. Multi-table worlds (the data lives in the schema's INSERTs) are handled.
- Pack export to the open RL-environment ecosystems (`agentsynth.pack_export`, and
  `agentsynth pack export --format {verifiers,openenv}`). A pack flows *into* the
  standards everyone is converging on instead of competing with them: a portable
  `scenario_reward` any framework can wrap, an OpenEnv server module (correct by
  construction — it calls our own `to_openenv`), and a Prime Intellect `verifiers`
  environment with `load_environment` whose reward is the same world-state check
  (`reward_from_messages` scores an OpenAI-style completion). `export_pack` writes a
  Hub-ready folder — the module, the bundled pack, a `pyproject.toml`, a README, and a
  framework-neutral `manifest.json` — ready to push to the Environments Hub.
- Reliability statistics beyond a single pass@1 (`agentsynth.reliability`, and the
  `bench --trials k` output). It reports the whole decay curve from pass^1 to pass^k
  (via the unbiased all-must-pass estimator `comb(c,k)/comb(n,k)`), a Wilson confidence
  interval on each — sane at 0%, 100%, and small n where the normal approximation
  collapses — which scenarios are flaky rather than cleanly passing or failing, and
  run throughput. Follows the 2026 reliability-science framing (arXiv:2603.29231). The
  numbers also land in the `--json` report.

## [0.7.4] - 2026-06-12

### Changed

- The playground demo (Agent runs and Compare tabs) and `agentsynth.demo` now run
  on `core_v2` — the harder flagship with conditionals, traps, and multi-table
  consistency. The expert clears it, a mutation-shy agent passes only the four
  read-only scenarios, and the lazy talker scores zero. `core_v1` keeps its own
  oracle (`examples/core_v1_oracle.py`), unchanged.

## [0.7.3] - 2026-06-12

### Added

- `core_v2` — the harder flagship pack (14 scenarios across easy/medium/hard
  tiers): conditional logic, aggregation, traps you must not fall for, and four
  multi-table scenarios that require keeping two tables in agreement (refund and
  restock, cancel and void a payment, return for store credit, reconcile stock
  from a join). A careless single-step agent scores ~7%, the oracle 100%, so the
  board can discriminate.
- `SQLEnvironment` accepts a multi-statement schema (several `CREATE TABLE`s plus
  inline `INSERT`s run as a script), and a scenario's `max_steps` now reaches the
  gym, so harder tasks get the room they need. The `sql_query` tool description
  lists the live schema.

## [0.7.2] - 2026-06-12

### Added

- `agentsynth pack new --from-schema db.sql` generates a starter pack from a
  `CREATE TABLE` — scenarios, checkers, and a working oracle, emitted together so
  the pack passes the gate out of the box. A fast way in from a database you
  already have; rename the scenarios and re-validate.

## [0.7.1] - 2026-06-12

### Added

- Bring-your-own-loop adapters: `to_openai_tools` exports a world's tools as
  OpenAI function-calling schemas and `action_from_openai_tool_call` turns the
  model's call back into a gym action, so OpenAI SDK / LangGraph / CrewAI loops
  drive `AgentGym.reset()`/`step()` directly. The final step's `info["outcome"]`
  carries the world-state verdict.
- `bench --compare a,b,c` runs model ids and/or policy refs side by side and
  prints one pass^k-aware table; `--json` captures every run, and `--submit`
  posts each under its own name.
- A green local bench suggests `--submit`, a clean validate points at the pack
  registry, and the CI action gained `submit`/`name` inputs so pipelines can
  feed the leaderboard on every run.

## [0.7.0] - 2026-06-12

### Added

- Pack tooling: `agentsynth pack new` scaffolds a working pack plus its oracle,
  `pack validate` is the merge gate (schema, the oracle solves everything, two
  runs agree, a do-nothing policy stays under half), and `pack teach` exports
  the oracle's episodes as verified gold trajectories for SFT seeding. Policies
  load from file paths (`oracle.py:fn`) as well as modules.
- Reliability scoring: `bench --trials K` runs a pack on K seeds and reports
  pass^K — a scenario counts only when every trial passes — with FLAKY called
  out per scenario; submissions send the reliability-adjusted numbers. FAIL
  lines now name the checkers that failed, and `--json` writes the full report
  for CI gates and analysis.
- The repo doubles as a GitHub Action: bench a pack in CI and fail the job when
  the pass rate drops under `min-pass-rate`.
- `agentsynth.demo`: the core_v1 expert (inspect, act, verify), the read-only
  baseline, and the lazy talker as importable policies, plus the demo pack with
  a local → hub → built-in fallback. The playground's new Agent runs tab plays
  them through real episodes.
- Trace donation: `redact_text` / `redact_trajectory` strip emails, API keys,
  bearer tokens, and phone-shaped numbers; `agentsynth import --redact` applies
  them before export. Plain numbers and dates survive.
- `scripts/hard_set.py` builds a training set from whatever the hub's breakdown
  says models fail most.

### Changed

- The core_v1 oracle works like a careful operator now — inspect the rows, make
  the change, read it back — which makes its episodes worth imitating.

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
