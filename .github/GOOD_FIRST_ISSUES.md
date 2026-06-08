# Good first issues

A starter backlog for new contributors. Each is scoped to a few files with clear
acceptance criteria. After the repo is public, turn these into real GitHub issues
(there's a `gh` snippet at the bottom) and label them `good first issue`.

Difficulty: 🟢 easy · 🟡 medium · 🔴 involved

---

### 🟢 Add a tool catalog for a new domain
Add a ready-made catalog (finance, devops, or support) alongside `DEFAULT_TOOL_CATALOG`.
- Files: `agentsynth/utils.py`, a test in `tests/`
- Done when: the catalog parses via `parse_tool_catalog`, has 4–6 well-formed tools, and a test covers it.

### 🟢 Add seed tasks for a new domain
Extend the taxonomy with 3–5 realistic tasks in a domain not well covered yet.
- Files: `agentsynth/tasks/taxonomy.py`, `tests/test_tasks.py`
- Done when: `domains()` includes the new domain and `sample_tasks` can return them.

### 🟢 Add a `--workers` flag to `agentsynth generate`
Wire a concurrency flag through to the recipe runner's `max_workers`.
- Files: `agentsynth/cli.py`
- Done when: `agentsynth generate ... --workers 4` runs and the output matches `--workers 1`.

### 🟢 Add Parquet to the app's export dropdown
The exporter already supports parquet; surface it in the Gradio UI.
- Files: `app.py`
- Done when: selecting `parquet` produces a downloadable file (pandas/pyarrow installed).

### 🟢 Add more sample queries / a second table to `SQLEnvironment`
Richer fixtures make grounded SQL trajectories more varied.
- Files: `agentsynth/environments/sql.py`, `tests/test_environments.py`
- Done when: the new fixtures load and `sample_args` can produce queries against them.

### 🟡 Carry the seed task's domain onto the trajectory
When a run samples from the taxonomy, set `Trajectory.domain` from the `SeedTask`.
- Files: `agentsynth/pipelines/runner.py`, `tests/test_pipelines.py`
- Done when: trajectories from a taxonomy run have their `domain` populated.

### 🟡 Add a filesystem environment
A `read_file` backend over a small set of in-memory fixture files.
- Files: new `agentsynth/environments/filesystem.py`, exports, a test
- Done when: `read_file` returns real fixture contents and routes through `CompositeEnvironment`.

### 🟡 Add a `plot_domain_distribution` figure
A Plotly bar of trajectories per domain, wired into the Metrics tab.
- Files: `agentsynth/metrics.py`, `app.py`
- Done when: the figure renders and handles empty input like the others.

### 🟡 Add an `agentsynth validate <file.jsonl>` command
Validate a JSONL dataset against the schema and report problems.
- Files: `agentsynth/cli.py`
- Done when: a valid file passes and a malformed line is reported with its line number.

### 🔴 Harden the Python sandbox
Add resource limits (CPU/memory via `resource` on POSIX) and an optional package allowlist.
- Files: `agentsynth/environments/python_sandbox.py`, tests, a note in `SECURITY.md`
- Done when: a runaway snippet is killed by limits and the docs reflect the new guarantees.

### 🟢 Write a 5-minute tutorial notebook
"Generate → evaluate → export" end to end, offline.
- Files: new `examples/quickstart.ipynb`
- Done when: the notebook runs top to bottom with no API key.

---

## Creating these as issues (after the repo is public)

```bash
gh label create "good first issue" --color 7057ff --force
# then, per item:
gh issue create --title "Add a tool catalog for a new domain" \
  --label "good first issue,enhancement" \
  --body "See .github/GOOD_FIRST_ISSUES.md for context and acceptance criteria."
```
