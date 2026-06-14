# Contributing to AgentSynth

Thanks for taking the time. The single most valuable thing you can add is a
**scenario pack** (see below); bug reports, new environments, export formats, and
docs fixes are all welcome too, and small PRs get reviewed fastest.

If you're planning something larger than a bug fix, open an issue or a Discussion
first so we can agree on the shape before you write code.

## Getting set up

The fastest path is `make`:

```bash
git clone https://github.com/agentsynth/agentsynth
cd agentsynth
make setup        # create a .venv, install the dev toolchain + pre-commit
make test         # run the suite, fully offline (mock mode)
make lint         # ruff + mypy, exactly like CI
```

`make setup-all` also installs the app / MCP / browser extras (and downloads a headless
Chromium); `make help` lists every target. Or open the repo in a **GitHub Codespace** —
the devcontainer installs everything for you.

Prefer to do it by hand?

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10+ for the app, 3.9+ for the core
pip install -e ".[app,dev]"
pre-commit install
pytest                       # tests run fully offline (mock mode)
ruff check . && ruff format --check .
mypy
```

`pre-commit` runs ruff and a few hygiene hooks on every commit. To run it over the whole
tree at once: `pre-commit run --all-files`.

## Ground rules for code

- **The core (`agentsynth/` minus `app.py`) stays Python 3.9-compatible.** Use
  `typing.Optional`/`List`/`Dict`, not `X | None` or `list[...]` in evaluated
  positions. CI runs the test matrix on 3.10–3.12, but the type target is 3.9.
- **Everything works offline.** Generation and evaluation must have a deterministic
  mock path that needs no API key. The LLM backend is optional and always falls
  back to mock. `AGENTSYNTH_FORCE_MOCK=1` forces offline.
- **Determinism in mock paths.** Seed any randomness through `agentsynth.utils.stable_seed`
  so tests stay stable.
- **Keep optional deps optional.** `plotly`, `pandas`, `datasets`, `gradio`, and
  `litellm` are imported lazily inside the functions that use them, never at module
  top (except `app.py`).
- Add or update a test for anything you change. The suite is in `tests/`.

## Contribute a scenario pack

The highest-leverage contribution is a pack for a domain you know: real tasks
over a seeded world, checkers on the end state, an oracle proving it's solvable.
Scaffold one, make it yours, and run the gate locally:

```bash
agentsynth pack new my_domain_v1 --dir packs
agentsynth pack validate packs/my_domain_v1.yaml
```

CI runs the same validation on every PR. [packs/README.md](packs/README.md) has
the full checklist and the registry table to add yourself to. Once merged, the
pack gets its own live leaderboard.

## Pull requests

1. Branch off `main`, make your change, add tests.
2. Make sure `pytest`, `ruff`, and `mypy` are green.
3. Open the PR against `main` and fill in the template. Link the issue it closes.

We keep history linear. Squash-and-merge is the default.

## Releases (maintainers)

Releases are cut from tags. Bump the version in `pyproject.toml` and `agentsynth/__init__.py`,
update `CHANGELOG.md`, then:

```bash
git tag v0.1.0 && git push --tags
```

The `release` workflow builds the wheel and publishes to PyPI via Trusted
Publishing (no token in the repo — the publisher is configured on PyPI itself).
