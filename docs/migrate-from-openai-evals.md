# Migrating from OpenAI Evals

OpenAI's hosted Evals platform (the API under `/v1/evals`, and the dashboard at
platform.openai.com) [becomes read-only on October 31, 2026 and shuts down on
November 30, 2026](https://community.openai.com/t/deprecation-notice-evals-will-be-shut-down-on-november-30th-2026/1385537).
OpenAI's own migration guide points to Promptfoo. This page is for a narrower
case: your evals aren't really testing single-turn text output — they're testing
an **agent** that calls tools, mutates state, and runs for more than one step.
OpenAI Evals was never a great fit for that (it's a prompt → completion → grade
pipeline), so this deadline is a reasonable moment to move to something built for
agent evaluation specifically, rather than a straight like-for-like port.

If your evals are single-turn Q&A graded by text similarity against a reference
answer, Promptfoo or DeepEval are a smaller lift — say so honestly rather than
force a fit here.

**Note:** this is about the hosted Evals *platform* (`data_source_config` +
graders, run through the API/dashboard). If you're instead on the older
open-source [`openai/evals`](https://github.com/openai/evals) YAML+`oaieval`
framework, the concepts below still map, but nothing about that repository is
shutting down — there's no deadline forcing your hand.

## The one-sentence difference

OpenAI Evals grades **what the model said** against a reference answer or a
model judge. AgentSynth grades **what happened to the world** — a scenario ends
in pass or fail depending on the state of a database, a sandbox, or an API,
after the agent acted on it. A transcript that claims success is worth nothing
if the checker looks at the world and the world didn't change.

## Concept map

| OpenAI Evals | AgentSynth | Notes |
| --- | --- | --- |
| `data_source_config` (dataset items) | a [`Scenario`](https://github.com/agentsynth/agentsynth/blob/main/packs/README.md)'s `environment` | Not a static dataset row — a *live* environment (SQL rows, a sandbox, a REST fixture) the agent actually acts on. |
| `string_check` grader (`eq`/`like`/`ilike`) | `AnswerContains` checker | Same idea (does the output contain/equal a string) and the same weakness — `agentsynth pack audit` flags these as text-graded, gameable by a policy that echoes the prompt. Prefer a state check below when the task has any state to check. |
| `text_similarity` grader (cosine/BLEU/ROUGE/fuzzy) | *(no equivalent, by design)* | Similarity-to-reference-text scoring is exactly the kind of soft signal outcome verification exists to avoid. If the task has a real answer, check the state it produced; if it's genuinely open-ended prose, that's a judged dimension, not a benchmark score — see below. |
| `score_model` / `label_model` (LLM-as-judge) | the trajectory judge (`TrajectoryEvaluator`) | Runs and reports, but carries **zero weight** in `bench` pass/fail — it's a quality signal for dataset curation (SFT filtering), not a source of truth for whether a benchmark scenario passed. `agentsynth pack audit` exists because judged/text-graded scoring is gameable; outcome checks aren't. |
| `python` grader (`grade(sample, item) -> float`) | `CodeCheck` checker | Closest analog, with one difference: OpenAI's python grader scores the sample directly; `CodeCheck` collects the code the agent *wrote during the episode*, then runs a **hidden** test against it in a sandbox after the fact — the agent never sees the test it's graded on. |
| `multi` grader (combine several) | a scenario's `checkers` list | A scenario passes only when *every* checker passes — the same AND-combination. |
| running an eval against a model | `agentsynth bench --pack X --model Y` | |
| a single accuracy number | `bench --trials k` → pass¹..passᵏ | OpenAI Evals reports one run; `--trials` reports the *reliability* — how often the same scenario passes across k attempts, with confidence intervals, which a single run can't tell you. |

## Worked example: a `string_check` eval

An OpenAI eval that checks a support-bot reply contains the right order status:

```yaml
data_source_config:
  type: custom
  item_schema:
    order_id: { type: integer }
    expected_status: { type: string }
testing_criteria:
  - type: string_check
    input: "{{sample.output_text}}"
    operation: like
    reference: "{{item.expected_status}}"
```

This checks the model's *words*. Nothing stops a model from saying "refunded"
without anything actually being refunded — which is precisely the failure mode
[recent audits found in production agent benchmarks](https://arxiv.org/abs/2605.12673).
The AgentSynth version checks the *database*:

```yaml
- id: refund-order-7
  task: Refund order 7 in the orders database, then confirm what you did.
  environment:
    type: sql
    schema: CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT)
    table: orders
    rows: [[7, "paid"]]
  checkers:
    - kind: sql
      query: SELECT status FROM orders WHERE id=7
      equals: [["refunded"]]
    - kind: answer
      any_of: ["refund"]
```

The `sql` checker is the one that matters; the `answer` checker is kept
deliberately weak (a text mention), matching how OpenAI Evals graded it —
migrate the *check*, not just the syntax.

## Worked example: a `python` grader

```python
# OpenAI python grader
def grade(sample, item):
    return 1.0 if "def is_prime" in sample["output_text"] else 0.0
```

This checks that the code *exists*, not that it *works*. `CodeCheck` runs it:

```yaml
- id: is-prime
  task: Write a function is_prime(n) that returns True iff n is prime.
  environment: { type: python }
  checkers:
    - kind: code
      test: |
        assert is_prime(7) is True
        assert is_prime(8) is False
        assert is_prime(1) is False
```

The hidden test runs against whatever the agent actually wrote, in the sandbox,
after the episode ends — a function that merely contains the string `is_prime`
scores zero.

## Running it

```bash
pip install agentsynth-ai
agentsynth pack validate my_pack.yaml          # oracle solves it, do-nothing doesn't
agentsynth pack audit my_pack.yaml             # how gameable are the checkers?
agentsynth bench --pack my_pack.yaml --model gpt-4o-mini --trials 4
```

`pack validate` is the equivalent of "does my eval config even work" — a
reference solution must pass every scenario, and a policy that does nothing
must pass none, gating out the two cheapest ways an eval can be broken
(unsolvable tasks, vacuous checks) before you ever run a model against it.

See the [README](https://github.com/agentsynth/agentsynth#scenarios--verify-the-outcome-not-the-vibes)
for the full scenario schema and [`packs/README.md`](https://github.com/agentsynth/agentsynth/blob/main/packs/README.md)
for the gates a pack has to pass to ship.
