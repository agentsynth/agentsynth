# Vision

## The problem

Agentic models are trained on agent trajectories: a goal, a sequence of tool calls
and observations, some reasoning, and an outcome. That data is scarce. Real
production traces are private, messy, and often can't be used for legal reasons, and
hand-written examples don't scale. So teams either don't have the data they need or
pay a lot for a black-box vendor to make it.

## What we're building

AgentSynth is an open engine for generating and verifying agent trajectories. You
give it a task and a set of tools; it produces multi-step trajectories, scores them,
and hands you a training-ready dataset. It runs offline for free, scales up with a
real LLM when you want richer data, and — this is the part that matters — it's built
around verification, not just generation.

The thesis in one line: **the value isn't generating trajectories, it's generating
ones you can trust.** Anyone can prompt a model into producing plausible-looking
tool-use transcripts. The hard, useful work is knowing which of them are actually
correct, grounded, and safe to train on. That's why the eval loop is core, not a
side feature, and why execution-based verification is the next big push.

## The flywheel

```
generate -> verify -> train -> evaluate -> mine failures -> generate ...
```

Generate trajectories, verify them, train on the ones that pass, evaluate the model
on a real benchmark, find where it still fails, and aim the next generation run at
those gaps. Each turn of the loop should produce a better dataset and a better model.

## Where we differ

- **distilabel** and similar tools do general synthetic data. AgentSynth is specific
  to agent trajectories and is built around execution-based verification.
- **Multi-agent frameworks** (Camel and friends) focus on running agents. We focus on
  turning agent runs into trustworthy, trainer-ready data.
- **Closed data vendors** sell datasets. We're open: the pipeline, the rubric, and the
  released datasets are all inspectable and reproducible.

## Principles

- **Open by default.** Pipeline, rubric, and flagship datasets are public and reproducible.
- **Verification over volume.** A smaller verified set beats a large unchecked one.
- **Offline-first.** It runs for free with no keys; a real LLM is an upgrade, never a requirement.
- **Trainer-native.** Output drops into TRL / Unsloth / Axolotl without a custom parser.
- **Reproducible.** Same inputs, same data — so results can be checked.

## How we'll know it's working

Trajectories generated and verified. Public datasets released. And the one that
counts most: models fine-tuned on AgentSynth data that move the needle on public
agent benchmarks, with the run published so anyone can reproduce it.
