# Why AgentSynth exists

Agents are only as good as the trajectories they learn from. And good agent data —
multi-step tool use, real execution, honest outcomes — is the scarcest thing in the
stack. Production traces are private and messy. Hand-written examples don't scale.
So most teams building agents are flying without the data they need, and the ones who
have it bought it from a black box.

We think that's backwards. The data engine for agents should be open.

**Our one belief:** the hard part isn't generating agent trajectories — it's
generating ones you can trust. Anyone can prompt a model into transcripts that *look*
like good tool use. Knowing which are actually correct, grounded, and safe to train on
is the real work. So verification isn't a feature we bolted on. It's the point.

That's why AgentSynth runs tools for real instead of imagining their output. Why the
eval loop sits at the center, not the edge. Why every dataset we release ships with the
recipe to reproduce it. If you can't check it, we don't want to train on it — and
neither should you.

**What we're building:** an open engine that generates, verifies, and scores agent
trajectories, plus the open datasets to prove it works. It runs free and offline out of
the box. It scales up with any model you like. And it's built so the output drops
straight into the trainers people already use.

**What we believe in:**

- Open by default — pipeline, rubric, and flagship datasets, all inspectable.
- Verification over volume — a smaller trusted set beats a big unchecked one.
- Reproducible — same inputs, same data, so claims can be checked.
- For everyone, not just the labs — the thousands of teams building agents without a
  data team.

If that resonates, there's a place for you here. Pick up a
[good first issue](https://github.com/agentsynth/agentsynth/contribute), bring a tool
catalog or an execution environment, file a sharp bug, or help us prove on a public
benchmark that synthetic agent data can make a model genuinely better.

The agents are coming either way. Let's make sure the data they learn from is worth
trusting.

— the AgentSynth project
