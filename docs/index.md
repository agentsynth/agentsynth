# AgentSynth

Synthetic, **verified** multi-step agent trajectories for fine-tuning agentic LLMs —
tool-use, grounded code execution, and multi-agent collaboration — with a built-in
LLM-as-Judge evaluation loop. It runs offline for free and scales up with any model.

The thesis in one line: the value isn't generating agent trajectories, it's
generating ones you can **trust**. So verification is core, not a side feature.

## Install

```bash
pip install agentsynth-ai                # core: generate + evaluate + export
pip install "agentsynth-ai[app]"         # + the Gradio UI
pip install "agentsynth-ai[all]"         # everything
```

## 60-second tour

```python
from agentsynth import AgentTrajectoryGenerator, TrajectoryEvaluator, verify_trajectory

gen = AgentTrajectoryGenerator()                      # offline mock by default
traj = gen.generate("What's the weather in Paris, and 18% tip on $54?")

result = TrajectoryEvaluator().evaluate(traj)         # 6-dimension rubric
print(result.overall, result.passed)

print(verify_trajectory(traj).verified)               # re-checks tool args, execution, safety
```

## Get on the leaderboard

`core_v2` is the flagship pack — 14 tiered, outcome-checked business tasks over a
writable SQL world, up to multi-table consistency. A run passes only when the world
ends up in the goal state. The pack downloads itself, so this works anywhere:

```bash
agentsynth bench --pack core_v2 --model <your-model> --submit
```

The live board is at [agentsynth.tech](https://agentsynth.tech/leaderboard);
`--policy mypkg.module:fn` benches your own agent loop instead of a LiteLLM model,
and `--trials k` scores `pass^k` reliability. (`core_v1` is a gentler 10-task starter.)

## Where to go next

- **[Vision](VISION.md)** — the problem, the bet, and the principles.
- **[Architecture](ARCHITECTURE.md)** — how the pieces fit together.
- **[Fine-tune & benchmark](BENCHMARK.md)** — turn the data into a model and prove it helps.
- **[API reference](reference.md)** — the public surface.

The live playground is at [app.agentsynth.tech](https://app.agentsynth.tech) (mirrored on
[Hugging Face Spaces](https://huggingface.co/spaces/agentsynth/agentsynth)), and the code is
on [GitHub](https://github.com/agentsynth/agentsynth).
