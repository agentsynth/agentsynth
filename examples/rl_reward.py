"""RL episodes and verified rewards, end to end and fully offline.

    python examples/rl_reward.py

Three things in one file: an AgentGym episode where every tool call runs against a
real SQLite database and the terminal reward comes from verification + the judge; a
TRL-style reward function scoring model completions by parse/tool/args/execution; and
the episode export for offline RL. Plug the reward function straight into TRL:

    GRPOTrainer(model, reward_funcs=make_reward_fn(environment=env), ...)
"""

import json
import os

os.environ.setdefault("AGENTSYNTH_FORCE_MOCK", "1")

from agentsynth import AgentGym, make_reward_fn
from agentsynth.environments import SQLEnvironment
from agentsynth.rl import episodes_to_grpo_jsonl


def main() -> None:
    env = SQLEnvironment()
    gym = AgentGym(env, task="Which region has the highest total revenue?")

    # 1. A scripted episode — swap the script for your policy / model.
    script = iter(
        [
            {
                "tool_name": "sql_query",
                "arguments": {
                    "query": "SELECT region, SUM(revenue) AS total "
                    "FROM sales GROUP BY region ORDER BY total DESC"
                },
            },
            {"answer": "APAC has the highest total revenue."},
        ]
    )
    episode = gym.rollout(lambda observation, g: next(script))
    print(f"episode reward: {episode.total_reward:.3f}  (per step: {episode.rewards})")
    print(
        "verified:",
        episode.info["verification"]["verified"],
        "| judge:",
        episode.info["judge"]["overall"],
    )

    # 2. The same verification stack as a TRL-compatible reward function.
    reward_fn = make_reward_fn(environment=env)
    completions = [
        json.dumps({"tool": "sql_query", "args": {"query": "SELECT COUNT(*) FROM sales"}}),
        json.dumps({"tool": "sql_query", "args": {}}),  # missing required arg
        "not a tool call at all",
    ]
    print("\nreward_fn scores:", reward_fn(prompts=["p"] * 3, completions=completions))

    # 3. Export episodes for offline methods (rejection sampling, best-of-n).
    path = episodes_to_grpo_jsonl([episode], "episodes.jsonl")
    print(f"\nwrote {path}:", open(path, encoding="utf-8").readline().strip()[:120], "…")
    os.remove(path)
    gym.close()


if __name__ == "__main__":
    main()
