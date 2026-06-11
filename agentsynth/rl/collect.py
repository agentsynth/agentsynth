"""Parallel episode collection and exports for offline training.

A gym holds one live episode, so collection runs one gym per worker via a
factory. `episodes_to_rft_jsonl` keeps the top reward quantile and writes
SFT-ready `messages` records (rejection sampling).

    episodes = collect_episodes(lambda: AgentGym.from_scenario(scenario),
                                policy, episodes=64, max_workers=8)
    episodes_to_rft_jsonl(episodes, "rft.jsonl", top_quantile=0.25)
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Sequence

from .episode import AgentGym, EpisodeResult

PolicyFn = Callable[[str, AgentGym], Any]


def collect_episodes(
    gym_factory: Callable[[], AgentGym],
    policy: PolicyFn,
    episodes: int = 16,
    max_workers: int = 4,
    seed: int = 7,
) -> List[EpisodeResult]:
    """Run `episodes` rollouts across a pool of gyms built from `gym_factory`.

    Episode i uses seed + i, so ids stay unique and a re-run reproduces the same
    episodes in mock mode. Results come back in episode order.
    """
    if episodes <= 0:
        return []

    def run_one(index: int) -> EpisodeResult:
        gym = gym_factory()
        try:
            return gym.rollout(policy, seed=seed + index)
        finally:
            gym.close()

    workers = max(1, min(max_workers, episodes))
    if workers == 1:
        return [run_one(i) for i in range(episodes)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_one, range(episodes)))


def episodes_to_rft_jsonl(
    episodes: Sequence[EpisodeResult],
    path: str,
    top_quantile: float = 0.5,
) -> str:
    """Keep the top reward quantile and write SFT-ready `messages` records.

    Ties at the cutoff are kept, so a batch where every episode scored the same
    exports whole."""
    if not episodes:
        raise ValueError("no episodes to export")
    if not 0.0 < top_quantile <= 1.0:
        raise ValueError("top_quantile must be in (0, 1]")

    rewards = sorted((e.total_reward for e in episodes), reverse=True)
    cutoff_index = max(0, min(len(rewards) - 1, int(len(rewards) * top_quantile) - 1))
    cutoff = rewards[cutoff_index]

    kept = [e for e in episodes if e.total_reward >= cutoff]
    with open(path, "w", encoding="utf-8") as fh:
        for episode in kept:
            record = {
                "messages": episode.trajectory.to_messages(),
                "reward": episode.total_reward,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


__all__ = ["collect_episodes", "episodes_to_rft_jsonl"]
