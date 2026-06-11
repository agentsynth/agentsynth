"""Reward functions for RL trainers, scored by checks instead of a judge.

`make_reward_fn(environment=env)` returns a callable in the shape TRL's
`reward_funcs` expects. Each completion is parsed as a tool call and scored on
four checks: it parses, the tool exists, required args are present, and the call
executes without an error. Weights renormalize over whichever checks are active;
pass `execute=False` (or no environment) to score format/schema only.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..environments import Environment
from ..schemas import ToolSpec
from ..utils import extract_json
from .episode import _ERROR_OBS_RE, EpisodeResult

_DEFAULT_WEIGHTS = {"parse": 0.25, "tool": 0.25, "args": 0.25, "execute": 0.25}


def _completion_text(completion: Any) -> str:
    """TRL passes either a string or a conversational message list per completion."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content") or "")
    return str(completion)


def score_tool_completion(
    completion: Any,
    tools: Sequence[ToolSpec],
    environment: Optional[Environment] = None,
    execute: bool = True,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Score one completion; returns `(reward, detail)` with per-check booleans."""
    active = dict(weights or _DEFAULT_WEIGHTS)
    if not (execute and environment is not None):
        active.pop("execute", None)
    total_weight = sum(active.values()) or 1.0

    checks = {key: False for key in active}
    detail: Dict[str, Any] = {"checks": checks}

    parsed = extract_json(_completion_text(completion))
    tool_name, args = None, None
    if isinstance(parsed, dict):
        # Accept both naming conventions: {"tool", "args"} and {"tool_name", "arguments"}.
        tool_name = parsed.get("tool") or parsed.get("tool_name")
        args = parsed.get("args") if "args" in parsed else parsed.get("arguments")
    if "parse" in checks:
        checks["parse"] = bool(tool_name) and isinstance(args, dict)

    spec = next((t for t in tools if t.name == tool_name), None)
    if "tool" in checks:
        checks["tool"] = spec is not None

    if "args" in checks and spec is not None and isinstance(args, dict):
        checks["args"] = all(r in args and args[r] is not None for r in spec.required_args())

    if "execute" in checks and spec is not None and isinstance(args, dict):
        try:
            observation = environment.execute(tool_name, args)  # type: ignore[union-attr, arg-type]
        except Exception as exc:
            observation = f"{type(exc).__name__}: {exc}"
        checks["execute"] = not _ERROR_OBS_RE.match(observation or "")
        detail["observation"] = (observation or "")[:200]

    detail["tool"] = tool_name
    reward = sum(active[key] for key, passed in checks.items() if passed) / total_weight
    return round(reward, 6), detail


def make_reward_fn(
    environment: Optional[Environment] = None,
    tools: Optional[Sequence[ToolSpec]] = None,
    execute: bool = True,
    weights: Optional[Dict[str, float]] = None,
) -> Callable[..., List[float]]:
    """A TRL-style reward function: `fn(prompts=..., completions=..., **kw) -> List[float]`."""
    if tools is None:
        if environment is None:
            raise ValueError("pass an environment, a tool list, or both")
        tools = environment.tools()
    tool_list = list(tools)

    def agentsynth_verified_reward(
        prompts: Optional[List[Any]] = None,
        completions: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> List[float]:
        rewards: List[float] = []
        for completion in completions or []:
            reward, _ = score_tool_completion(
                completion, tool_list, environment=environment, execute=execute, weights=weights
            )
            rewards.append(reward)
        return rewards

    return agentsynth_verified_reward


def episodes_to_grpo_jsonl(episodes: Sequence[EpisodeResult], path: str) -> str:
    """One `{"prompt", "completion", "reward"}` row per episode.

    The completion is the episode's first tool call (or the final answer when no
    tool was called). For the full multi-step data, use the trajectories on the
    episodes themselves."""
    if not episodes:
        raise ValueError("no episodes to export")
    with open(path, "w", encoding="utf-8") as fh:
        for episode in episodes:
            calls = episode.trajectory.tool_calls()
            if calls:
                completion = json.dumps(
                    {"tool": calls[0].tool_name, "args": calls[0].tool_args or {}}
                )
            else:
                completion = episode.trajectory.final_answer
            fh.write(
                json.dumps(
                    {
                        "prompt": episode.trajectory.query,
                        "completion": completion,
                        "reward": episode.total_reward,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


__all__ = ["score_tool_completion", "make_reward_fn", "episodes_to_grpo_jsonl"]
