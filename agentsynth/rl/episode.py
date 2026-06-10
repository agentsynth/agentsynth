"""Gym-style episodes over any AgentSynth environment, with verified rewards.

An agent (or a policy under RL training) interacts step by step: each action is a
tool call that really executes, and when the episode ends the whole trajectory goes
through the standard verification stack plus the judge — so the terminal reward is
grounded in what actually happened, not in how plausible the text looks.

    gym = AgentGym(SQLEnvironment(), task="total revenue by region")
    obs = gym.reset()
    out = gym.step({"tool_name": "sql_query", "arguments": {"query": "SELECT ..."}})
    out = gym.step({"answer": "EMEA leads with ..."})   # ends + scores the episode

The same episodes export to JSONL for offline methods (rejection sampling, GRPO-style
filtering), and `agentsynth.rl.to_openenv` bridges a gym onto the OpenEnv standard.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ..environments import Environment
from ..evaluator import TrajectoryEvaluator
from ..schemas import Trajectory, TrajectoryStep
from ..utils import extract_json, stable_seed

# Matches our environments' error format ("SQLError: ...", "KeyError: 'x'") without
# tripping on data that merely starts with the word ("Error rates dropped 5%").
_ERROR_OBS_RE = re.compile(r"^[A-Za-z_]*(Error|Exception):")

ActionLike = Union["ToolAction", Dict[str, Any], str]


class ToolAction(BaseModel):
    """One agent action: call a tool, or give the final answer to end the episode."""

    tool_name: Optional[str] = None
    arguments: Dict[str, Any] = Field(default_factory=dict)
    answer: Optional[str] = None


class StepOutcome(BaseModel):
    observation: str
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class EpisodeResult(BaseModel):
    trajectory: Trajectory
    total_reward: float
    rewards: List[float] = Field(default_factory=list)
    info: Dict[str, Any] = Field(default_factory=dict)


def _coerce_action(action: ActionLike) -> ToolAction:
    """Accept ToolAction, a dict, or raw model text (JSON tool call / final answer)."""
    if isinstance(action, ToolAction):
        return action
    if isinstance(action, str):
        parsed = extract_json(action)
        if isinstance(parsed, dict) and ("tool" in parsed or "tool_name" in parsed):
            action = parsed
        else:
            return ToolAction(answer=action)
    if isinstance(action, dict):
        args = action.get("arguments") or action.get("args") or {}
        return ToolAction(
            tool_name=action.get("tool_name") or action.get("tool"),
            arguments=args if isinstance(args, dict) else {},
            answer=action.get("answer"),
        )
    raise TypeError(f"cannot interpret action of type {type(action).__name__}")


class AgentGym:
    """Step-by-step episodes whose terminal reward comes from verification + the judge.

    Per-step shaping is deliberately small and transparent: an action that names no
    real tool is penalized by `invalid_action_penalty`, an observation that comes back
    as an error is penalized by `error_penalty` (both applied as negative rewards),
    and everything else is 0 until the episode ends. The terminal reward is
    `verify_weight * verification.score + judge_weight * judge_overall`, both in
    [0, 1] — and with `require_grounding` (the default) the verification credit is
    only paid when the trajectory actually executed something, so a policy can't farm
    reward by skipping the tools and emitting a plausible answer.

    An action carrying `answer` always ends the episode, even if it also names a tool.
    `state()` is a method here; the OpenEnv bridge exposes it as a property, as that
    spec requires. Not thread-shareable (one live episode at a time) — use one gym
    per worker.
    """

    def __init__(
        self,
        environment: Environment,
        task: Optional[str] = None,
        judge: Optional[TrajectoryEvaluator] = None,
        verifiers: Optional[List[Any]] = None,
        max_steps: int = 8,
        verify_weight: float = 0.5,
        judge_weight: float = 0.5,
        invalid_action_penalty: float = 0.05,
        error_penalty: float = 0.02,
        require_grounding: bool = True,
        max_observation_chars: int = 4000,
        seed: int = 7,
    ) -> None:
        self.environment = environment
        self.task = task or ""
        self.judge = judge or TrajectoryEvaluator(use_mock="auto")
        self.verifiers = verifiers
        self.max_steps = max_steps
        self.verify_weight = verify_weight
        self.judge_weight = judge_weight
        self.invalid_action_penalty = invalid_action_penalty
        self.error_penalty = error_penalty
        self.require_grounding = require_grounding
        self.max_observation_chars = max_observation_chars
        self.seed = seed
        self._steps: List[TrajectoryStep] = []
        self._rewards: List[float] = []
        self._actions_taken = 0
        self._episode_index = 0
        self._done = True  # call reset() to start an episode

    # -- episode lifecycle -----------------------------------------------------

    def reset(self, task: Optional[str] = None, seed: Optional[int] = None) -> str:
        if task is not None:
            self.task = task
        if seed is not None:
            self.seed = seed
        if not self.task:
            raise ValueError("no task: pass task= to AgentGym(...) or reset(task=...)")
        self.environment.reset()
        self._steps = []
        self._rewards = []
        self._actions_taken = 0
        self._episode_index += 1
        self._done = False
        return self.task

    @property
    def done(self) -> bool:
        return self._done

    @property
    def step_count(self) -> int:
        return self._actions_taken

    def state(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "steps_taken": self._actions_taken,
            "tools": self.environment.tool_names(),
            "done": self._done,
        }

    def step(self, action: ActionLike) -> StepOutcome:
        if self._done:
            raise RuntimeError("episode is over — call reset() first")
        act = _coerce_action(action)
        self._actions_taken += 1

        if act.answer is not None:
            return self._finish(act.answer, truncated=False)

        outcome = self._call_tool(act)
        if not outcome.done and self._actions_taken >= self.max_steps:
            step_reward = self._rewards.pop()  # fold this step's reward into the terminal one
            forced = self._finish("", truncated=True)
            forced.reward += step_reward
            self._rewards[-1] = forced.reward
            forced.info["last_observation"] = outcome.observation
            return forced
        return outcome

    def _call_tool(self, act: ToolAction) -> StepOutcome:
        if not act.tool_name or act.tool_name not in self.environment.tool_names():
            reward = -self.invalid_action_penalty
            self._rewards.append(reward)
            known = ", ".join(self.environment.tool_names())
            return StepOutcome(
                observation=f"InvalidAction: unknown tool {act.tool_name!r}; available: {known}",
                reward=reward,
                done=False,
                info={"invalid_action": True},
            )

        try:
            observation = self.environment.execute(act.tool_name, act.arguments)
        except Exception as exc:
            observation = f"{type(exc).__name__}: {exc}"
        if len(observation or "") > self.max_observation_chars:
            observation = observation[: self.max_observation_chars - 1].rstrip() + "…"
        self._steps.append(
            TrajectoryStep(step_type="tool_call", tool_name=act.tool_name, tool_args=act.arguments)
        )
        self._steps.append(TrajectoryStep(step_type="observation", observation=observation))

        reward = -self.error_penalty if _ERROR_OBS_RE.match(observation or "") else 0.0
        self._rewards.append(reward)
        return StepOutcome(observation=observation, reward=reward, done=False, info={})

    def _finish(self, answer: str, truncated: bool) -> StepOutcome:
        from ..verification import verify_trajectory

        if answer:
            self._steps.append(TrajectoryStep(step_type="final_answer", content=answer))
        trajectory = self._build_trajectory(answer, truncated)

        verification = verify_trajectory(trajectory, self.verifiers)
        trajectory.verification = verification.model_dump()
        judged = self.judge.evaluate(trajectory)

        # Verification credit must be earned: an episode that never executed anything
        # has nothing verified, so it gets no credit for "passing" vacuously.
        grounded = bool(trajectory.tool_calls()) or any(
            s.step_type == "code_execution" for s in trajectory.steps
        )
        verify_credit = verification.score if (grounded or not self.require_grounding) else 0.0
        reward = self.verify_weight * verify_credit + self.judge_weight * judged.overall
        self._rewards.append(reward)
        self._done = True
        return StepOutcome(
            observation=f"Episode finished: {answer or '(no answer)'}",
            reward=reward,
            done=True,
            info={
                "truncated": truncated,
                "trajectory_id": trajectory.id,
                "verification": verification.model_dump(),
                "judge": judged.flat(),
                "trajectory": trajectory,
            },
        )

    def _build_trajectory(self, answer: str, truncated: bool) -> Trajectory:
        # Unique per episode (the index) but reproducible for the same seed + task +
        # rollout order, so repeated collection never silently collides ids.
        key = f"{self.task}#{self._episode_index}"
        traj_id = format(stable_seed(self.seed, key) & 0xFFFFFFFFFFFF, "012x")
        return Trajectory(
            id=traj_id,
            query=self.task,
            mode="single_agent",
            tools=self.environment.tools(),
            steps=list(self._steps),
            final_answer=answer,
            success=bool(answer) and not truncated,
            generator_model="rl_episode",
            metadata={"source": "rl_episode", "truncated": truncated},
        )

    # -- conveniences ------------------------------------------------------------

    def rollout(
        self,
        policy: Callable[[str, "AgentGym"], ActionLike],
        task: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> EpisodeResult:
        """Run one full episode driven by `policy(observation, gym) -> action`."""
        observation = self.reset(task=task, seed=seed)
        outcome = StepOutcome(observation=observation, reward=0.0, done=False)
        while not outcome.done:
            outcome = self.step(policy(outcome.observation, self))
        info = dict(outcome.info)
        trajectory = info.pop("trajectory")
        return EpisodeResult(
            trajectory=trajectory,
            total_reward=round(sum(self._rewards), 6),
            rewards=list(self._rewards),
            info=info,
        )

    def close(self) -> None:
        self.environment.close()


__all__ = ["ToolAction", "StepOutcome", "EpisodeResult", "AgentGym"]
