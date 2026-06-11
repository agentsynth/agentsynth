"""Bridge an AgentGym onto the OpenEnv standard (meta-pytorch).

OpenEnv standardizes the environment interface — `reset()` / `step()` / `state`
over a client/server protocol — and leaves rewards to the environment library.
The bridged gym serves real tool execution per step and the verification + judge
reward at the end of the episode.

    from agentsynth.rl import AgentGym, to_openenv

    env = to_openenv(AgentGym(SQLEnvironment(), task="revenue by region"))
    obs = env.reset()
    obs = env.step(CallToolAction(tool_name="sql_query", arguments={"query": "..."}))
    obs = env.step(CallToolAction(tool_name="final_answer",
                                  arguments={"answer": "EMEA leads"}))
    print(obs.reward, obs.done)

The agent ends an episode by calling the reserved `final_answer` tool; its
`answer` argument is coerced to a string. `state` is a property here because the
OpenEnv spec requires one. Requires `pip install "agentsynth-ai[rl]"`
(Python 3.10+); the core AgentGym works without it.
"""

from __future__ import annotations

from typing import Any, Optional

from .episode import AgentGym

FINAL_ANSWER_TOOL = "final_answer"


def to_openenv(gym: AgentGym) -> Any:
    """Wrap an AgentGym in an `openenv.core.Environment` (lazy import)."""
    try:
        from openenv.core import CallToolObservation, Environment, State
    except ImportError as exc:
        raise ImportError(
            'the OpenEnv bridge needs openenv-core: pip install "agentsynth-ai[rl]"'
        ) from exc

    class AgentSynthOpenEnv(Environment):
        """OpenEnv wrapper: CallToolAction in, CallToolObservation (+ reward) out."""

        def __init__(self, inner: AgentGym) -> None:
            self.gym = inner
            self._episode_id: Optional[str] = None

        def reset(
            self,
            seed: Optional[int] = None,
            episode_id: Optional[str] = None,
            task: Optional[str] = None,
            **kwargs: Any,
        ) -> Any:
            prompt = self.gym.reset(task=task, seed=seed)
            self._episode_id = episode_id
            return CallToolObservation(
                tool_name="reset",
                result={
                    "task": prompt,
                    "tools": self.gym.environment.tool_names() + [FINAL_ANSWER_TOOL],
                },
                done=False,
                reward=None,
            )

        def step(self, action: Any, timeout_s: Optional[float] = None, **kwargs: Any) -> Any:
            if action.tool_name == FINAL_ANSWER_TOOL:
                outcome = self.gym.step({"answer": str((action.arguments or {}).get("answer", ""))})
            else:
                outcome = self.gym.step(
                    {"tool_name": action.tool_name, "arguments": action.arguments or {}}
                )
            info = {k: v for k, v in outcome.info.items() if k != "trajectory"}
            return CallToolObservation(
                tool_name=action.tool_name,
                result=outcome.observation,
                reward=outcome.reward,
                done=outcome.done,
                metadata=info,
            )

        @property
        def state(self) -> Any:
            return State(episode_id=self._episode_id, step_count=self.gym.step_count)

        def close(self) -> None:
            self.gym.close()

    return AgentSynthOpenEnv(gym)


__all__ = ["to_openenv", "FINAL_ANSWER_TOOL"]
