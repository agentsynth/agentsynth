"""Execution environments.

An environment owns a few tools and actually runs them, so a trajectory's
observations are real output instead of templated text. Environments are
optional — without one the generator falls back to templated mock observations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List

from ..schemas import ToolSpec


class Environment(ABC):
    name: str = "environment"

    @abstractmethod
    def tools(self) -> List[ToolSpec]:
        """The tools this environment can run."""

    @abstractmethod
    def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Run a tool call and return the observation text.

        Failures should come back as observation strings ("SQLError: ...") so the
        agent can read them; raising for an unknown tool is also fine — callers in
        the RL and generation layers tolerate both.
        """

    def sample_args(self, tool_name: str, query: str, seed: int) -> Dict[str, Any]:
        """A valid example call for `tool_name`, so generated calls actually run.

        Returns an empty dict by default; callers then synthesize their own args.
        """
        return {}

    def tool_names(self) -> List[str]:
        return [t.name for t in self.tools()]

    def reset(self) -> None:
        """Restore initial state. No-op for stateless environments."""

    def close(self) -> None:
        """Release any resources."""


class CompositeEnvironment(Environment):
    """Several environments behind one interface, routed by tool name."""

    name = "composite"

    def __init__(self, environments: Iterable[Environment]) -> None:
        self._envs: List[Environment] = list(environments)
        self._route: Dict[str, Environment] = {}
        for env in self._envs:
            for tool_name in env.tool_names():
                self._route.setdefault(tool_name, env)

    def tools(self) -> List[ToolSpec]:
        out: List[ToolSpec] = []
        seen = set()
        for env in self._envs:
            for tool in env.tools():
                if tool.name not in seen:
                    seen.add(tool.name)
                    out.append(tool)
        return out

    def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        env = self._route.get(tool_name)
        if env is None:
            raise KeyError(f"no environment handles tool '{tool_name}'")
        return env.execute(tool_name, args)

    def sample_args(self, tool_name: str, query: str, seed: int) -> Dict[str, Any]:
        env = self._route.get(tool_name)
        return env.sample_args(tool_name, query, seed) if env else {}

    def reset(self) -> None:
        for env in self._envs:
            env.reset()

    def close(self) -> None:
        for env in self._envs:
            env.close()


__all__ = ["Environment", "CompositeEnvironment"]
