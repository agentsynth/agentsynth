"""A small plugin registry so the community can extend AgentSynth without forking it.

Register a custom environment under a name and any scenario can use it by `type`:

    from agentsynth.plugins import register_environment

    register_environment("my_api", lambda **cfg: MyApiEnvironment(**cfg))
    Scenario(id="...", environment={"type": "my_api", "base_url": "..."}, checkers=[...])

Packages can also ship environments via an entry point, so installing them is enough:

    [project.entry-points."agentsynth.environments"]
    my_api = "my_pkg.envs:MyApiEnvironment"

The built-in `sql` / `python` / `rest` types are resolved directly; anything else falls
through to this registry, then to entry points.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

EnvironmentFactory = Callable[..., Any]

_REGISTRY: Dict[str, EnvironmentFactory] = {}
_ENTRY_POINTS_LOADED = False


def register_environment(name: str, factory: EnvironmentFactory) -> None:
    """Register an environment factory under `name` (callable taking the scenario config)."""
    _REGISTRY[name] = factory


def _load_entry_points() -> None:
    """Discover environments other installed packages advertise (best effort, once)."""
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    try:
        from importlib.metadata import entry_points

        # The API differs across 3.9 (dict) and 3.10+ (EntryPoints.select); treat the
        # result as opaque so one code path covers both.
        found: Any = entry_points()
        if hasattr(found, "select"):
            eps = found.select(group="agentsynth.environments")
        else:
            eps = found.get("agentsynth.environments", [])
        for ep in eps:
            if ep.name not in _REGISTRY:
                _REGISTRY[ep.name] = ep.load()
    except Exception:
        # a broken third-party entry point should never take the whole import down
        pass


def get_environment_factory(name: str) -> Optional[EnvironmentFactory]:
    """The factory registered (or advertised) for `name`, or None."""
    if name in _REGISTRY:
        return _REGISTRY[name]
    _load_entry_points()
    return _REGISTRY.get(name)


def available_environments() -> List[str]:
    """Every plugin environment name known so far (registered + advertised)."""
    _load_entry_points()
    return sorted(_REGISTRY)


__all__ = [
    "register_environment",
    "get_environment_factory",
    "available_environments",
]
