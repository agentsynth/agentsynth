"""Declarative run recipes.

A Recipe captures everything a generation run needs in one object that can also
be loaded from YAML. `make_environment` turns the short `environment` string
("sql", "python", "sql+python") into a real environment.
"""

from __future__ import annotations

import re
from typing import List, Optional, Union

from pydantic import BaseModel

from ..environments.base import Environment


class Recipe(BaseModel):
    name: str = "agentsynth-run"

    # One of: an explicit query, a list of queries, or sampled from the taxonomy.
    query: Optional[str] = None
    queries: Optional[List[str]] = None
    domains: Optional[List[str]] = None

    num_trajectories: int = 10
    mode: str = "single_agent"
    vary_modes: bool = False
    max_steps: int = 6
    temperature: float = 0.7
    model: Optional[str] = None
    use_mock: Union[str, bool] = "auto"
    seed: int = 0

    environment: Optional[str] = None  # "sql" | "python" | "sql+python"
    evaluate: bool = True
    rubric: Optional[str] = None  # judge preset: balanced / strict / lenient / safety_first
    verify: bool = False  # run execution / tool-arg / safety verifiers
    dedup: bool = False  # drop near-duplicate trajectories before scoring
    dedup_threshold: float = 0.85
    export_format: Optional[str] = None  # jsonl | sharegpt | adp | parquet
    export_path: Optional[str] = None
    max_workers: int = 1


def load_recipe(path: str) -> Recipe:
    import yaml

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("a recipe file must be a YAML mapping")
    return Recipe(**data)


def make_environment(spec: Optional[str]) -> Optional[Environment]:
    if not spec:
        return None
    from ..environments import CompositeEnvironment, PythonSandbox, SQLEnvironment

    envs: List[Environment] = []
    for part in re.split(r"[+,]", spec.strip().lower()):
        part = part.strip()
        if not part:
            continue
        if part == "sql":
            envs.append(SQLEnvironment())
        elif part == "python":
            envs.append(PythonSandbox())
        else:
            raise ValueError(f"unknown environment '{part}' (expected sql or python)")
    if not envs:
        return None
    return envs[0] if len(envs) == 1 else CompositeEnvironment(envs)


__all__ = ["Recipe", "load_recipe", "make_environment"]
