"""Execution environments that run tool calls for real."""

from __future__ import annotations

from .base import CompositeEnvironment, Environment
from .browser import BrowserEnvironment
from .docker_sandbox import DockerSandbox
from .mcp_env import MCPEnvironment
from .python_sandbox import PythonSandbox
from .rest import RestEnvironment
from .sql import SQLEnvironment

__all__ = [
    "Environment",
    "CompositeEnvironment",
    "SQLEnvironment",
    "PythonSandbox",
    "MCPEnvironment",
    "BrowserEnvironment",
    "DockerSandbox",
    "RestEnvironment",
]
