"""Execution environments that run tool calls for real."""

from __future__ import annotations

from .base import CompositeEnvironment, Environment
from .browser import BrowserEnvironment
from .mcp_env import MCPEnvironment
from .python_sandbox import PythonSandbox
from .sql import SQLEnvironment

__all__ = [
    "Environment",
    "CompositeEnvironment",
    "SQLEnvironment",
    "PythonSandbox",
    "MCPEnvironment",
    "BrowserEnvironment",
]
