"""Shared helpers for AgentSynth: tool-catalog parsing, a toy Python REPL, and a
LiteLLM-backed client that falls back to a no-op when no provider key is set.
Nothing here imports heavy/optional deps at module load.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import re
import time
import traceback
from typing import Any, Dict, List, Optional, Union

from .schemas import ToolSpec

# Default catalog backing the UI, examples, and mock generation.
DEFAULT_TOOL_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "web_search",
        "description": "Search the web for up-to-date information and return the top results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "top_k": {"type": "integer", "description": "Number of results to return"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression and return the numeric result.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "e.g. '23 * (4 + 1)'"}},
            "required": ["expression"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get the current weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {
                    "type": "string",
                    "description": "Temperature unit",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["city"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file from the workspace.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to the file"}},
            "required": ["path"],
        },
    },
    {
        "name": "sql_query",
        "description": "Run a read-only SQL query against the analytics database.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A SELECT statement"}},
            "required": ["query"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
]


def safe_json_loads(text: str, default: Any = None) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return default


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Optional[Union[dict, list]]:
    """Pull the first JSON object/array out of LLM text.

    Handles ```json fenced blocks and bare objects/arrays embedded in prose.
    """
    if not text:
        return None
    candidates: List[str] = []
    for m in _JSON_FENCE_RE.finditer(text):
        candidates.append(m.group(1).strip())
    candidates.append(text.strip())

    # Also try the outermost {...} or [...] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start : end + 1])

    for cand in candidates:
        parsed = safe_json_loads(cand)
        if isinstance(parsed, (dict, list)):
            return parsed
    return None


def stable_seed(*parts: Any) -> int:
    """Deterministic int seed from arbitrary parts, so mocks are reproducible."""
    h = hashlib.sha256("||".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:12], 16)


def now_ts() -> float:
    return time.time()


def _normalise_one(raw: Dict[str, Any]) -> Optional[ToolSpec]:
    if not isinstance(raw, dict):
        return None
    # Unwrap OpenAI-style {"type": "function", "function": {...}}.
    if "function" in raw and isinstance(raw["function"], dict):
        raw = raw["function"]
    name = raw.get("name")
    if not name:
        return None
    # Deep-copy so the ToolSpec never aliases the caller's dict (or our module-global
    # default catalog) — editing one tool's schema must not leak into the others.
    raw_params = raw.get("parameters") or raw.get("input_schema")
    params = copy.deepcopy(raw_params) if isinstance(raw_params, dict) else {}
    params.setdefault("type", "object")
    params.setdefault("properties", {})
    params.setdefault("required", [])
    return ToolSpec(
        name=str(name),
        description=str(raw.get("description", "")),
        parameters=params,
    )


def parse_tool_catalog(raw: Union[str, List[Any], Dict[str, Any], None]) -> List[ToolSpec]:
    """Coerce assorted inputs into a list of `ToolSpec`.

    Accepts a JSON string, a list of tool dicts, a single tool dict, or the
    OpenAI `{"tools": [...]}` / function-calling shapes. Bad entries are dropped
    so user input in the UI never hard-crashes us.
    """
    if raw is None or raw == "":
        return [t for t in (_normalise_one(d) for d in DEFAULT_TOOL_CATALOG) if t]

    data: Any = raw
    if isinstance(raw, str):
        data = safe_json_loads(raw)
        if data is None:
            return []

    if isinstance(data, dict):
        if "tools" in data and isinstance(data["tools"], list):
            data = data["tools"]
        else:
            data = [data]

    if not isinstance(data, list):
        return []

    out: List[ToolSpec] = []
    for entry in data:
        spec = _normalise_one(entry)
        if spec is not None:
            out.append(spec)
    return out


def tool_catalog_to_json(tools: List[ToolSpec], indent: int = 2) -> str:
    return json.dumps([t.model_dump() for t in tools], indent=indent)


def default_tool_catalog() -> List[ToolSpec]:
    return parse_tool_catalog(DEFAULT_TOOL_CATALOG)


# Modules the REPL is allowed to import.
_ALLOWED_MODULES = {
    "math",
    "random",
    "statistics",
    "json",
    "re",
    "datetime",
    "itertools",
    "functools",
    "collections",
    "string",
    "decimal",
    "fractions",
    "heapq",
    "bisect",
    "textwrap",
    "numpy",
    "pandas",
}

_BLOCKED_PATTERNS = (
    "open(",
    "subprocess",
    "os.system",
    "os.popen",
    "shutil",
    "socket",
    "eval(",
    "exec(",
    "__import__",
    "globals(",
    "compile(",
)


def _guarded_import(name, *args, **kwargs):
    root = name.split(".")[0]
    if root not in _ALLOWED_MODULES:
        raise ImportError(f"import of module '{name}' is not allowed in the REPL sandbox")
    return __import__(name, *args, **kwargs)


class PythonREPL:
    """A tiny Python REPL for grounding synthetic `code_execution` steps in real stdout.

    WARNING: this is not a security boundary. Imports are restricted to a
    numeric/data whitelist and the most obvious dangerous patterns are blocked,
    but do not run untrusted code through it on a sensitive host.

    The namespace persists across `run` calls, like a real REPL.
    """

    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout = timeout
        self._ns: Dict[str, Any] = {}
        self._reset_ns()

    def _reset_ns(self) -> None:
        safe_builtins: Dict[str, Any] = {}
        import builtins as _b

        for fn in (
            "abs",
            "all",
            "any",
            "bin",
            "bool",
            "dict",
            "divmod",
            "enumerate",
            "filter",
            "float",
            "format",
            "frozenset",
            "hex",
            "int",
            "isinstance",
            "issubclass",
            "iter",
            "len",
            "list",
            "map",
            "max",
            "min",
            "next",
            "oct",
            "ord",
            "chr",
            "pow",
            "print",
            "range",
            "repr",
            "reversed",
            "round",
            "set",
            "slice",
            "sorted",
            "str",
            "sum",
            "tuple",
            "type",
            "zip",
        ):
            if hasattr(_b, fn):
                safe_builtins[fn] = getattr(_b, fn)
        for exc in (
            "ValueError",
            "TypeError",
            "KeyError",
            "IndexError",
            "ZeroDivisionError",
            "Exception",
        ):
            safe_builtins[exc] = getattr(_b, exc)
        safe_builtins["__import__"] = _guarded_import
        self._ns = {"__builtins__": safe_builtins}

    def reset(self) -> None:
        self._reset_ns()

    def run(self, code: str) -> str:
        """Run `code`, returning captured stdout plus the last expression value.

        On error, returns a compact traceback string instead of raising.
        """
        if not code or not code.strip():
            return ""

        lowered = code.lower()
        for pat in _BLOCKED_PATTERNS:
            if pat in lowered and pat not in ("eval(", "exec(", "compile("):
                return f"REPLError: blocked operation '{pat}' is not permitted in the sandbox"

        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                self._exec_with_echo(code)
            out = stdout.getvalue()
            return out.rstrip("\n") if out else ""
        except Exception:
            tb = traceback.format_exc(limit=2)
            captured = stdout.getvalue()
            return (captured + tb).strip()

    def _exec_with_echo(self, code: str) -> None:
        import ast

        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError:
            # Let plain exec raise a clean error.
            exec(compile(code, "<repl>", "exec"), self._ns)
            return

        # Echo the trailing expression's value, the way an interactive REPL does.
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = tree.body.pop()
            assert isinstance(last, ast.Expr)  # checked above; keeps mypy happy
            if tree.body:
                exec(compile(tree, "<repl>", "exec"), self._ns)
            value = eval(compile(ast.Expression(last.value), "<repl>", "eval"), self._ns)
            if value is not None:
                print(repr(value))
        else:
            exec(compile(tree, "<repl>", "exec"), self._ns)


# Provider key -> default model id, checked in order.
_PROVIDER_DEFAULTS = [
    ("ANTHROPIC_API_KEY", "claude-3-5-haiku-latest"),
    ("XAI_API_KEY", "xai/grok-2-latest"),
    ("GROQ_API_KEY", "groq/llama-3.3-70b-versatile"),
    ("OPENAI_API_KEY", "gpt-4o-mini"),
    ("OPENROUTER_API_KEY", "openrouter/meta-llama/llama-3.1-70b-instruct"),
]


def detect_default_model() -> Optional[str]:
    """Pick a default model id from whichever provider key is in the env."""
    for env_key, model in _PROVIDER_DEFAULTS:
        if os.environ.get(env_key):
            return model
    return None


class LLMClient:
    """Thin wrapper over `litellm.completion`.

    When LiteLLM isn't installed or no provider key/model is configured,
    `available` is False and `complete` returns `""`; callers then fall back to
    deterministic mock generation.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1536,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model or detect_default_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.last_error: Optional[str] = None
        self._litellm: Any = None
        self._import_litellm()

    def _import_litellm(self) -> None:
        try:
            import litellm  # type: ignore

            litellm.drop_params = True
            self._litellm = litellm
        except Exception as exc:
            self.last_error = f"litellm unavailable: {exc}"
            self._litellm = None

    @property
    def available(self) -> bool:
        return self._litellm is not None and bool(self.model)

    def complete(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Return the assistant text for `messages`, or `""` on any failure."""
        if not self.available:
            self.last_error = self.last_error or "LLM client not available"
            return ""
        try:
            resp = self._litellm.completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                api_key=self.api_key,
                **kwargs,
            )
            return resp["choices"][0]["message"]["content"] or ""
        except Exception as exc:
            self.last_error = f"completion failed: {exc}"
            return ""


__all__ = [
    "DEFAULT_TOOL_CATALOG",
    "safe_json_loads",
    "extract_json",
    "stable_seed",
    "now_ts",
    "parse_tool_catalog",
    "tool_catalog_to_json",
    "default_tool_catalog",
    "PythonREPL",
    "LLMClient",
    "detect_default_model",
]
