"""An environment that turns an OpenAPI spec into runnable tools.

Point it at a spec — a dict, a JSON/YAML string, a file path, or a URL — and every
operation becomes a tool (`operationId` → tool name). Calls go over plain HTTP with
the stdlib, so there's nothing extra to install, and the observation is the real
response body. Path, query, and JSON-body parameters are flattened into one args
object and routed back to the right place on the wire; local `$ref`s are resolved.

    env = RestEnvironment("https://petstore3.swagger.io/api/v3/openapi.json")
    env = RestEnvironment(spec_dict, base_url="http://localhost:8000")
    gen = AgentTrajectoryGenerator(environment=env)

WARNING: the calls are real. Point it at a sandbox or staging server when the spec
has destructive operations, or pass `methods=("get",)` to expose only reads.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib import error, parse, request

from ..schemas import ToolSpec
from .base import Environment
from .mcp_env import _synth_args

_HTTP_METHODS = ("get", "post", "put", "patch", "delete")
_MAX_CHARS = 2000


def _resolve_refs(node: Any, root: Dict[str, Any], seen: Optional[set] = None) -> Any:
    """Inline local `#/...` references, guarding against cycles."""
    seen = seen or set()
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/"):
            if ref in seen:
                return {}
            target: Any = root
            for part in ref[2:].split("/"):
                target = target.get(part, {}) if isinstance(target, dict) else {}
            return _resolve_refs(target, root, seen | {ref})
        return {k: _resolve_refs(v, root, seen) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_refs(v, root, seen) for v in node]
    return node


def _load_spec(spec: Any) -> Dict[str, Any]:
    if isinstance(spec, dict):
        return spec
    if not isinstance(spec, str):
        raise TypeError("spec must be a dict, a JSON/YAML string, a file path, or a URL")
    text = spec
    if spec.startswith(("http://", "https://")):
        with request.urlopen(spec, timeout=20) as resp:
            text = resp.read().decode("utf-8", "replace")
    elif os.path.exists(spec):
        with open(spec, encoding="utf-8") as fh:
            text = fh.read()
    try:
        return json.loads(text)
    except ValueError:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError(
                "the spec isn't JSON and PyYAML isn't installed (pip install pyyaml)"
            ) from exc
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError("the spec didn't parse to an object")
        return loaded


def _fallback_name(method: str, path: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", path).strip("_") or "root"
    return f"{method}_{slug}"


class RestEnvironment(Environment):
    name = "rest"

    def __init__(
        self,
        spec: Any,
        base_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 15.0,
        methods: Optional[Sequence[str]] = None,
        max_chars: int = _MAX_CHARS,
    ) -> None:
        self._spec = _load_spec(spec)
        self._headers = dict(headers or {})
        self.timeout = timeout
        self.max_chars = max_chars
        self._methods = tuple(m.lower() for m in methods) if methods else _HTTP_METHODS

        servers = self._spec.get("servers") or []
        self.base_url = (base_url or (servers[0].get("url") if servers else "") or "").rstrip("/")
        if not self.base_url:
            raise ValueError("no base_url given and the spec has no servers[0].url")

        self._tools: List[ToolSpec] = []
        # tool name -> (method, path template, {arg: path|query|body})
        self._routes: Dict[str, Tuple[str, str, Dict[str, str]]] = {}
        self._parse_operations()

    # -- spec parsing ----------------------------------------------------------

    def _parse_operations(self) -> None:
        for path, item in (self._spec.get("paths") or {}).items():
            if not isinstance(item, dict):
                continue
            shared = item.get("parameters") or []
            for method in self._methods:
                op = item.get(method)
                if not isinstance(op, dict):
                    continue
                self._add_operation(method, path, op, shared)

    def _add_operation(self, method: str, path: str, op: Dict[str, Any], shared: List[Any]) -> None:
        name = op.get("operationId") or _fallback_name(method, path)
        properties: Dict[str, Any] = {}
        required: List[str] = []
        where: Dict[str, str] = {}

        for raw in list(shared) + list(op.get("parameters") or []):
            param = _resolve_refs(raw, self._spec)
            if not isinstance(param, dict) or param.get("in") not in ("path", "query"):
                continue
            arg = param.get("name")
            if not arg:
                continue
            schema = param.get("schema") or {"type": "string"}
            if param.get("description") and "description" not in schema:
                schema = dict(schema, description=param["description"])
            properties[arg] = schema
            where[arg] = param["in"]
            if param.get("in") == "path" or param.get("required"):
                required.append(arg)

        body = _resolve_refs(
            ((op.get("requestBody") or {}).get("content") or {})
            .get("application/json", {})
            .get("schema"),
            self._spec,
        )
        if isinstance(body, dict) and body.get("type") == "object":
            for arg, schema in (body.get("properties") or {}).items():
                properties.setdefault(arg, schema)
                where.setdefault(arg, "body")
            required += [a for a in body.get("required") or [] if a not in required]

        description = op.get("summary") or op.get("description") or f"{method.upper()} {path}"
        self._tools.append(
            ToolSpec(
                name=name,
                description=description,
                parameters={"type": "object", "properties": properties, "required": required},
            )
        )
        self._routes[name] = (method, path, where)

    # -- Environment API -------------------------------------------------------

    def tools(self) -> List[ToolSpec]:
        return list(self._tools)

    def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        route = self._routes.get(tool_name)
        if route is None:
            raise KeyError(tool_name)
        method, path, where = route
        args = args or {}

        query: Dict[str, Any] = {}
        body: Dict[str, Any] = {}
        for arg, value in args.items():
            place = where.get(arg)
            if place == "path":
                path = path.replace("{" + arg + "}", parse.quote(str(value), safe=""))
            elif place == "query" and value is not None:
                query[arg] = value
            elif place == "body":
                body[arg] = value

        missing = re.findall(r"\{([^{}]+)\}", path)
        if missing:
            return f"RestError: missing path parameter(s) {', '.join(sorted(set(missing)))}"

        url = self.base_url + path
        if query:
            url += "?" + parse.urlencode(query, doseq=True)
        data = json.dumps(body).encode("utf-8") if body else None
        headers = {"Accept": "application/json", **self._headers}
        if data is not None:
            headers.setdefault("Content-Type", "application/json")

        req = request.Request(url, data=data, method=method.upper(), headers=headers)
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", "replace")
        except error.HTTPError as exc:
            snippet = exc.read().decode("utf-8", "replace")[:200].strip()
            detail = f" — {snippet}" if snippet else ""
            return f"RestError: HTTP {exc.code} {exc.reason}{detail}"
        except error.URLError as exc:
            return f"RestError: {exc.reason}"
        except TimeoutError:
            return f"RestError: timed out after {self.timeout:g}s"
        return self._render(raw)

    def _render(self, raw: str) -> str:
        text = raw.strip()
        try:
            text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
        except ValueError:
            pass
        if len(text) > self.max_chars:
            text = text[: self.max_chars - 1].rstrip() + "…"
        return text or "(empty response)"

    def sample_args(self, tool_name: str, query: str, seed: int) -> Dict[str, Any]:
        for spec in self._tools:
            if spec.name == tool_name:
                return _synth_args(spec, query, seed)
        return {}


__all__ = ["RestEnvironment"]
