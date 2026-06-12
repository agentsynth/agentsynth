"""Import agent traces (OpenAI- and Anthropic-format logs) as Trajectories.

Once imported, the rest of the pipeline applies: judging, verification, dedup,
failure mining, SFT/DPO export.

    trajectories = load_traces_jsonl("prod_logs.jsonl")   # format auto-detected
    results = TrajectoryEvaluator().evaluate_batch(trajectories)

Two shapes are recognized: OpenAI-style chat messages (assistant `tool_calls`
with JSON-string arguments — also what LiteLLM/OpenRouter/vLLM log) and
Anthropic content blocks (`tool_use` / `tool_result`). Anything close to
`[{"role", "content"}, ...]` goes down the OpenAI path. Tool schemas usually
aren't in the logs; pass `tools=` when you have them, otherwise minimal specs
are inferred from the calls.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Union

from .schemas import ToolSpec, Trajectory, TrajectoryStep
from .utils import stable_seed

MessageList = List[Dict[str, Any]]


def _stable_id(messages: MessageList) -> str:
    key = json.dumps(messages, sort_keys=True, ensure_ascii=False, default=str)
    return format(stable_seed(0, key) & 0xFFFFFFFFFFFF, "012x")


def _parse_args(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
    return {}


def _infer_tools(steps: Sequence[TrajectoryStep]) -> List[ToolSpec]:
    names: List[str] = []
    for step in steps:
        if step.step_type == "tool_call" and step.tool_name and step.tool_name not in names:
            names.append(step.tool_name)
    return [
        ToolSpec(name=n, description="inferred from a trace", parameters={"type": "object"})
        for n in names
    ]


def _content_text(content: Any) -> str:
    """Flatten message content that may be a string or a list of text blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def _looks_anthropic(messages: MessageList) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in ("tool_use", "tool_result"):
                    return True
    return False


def trajectory_from_messages(
    messages: MessageList,
    tools: Optional[Sequence[Any]] = None,
    query: Optional[str] = None,
    domain: Optional[str] = None,
    source: str = "openai",
) -> Trajectory:
    """OpenAI-style chat messages to a Trajectory (roughly `to_messages` inverted).

    First user message becomes the query. Assistant text becomes thoughts, except
    the last one, which becomes the final answer. `tool_calls` become tool_call
    steps and tool/function-role messages become observations.
    """
    steps: List[TrajectoryStep] = []
    resolved_query = query or ""
    last_text_idx = -1

    for message in messages:
        role = message.get("role")
        text = _content_text(message.get("content"))
        if role == "user" and not resolved_query:
            resolved_query = text
            continue
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function") if isinstance(call, dict) else None
                name = (function or {}).get("name") or (call or {}).get("name")
                if name:
                    steps.append(
                        TrajectoryStep(
                            step_type="tool_call",
                            tool_name=name,
                            tool_args=_parse_args((function or call or {}).get("arguments")),
                        )
                    )
            if text:
                steps.append(TrajectoryStep(step_type="thought", thought=text))
                last_text_idx = len(steps) - 1
        elif role in ("tool", "function"):
            steps.append(TrajectoryStep(step_type="observation", observation=text))

    final_answer = ""
    if last_text_idx >= 0:
        final_answer = steps[last_text_idx].thought or ""
        steps[last_text_idx] = TrajectoryStep(step_type="final_answer", content=final_answer)

    return _wrap(messages, steps, resolved_query, tools, domain, source, final_answer)


def trajectory_from_anthropic(
    messages: MessageList,
    tools: Optional[Sequence[Any]] = None,
    query: Optional[str] = None,
    domain: Optional[str] = None,
) -> Trajectory:
    """Anthropic Messages content blocks (`tool_use` / `tool_result`) → a Trajectory."""
    steps: List[TrajectoryStep] = []
    resolved_query = query or ""
    last_text_idx = -1

    for message in messages:
        role = message.get("role")
        content = message.get("content")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
        for block in blocks:
            if not isinstance(block, dict):
                block = {"type": "text", "text": str(block)}
            kind = block.get("type")
            if kind == "text":
                text = str(block.get("text") or "")
                if not text:
                    continue
                if role == "user" and not resolved_query:
                    resolved_query = text
                elif role == "assistant":
                    steps.append(TrajectoryStep(step_type="thought", thought=text))
                    last_text_idx = len(steps) - 1
            elif kind == "tool_use" and role == "assistant":
                steps.append(
                    TrajectoryStep(
                        step_type="tool_call",
                        tool_name=str(block.get("name") or ""),
                        tool_args=_parse_args(block.get("input")),
                    )
                )
            elif kind == "tool_result":
                steps.append(
                    TrajectoryStep(
                        step_type="observation",
                        observation=_content_text(block.get("content")),
                    )
                )

    final_answer = ""
    if last_text_idx >= 0:
        final_answer = steps[last_text_idx].thought or ""
        steps[last_text_idx] = TrajectoryStep(step_type="final_answer", content=final_answer)

    return _wrap(messages, steps, resolved_query, tools, domain, "anthropic", final_answer)


def _wrap(
    messages: MessageList,
    steps: List[TrajectoryStep],
    query: str,
    tools: Optional[Sequence[Any]],
    domain: Optional[str],
    source: str,
    final_answer: str,
) -> Trajectory:
    from .utils import parse_tool_catalog

    if tools:
        specs = parse_tool_catalog(list(tools)) if isinstance(tools[0], dict) else list(tools)
    else:
        specs = _infer_tools(steps)
    return Trajectory(
        id=_stable_id(messages),
        query=query,
        mode="single_agent",
        tools=specs,
        steps=steps,
        final_answer=final_answer,
        success=bool(final_answer),
        generator_model=f"import:{source}",
        metadata={"source": f"trace:{source}", "imported": True},
    )


def _flat_attrs(span: Dict[str, Any]) -> Dict[str, Any]:
    """Span attributes, whether a flat dict or the OTLP-JSON key/value list."""
    attrs = span.get("attributes") or {}
    if isinstance(attrs, dict):
        return attrs
    flat: Dict[str, Any] = {}
    for item in attrs:
        if not isinstance(item, dict) or "key" not in item:
            continue
        value = item.get("value") or {}
        if isinstance(value, dict):
            for field in ("stringValue", "intValue", "doubleValue", "boolValue"):
                if field in value:
                    value = value[field]
                    break
        flat[item["key"]] = value
    return flat


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str) and value.strip().startswith(("[", "{")):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _otel_to_openai(message: Any) -> Optional[Dict[str, Any]]:
    """One semconv message ({role, parts} or {role, content}) to OpenAI shape."""
    if not isinstance(message, dict):
        return None
    role = message.get("role") or "assistant"
    if "parts" not in message:
        return {"role": role, "content": str(message.get("content") or "")}
    texts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for part in message.get("parts") or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            texts.append(str(part.get("content") or ""))
        elif part.get("type") == "tool_call":
            tool_calls.append(
                {
                    "function": {
                        "name": part.get("name"),
                        "arguments": _maybe_json(part.get("arguments")) or {},
                    }
                }
            )
        elif part.get("type") == "tool_call_response":
            return {
                "role": "tool",
                "content": str(part.get("result") or part.get("response") or ""),
            }
    out: Dict[str, Any] = {"role": role, "content": "\n".join(t for t in texts if t)}
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def _flattened_otel_messages(attrs: Dict[str, Any], prefix: str) -> List[Dict[str, Any]]:
    """Rebuild messages from flattened keys (gen_ai.prompt.0.role / .content, ...)."""
    by_index: Dict[int, Dict[str, Any]] = {}
    for key, value in attrs.items():
        parts = key.split(".")
        if len(parts) >= 4 and parts[0] == "gen_ai" and parts[1] == prefix and parts[2].isdigit():
            by_index.setdefault(int(parts[2]), {})[".".join(parts[3:])] = value
    messages = []
    for index in sorted(by_index):
        fields = by_index[index]
        message: Dict[str, Any] = {
            "role": fields.get("role", "user" if prefix == "prompt" else "assistant"),
            "content": fields.get("content", ""),
        }
        name = fields.get("tool_calls.0.name") or fields.get("function_call.name")
        if name:
            args = fields.get("tool_calls.0.arguments") or fields.get("function_call.arguments")
            message["tool_calls"] = [{"function": {"name": name, "arguments": args or {}}}]
        messages.append(message)
    return messages


def trajectory_from_otel_spans(
    spans: Sequence[Dict[str, Any]],
    tools: Optional[Sequence[Any]] = None,
    query: Optional[str] = None,
) -> Trajectory:
    """OpenTelemetry GenAI spans to a Trajectory.

    The GenAI semconv is still incubating, so two common encodings are accepted:
    `gen_ai.input.messages` / `gen_ai.output.messages` JSON attributes on chat
    spans, and flattened `gen_ai.prompt.{i}.*` / `gen_ai.completion.{i}.*` keys.
    Tool spans (`gen_ai.operation.name` == "execute_tool") become a tool call
    plus its observation. Spans are ordered by `start_time_unix_nano` when present.
    """
    ordered = sorted(spans, key=lambda s: int(s.get("start_time_unix_nano") or 0))
    messages: MessageList = []
    saw_input = False

    for span in ordered:
        attrs = _flat_attrs(span)
        operation = attrs.get("gen_ai.operation.name") or str(span.get("name") or "")
        if operation.startswith("execute_tool") or attrs.get("gen_ai.tool.name"):
            name = attrs.get("gen_ai.tool.name") or operation.replace("execute_tool ", "")
            args = _maybe_json(attrs.get("gen_ai.tool.call.arguments")) or {}
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": name, "arguments": args}}],
                }
            )
            result = attrs.get("gen_ai.tool.call.result")
            if result is not None:
                messages.append({"role": "tool", "content": str(result)})
            continue

        inputs = _maybe_json(attrs.get("gen_ai.input.messages"))
        if not isinstance(inputs, list):
            inputs = _flattened_otel_messages(attrs, "prompt")
        if inputs and not saw_input:
            saw_input = True
            for message in inputs:
                converted = _otel_to_openai(message)
                if converted:
                    messages.append(converted)

        outputs = _maybe_json(attrs.get("gen_ai.output.messages"))
        if not isinstance(outputs, list):
            outputs = _flattened_otel_messages(attrs, "completion")
        for message in outputs or []:
            converted = _otel_to_openai(message)
            if converted:
                messages.append(converted)

    return trajectory_from_messages(messages, tools=tools, query=query, source="otel")


def import_traces(
    records: Sequence[Union[MessageList, Dict[str, Any]]],
    tools: Optional[Sequence[Any]] = None,
    format: str = "auto",
) -> List[Trajectory]:
    """Convert a batch of traces. Each record is a message list, a dict with a
    `messages` key, or a dict with a `spans` key (OTel). `format` is `auto`,
    `openai`, `anthropic`, or `otel`."""
    trajectories: List[Trajectory] = []
    for record in records:
        if isinstance(record, dict) and (format in ("auto", "otel")) and "spans" in record:
            spans = record.get("spans")
            if isinstance(spans, list) and spans:
                trajectories.append(trajectory_from_otel_spans(spans, tools=tools))
            continue
        messages = record.get("messages") if isinstance(record, dict) else record
        if not isinstance(messages, list) or not messages:
            continue
        kind = format
        if kind == "auto":
            kind = "anthropic" if _looks_anthropic(messages) else "openai"
        if kind == "anthropic":
            trajectories.append(trajectory_from_anthropic(messages, tools=tools))
        else:
            trajectories.append(trajectory_from_messages(messages, tools=tools))
    return trajectories


def load_traces_jsonl(
    path: str, tools: Optional[Sequence[Any]] = None, format: str = "auto"
) -> List[Trajectory]:
    """Read one trace per line (a message list, or an object with `messages`)."""
    with open(path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    return import_traces(records, tools=tools, format=format)


# Conservative by design: secrets and contact details go, plain numbers stay
# ("order 7" must survive). Phones need separators or a leading + to match.
_REDACTIONS = [
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("token", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}")),
    ("token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._+/=-]{12,}")),
    ("api_key", re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}\b")),
    ("api_key", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("api_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("hex_id", re.compile(r"\b[0-9a-fA-F]{32,}\b")),
    ("phone", re.compile(r"(?<![\w.])\+?\d{1,4}[ .-]\(?\d{2,4}\)?[ .-]\d{3,4}(?:[ .-]\d{2,4})?\b")),
]


def redact_text(text: str) -> str:
    """Strip emails, keys, tokens, long hex ids, and phone-shaped numbers."""
    for label, pattern in _REDACTIONS:
        text = pattern.sub(f"[redacted-{label}]", text)
    return text


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {k: _redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(v) for v in value]
    return value


def redact_trajectory(traj: Trajectory) -> Trajectory:
    """Redact every text surface of a trajectory in place, then return it.

    Run this before sharing or donating imported production traces.
    """
    traj.query = redact_text(traj.query or "")
    if traj.final_answer:
        traj.final_answer = redact_text(traj.final_answer)
    for step in traj.steps:
        for field in ("thought", "content", "observation", "code", "code_output"):
            value = getattr(step, field, None)
            if isinstance(value, str) and value:
                setattr(step, field, redact_text(value))
        if step.tool_args:
            step.tool_args = _redact_value(step.tool_args)
    return traj


__all__ = [
    "trajectory_from_messages",
    "trajectory_from_anthropic",
    "trajectory_from_otel_spans",
    "import_traces",
    "load_traces_jsonl",
    "redact_text",
    "redact_trajectory",
]
