"""Bring real agent traces into the engine: import, then verify, judge, export.

Production agent logs are already a dataset — they're just not trainable yet. These
importers convert the two formats almost every stack emits into `Trajectory`
objects, so the whole pipeline applies to *real* traffic: judge it, verify it,
dedup it, mine its failures, export SFT/DPO.

    trajectories = load_traces_jsonl("prod_logs.jsonl")        # auto-detects format
    results = TrajectoryEvaluator().evaluate_batch(trajectories)
    keep = [t for t, r in zip(trajectories, results) if r.passed]

Supported shapes:
- **OpenAI-style chat messages** — `role` user/assistant/tool, assistant
  `tool_calls` with JSON-string arguments. What the OpenAI SDK and most proxies
  (LiteLLM, OpenRouter, vLLM) log.
- **Anthropic Messages content blocks** — `tool_use` / `tool_result` / `text`.
- Anything close to `[{"role", "content"}, ...]` falls back to the OpenAI path.

Tool schemas usually aren't in the logs; pass `tools=` when you have them so
verification can check required arguments — otherwise minimal specs are inferred
from the calls themselves.
"""

from __future__ import annotations

import json
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
    """OpenAI-style chat messages → a Trajectory (roughly `to_messages` inverted).

    The first user message becomes the query; assistant text becomes thoughts —
    except the last one, which becomes the final answer; `tool_calls` become
    tool_call steps (JSON-string arguments are parsed); tool/function-role
    messages become observations.
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


def import_traces(
    records: Sequence[Union[MessageList, Dict[str, Any]]],
    tools: Optional[Sequence[Any]] = None,
    format: str = "auto",
) -> List[Trajectory]:
    """Convert a batch of traces. Each record is a message list or a dict holding one
    under a `messages` key. `format` is `auto`, `openai`, or `anthropic`."""
    trajectories: List[Trajectory] = []
    for record in records:
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


__all__ = [
    "trajectory_from_messages",
    "trajectory_from_anthropic",
    "import_traces",
    "load_traces_jsonl",
]
