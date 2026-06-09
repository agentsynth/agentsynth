"""Dataset exporters for AgentSynth.

Turn Trajectory objects into the formats agentic-LLM fine-tuning stacks consume:
JSONL, ShareGPT, ADP, and a flat Parquet table. Only JSONL is required; Parquet
imports pandas lazily and tells you what to install if it's missing.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .schemas import ToolSpec, Trajectory, TrajectoryStep


def to_openai_messages(trajectory: Trajectory) -> List[Dict[str, Any]]:
    return trajectory.to_messages()


# JSONL is the format we round-trip: to_jsonl writes it, load_jsonl reads it
# straight back into the same Trajectory objects.


def _traj_to_record(traj: Trajectory) -> Dict[str, Any]:
    record = {
        "id": traj.id,
        "query": traj.query,
        "mode": traj.mode,
        "domain": traj.domain,
        "messages": traj.to_messages(),
        "tools": [t.model_dump() for t in traj.tools],
        "steps": [s.model_dump() for s in traj.steps],
        "final_answer": traj.final_answer,
        "success": traj.success,
        "generator_model": traj.generator_model,
        "metadata": traj.metadata,
    }
    if traj.verification is not None:
        record["verification"] = traj.verification
    return record


def to_jsonl(trajectories: List[Trajectory], path: Optional[str] = None) -> str:
    """Serialise trajectories to JSONL, one object per line.

    Each line holds the rendered `messages` plus the raw `tools`/`steps` that
    load_jsonl needs to rebuild the trajectory. Writes to `path` (UTF-8) when
    given; either way the JSONL string is returned.
    """
    lines = [json.dumps(_traj_to_record(traj), ensure_ascii=False) for traj in trajectories]
    text = "\n".join(lines)
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return text


def load_jsonl(path: str) -> List[Trajectory]:
    """Read a JSONL file from to_jsonl back into trajectories.

    `messages` is derived from the steps, so it's ignored on load and
    regenerated.
    """
    trajectories: List[Trajectory] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tools = [ToolSpec(**t) for t in obj.get("tools", [])]
            steps = [TrajectoryStep(**s) for s in obj.get("steps", [])]
            traj_kwargs: Dict[str, Any] = {
                "query": obj.get("query", ""),
                "mode": obj.get("mode", "single_agent"),
                "domain": obj.get("domain"),
                "tools": tools,
                "steps": steps,
                "final_answer": obj.get("final_answer", ""),
                "success": obj.get("success", True),
                "generator_model": obj.get("generator_model", "mock"),
                "metadata": obj.get("metadata", {}),
            }
            if "verification" in obj:
                traj_kwargs["verification"] = obj["verification"]
            if obj.get("id"):
                traj_kwargs["id"] = obj["id"]
            trajectories.append(Trajectory(**traj_kwargs))
    return trajectories


def _tool_schemas(traj: Trajectory) -> List[Dict[str, Any]]:
    """OpenAI function-calling tool schemas for a trajectory's tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in traj.tools
    ]


def to_sharegpt(trajectories: List[Trajectory], path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Convert trajectories to the ShareGPT `conversations` format.

    Roles map off Trajectory.to_messages: user becomes human, assistant text
    becomes gpt, assistant tool_calls become function_call (value is JSON), and
    tool becomes observation. Dumps to JSON at `path` when given; the list is
    always returned.
    """
    dataset: List[Dict[str, Any]] = []
    for traj in trajectories:
        conversations: List[Dict[str, str]] = []
        for msg in traj.to_messages():
            role = msg.get("role")
            if role == "user":
                conversations.append({"from": "human", "value": msg.get("content", "")})
            elif role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for call in tool_calls:
                        fn = call.get("function", {})
                        value = json.dumps(
                            {
                                "name": fn.get("name"),
                                "arguments": fn.get("arguments", {}),
                            },
                            ensure_ascii=False,
                        )
                        conversations.append({"from": "function_call", "value": value})
                else:
                    conversations.append({"from": "gpt", "value": msg.get("content", "")})
            elif role == "tool":
                conversations.append({"from": "observation", "value": msg.get("content", "")})
        dataset.append({"conversations": conversations, "tools": _tool_schemas(traj)})

    if path:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(dataset, fh, ensure_ascii=False, indent=2)
    return dataset


def _step_content(step: TrajectoryStep) -> str:
    if step.step_type == "tool_call":
        return json.dumps(
            {"tool": step.tool_name, "args": step.tool_args or {}},
            ensure_ascii=False,
        )
    if step.step_type == "observation":
        return step.observation or ""
    if step.step_type == "code_execution":
        return json.dumps(
            {"code": step.code or "", "output": step.code_output or ""},
            ensure_ascii=False,
        )
    # thought / plan / critique / final_answer
    return step.thought or step.content or ""


def to_adp(trajectories: List[Trajectory], path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Convert trajectories to Agent Data Protocol-style records.

    Each record carries the instruction, the tool catalog (names + parameters),
    a flat list of typed steps, and the final output. Writes JSON to `path` when
    given; the list is always returned.
    """
    dataset: List[Dict[str, Any]] = []
    for traj in trajectories:
        tools = [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in traj.tools
        ]
        steps = [
            {
                "type": step.step_type,
                "agent": step.agent,
                "content": _step_content(step),
            }
            for step in traj.steps
        ]
        dataset.append(
            {
                "schema": "adp/0.1",
                "trajectory_id": traj.id,
                "instruction": traj.query,
                "tools": tools,
                "steps": steps,
                "output": traj.final_answer,
            }
        )

    if path:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(dataset, fh, ensure_ascii=False, indent=2)
    return dataset


def to_parquet(
    trajectories: List[Trajectory],
    path: str,
    eval_results: Optional[List[Any]] = None,
) -> str:
    """Write a flat one-row-per-trajectory Parquet table.

    Columns: id, query, mode, domain, num_steps, num_tool_calls, tools_used,
    final_answer, messages (a JSON string). With `eval_results`, each result's
    overall/passed and the six rubric scores are merged in by trajectory_id.

    Needs pandas and pyarrow; raises ImportError naming what to install.
    """
    try:
        import pandas as pd
    except Exception as exc:
        raise ImportError(
            "to_parquet requires pandas and pyarrow. Install them with: pip install pandas pyarrow"
        ) from exc

    eval_by_id: Dict[str, Dict[str, Any]] = {}
    for res in eval_results or []:
        flat = res.flat() if hasattr(res, "flat") else dict(res)
        tid = flat.get("trajectory_id")
        if tid is not None:
            eval_by_id[tid] = flat

    rows: List[Dict[str, Any]] = []
    for traj in trajectories:
        row: Dict[str, Any] = {
            "id": traj.id,
            "query": traj.query,
            "mode": traj.mode,
            "domain": traj.domain,
            "num_steps": traj.num_steps(),
            "num_tool_calls": len(traj.tool_calls()),
            "tools_used": ",".join(traj.tool_names_used()),
            "final_answer": traj.final_answer,
            "messages": json.dumps(traj.to_messages(), ensure_ascii=False),
        }
        ev = eval_by_id.get(traj.id)
        if ev is not None:
            row["overall"] = ev.get("overall")
            row["passed"] = ev.get("passed")
            for dim in (
                "task_completion",
                "tool_correctness",
                "faithfulness",
                "reasoning_coherence",
                "efficiency",
                "safety",
            ):
                row[dim] = ev.get(dim)
        rows.append(row)

    df = pd.DataFrame(rows)
    try:
        df.to_parquet(path)
    except Exception as exc:
        raise ImportError(
            "Writing Parquet requires pyarrow. Install it with: pip install pyarrow"
        ) from exc
    return path


def save_dataset(
    trajectories: List[Trajectory],
    path: str,
    fmt: str = "jsonl",
    eval_results: Optional[List[Any]] = None,
) -> str:
    """Write trajectories to `path` in the requested format.

    `fmt` is one of jsonl, sharegpt, adp, parquet. Returns the output path and
    raises ValueError on an unknown format.
    """
    if fmt == "jsonl":
        to_jsonl(trajectories, path=path)
        return path
    if fmt == "sharegpt":
        to_sharegpt(trajectories, path=path)
        return path
    if fmt == "adp":
        to_adp(trajectories, path=path)
        return path
    if fmt == "parquet":
        return to_parquet(trajectories, path, eval_results=eval_results)
    raise ValueError(
        f"Unknown export format {fmt!r}; expected one of: jsonl, sharegpt, adp, parquet"
    )


__all__ = [
    "to_openai_messages",
    "to_jsonl",
    "to_sharegpt",
    "to_adp",
    "to_parquet",
    "save_dataset",
    "load_jsonl",
]
