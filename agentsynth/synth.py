"""Synthesize verifiable scenarios from a demonstration.

Hand it a seed world and the actions that solve a task. It runs them, diffs the end
state, and writes a scenario whose checkers assert exactly what changed — the robust,
hard-to-game kind, not an answer-string match. The result passes both `pack validate`
(the actions are the oracle) and `pack audit` (the checks are on the world).

This is the cheap-authoring side of the "verifier problem": demonstrate a task once and
the verifier writes itself, instead of hand-writing checkers and hoping they hold.

    demo = {
        "task": "Refund order 7. Leave the others alone.",
        "schema": "CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT)",
        "rows": [[7, "paid"], [8, "paid"]],
        "actions": ["UPDATE orders SET status='refunded' WHERE id=7"],
        "answer": "Refunded order 7.",
    }
    scenario, oracle_actions = scenario_from_demonstration(**demo)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

_CREATE_TABLE_RE = re.compile(r"CREATE\s+TABLE\s+[\"'`]?(\w+)", re.IGNORECASE)

# Don't bury a pack in checks — assert the changed rows up to this many, plus the
# per-table count and one untouched witness row.
_MAX_ROW_CHECKS = 8


def _first_table(schema: str) -> Optional[str]:
    match = _CREATE_TABLE_RE.search(schema)
    return match.group(1) if match else None


def _sql_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if value is None:
        return "NULL"
    return str(value)


def _table_layout(env: Any, table: str) -> Tuple[List[str], Optional[int]]:
    """Column names in order, and the index of the primary-key column (or None)."""
    info = [tuple(r) for r in env.rows(f"PRAGMA table_info({table})")]
    cols = [r[1] for r in info]  # (cid, name, type, notnull, dflt, pk)
    pk_name = next((r[1] for r in info if r[5]), None)
    pk_idx = cols.index(pk_name) if pk_name is not None else None
    return cols, pk_idx


def _live_tables(env: Any) -> List[str]:
    rows = env.rows(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    return [r[0] for r in rows]


def _keyed(rows: Sequence[Tuple[Any, ...]], pk_idx: int) -> Dict[Any, Tuple[Any, ...]]:
    return {row[pk_idx]: row for row in rows}


def _row_check(table: str, pk_col: str, key: Any, row: Tuple[Any, ...]) -> Dict[str, Any]:
    return {
        "kind": "sql",
        "query": f"SELECT * FROM {table} WHERE {pk_col}={_sql_literal(key)}",
        "equals": [list(row)],
    }


def _derive_checks(env_before: Any, env_after: Any) -> List[Dict[str, Any]]:
    """Compare the seeded world to the world after the demo and assert the delta."""
    checks: List[Dict[str, Any]] = []
    budget = _MAX_ROW_CHECKS
    for table in _live_tables(env_after):
        cols, pk_idx = _table_layout(env_after, table)
        before = [tuple(r) for r in env_before.rows(f"SELECT * FROM {table}")]
        after = [tuple(r) for r in env_after.rows(f"SELECT * FROM {table}")]

        if pk_idx is None:
            # No key to diff on — assert the whole table verbatim.
            if before != after:
                checks.append(
                    {
                        "kind": "sql",
                        "query": f"SELECT * FROM {table}",
                        "equals": [list(r) for r in after],
                    }
                )
            continue

        pk_col = cols[pk_idx]
        bmap, amap = _keyed(before, pk_idx), _keyed(after, pk_idx)
        changed = [k for k in amap if k in bmap and amap[k] != bmap[k]]
        inserted = [k for k in amap if k not in bmap]
        deleted = [k for k in bmap if k not in amap]
        touched = bool(changed or inserted or deleted)

        for key in changed + inserted:
            if budget <= 0:
                break
            checks.append(_row_check(table, pk_col, key, amap[key]))
            budget -= 1
        for key in deleted:
            if budget <= 0:
                break
            checks.append(
                {
                    "kind": "sql",
                    "query": f"SELECT COUNT(*) FROM {table} WHERE {pk_col}={_sql_literal(key)}",
                    "equals": [[0]],
                }
            )
            budget -= 1

        if touched:
            # Pin the row count (catches stray inserts/deletes) and one untouched row
            # (catches over-mutation), so the check rewards a surgical change.
            checks.append(
                {"kind": "sql", "query": f"SELECT COUNT(*) FROM {table}", "equals": [[len(after)]]}
            )
            witness = next((k for k in amap if k in bmap and amap[k] == bmap[k]), None)
            if witness is not None:
                checks.append(_row_check(table, pk_col, witness, amap[witness]))
    return checks


def scenario_from_demonstration(
    task: str,
    schema: str,
    actions: Sequence[str],
    rows: Optional[Sequence[Sequence[Any]]] = None,
    table: Optional[str] = None,
    answer: Optional[str] = None,
    scenario_id: str = "demo",
    max_steps: Optional[int] = None,
    seed: int = 7,
) -> Tuple[Any, List[str]]:
    """Build a scenario from a worked example, deriving state checks from the diff.

    `rows=None` means a multi-table world that seeds itself from INSERTs in the schema
    (matching the pack convention). Returns the scenario and the oracle's actions.
    """
    from .environments import SQLEnvironment
    from .scenarios import Scenario

    seed_table = table or _first_table(schema)
    if seed_table is None:
        raise ValueError("could not find a table name in the schema")

    env_rows = [] if rows is None else [tuple(r) for r in rows]
    before = SQLEnvironment(schema=schema, rows=env_rows, table=seed_table, read_only=False)
    after = SQLEnvironment(schema=schema, rows=env_rows, table=seed_table, read_only=False)
    try:
        for statement in actions:
            result = after.execute("sql_query", {"query": statement})
            if str(result).startswith("SQLError"):
                raise ValueError(f"demo action failed: {statement!r} -> {result}")
        checkers = _derive_checks(before, after)
    finally:
        before.close()
        after.close()

    if not checkers:
        raise ValueError(
            "the demo changed nothing, so there is no state to check — add actions that "
            "mutate the world, or write the scenario by hand for a pure read/refusal task"
        )

    # Grounding only — the state checks above are the verification. We deliberately do
    # not add an answer-string check: it would be brittle (the exact wording) and weak
    # (gameable by echo), and the world state already settles whether the task was done.
    checkers.append({"kind": "called_tool", "name": "sql_query"})

    environment: Dict[str, Any] = {"type": "sql", "schema": schema, "table": seed_table}
    environment["rows"] = [list(r) for r in env_rows]
    metadata: Dict[str, Any] = {"source": "demonstration"}
    if answer:
        metadata["answer"] = answer  # the gold final answer travels with the scenario
    scenario = Scenario.model_validate(
        {
            "id": scenario_id,
            "task": task,
            "environment": environment,
            "checkers": checkers,
            "max_steps": max_steps or (len(actions) + 3),
            "metadata": metadata,
        }
    )
    return scenario, list(actions)


def pack_from_demonstrations(demos: Sequence[Dict[str, Any]], pack_id: str) -> Tuple[str, str]:
    """Turn a list of demonstrations into a pack + a matching oracle, ready to validate.

    Mirrors the `--from-schema` output: returns (pack_yaml, oracle_py).
    """
    import json

    import yaml

    scenarios: List[Dict[str, Any]] = []
    plan: Dict[str, List[str]] = {}
    answers: Dict[str, str] = {}
    for i, demo in enumerate(demos):
        sid = str(demo.get("id") or f"demo-{i + 1}")
        scenario, actions = scenario_from_demonstration(
            task=demo["task"],
            schema=demo["schema"],
            actions=demo["actions"],
            rows=demo.get("rows"),
            table=demo.get("table"),
            answer=demo.get("answer"),
            scenario_id=sid,
            max_steps=demo.get("max_steps"),
        )
        scenarios.append(scenario.model_dump())
        plan[sid] = actions
        answers[sid] = demo.get("answer") or "Done."

    header = (
        f"# {pack_id} — generated from demonstrations. Checkers were derived from the\n"
        f"# end state each demo produced, so they assert the change, not the wording.\n"
    )
    pack_yaml = header + yaml.safe_dump(scenarios, sort_keys=False, allow_unicode=True)

    oracle_py = (
        f'"""Auto-generated reference solution for {pack_id} (from demonstrations)."""\n\n'
        f"_PLAN = {json.dumps(plan, indent=4)}\n\n"
        f"_ANSWER = {json.dumps(answers, indent=4)}\n\n\n"
        "def solve(observation, gym):\n"
        '    sid = gym.scenario.id if gym.scenario is not None else ""\n'
        "    plan = _PLAN.get(sid, [])\n"
        "    if gym.step_count < len(plan):\n"
        '        return {"tool_name": "sql_query", "arguments": {"query": plan[gym.step_count]}}\n'
        '    return {"answer": _ANSWER.get(sid, "Done.")}\n'
    )
    return pack_yaml, oracle_py


__all__ = ["scenario_from_demonstration", "pack_from_demonstrations"]
