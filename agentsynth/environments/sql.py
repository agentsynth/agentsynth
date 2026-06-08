"""A SQLite-backed analytics environment.

`sql_query` runs against a real in-memory SQLite database seeded with a small
sales table, so the observation is the actual query result. Read-only: anything
other than SELECT/WITH is rejected.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..schemas import ToolSpec
from ..utils import stable_seed
from .base import Environment

_SCHEMA = (
    "CREATE TABLE sales ("
    "id INTEGER PRIMARY KEY, region TEXT, product TEXT, "
    "quarter TEXT, revenue REAL, units INTEGER)"
)

# A small, fixed fixture so results are deterministic and reproducible.
_ROWS: List[Tuple[Any, ...]] = [
    (1, "EMEA", "Widget", "Q1", 12400.0, 248),
    (2, "EMEA", "Gadget", "Q1", 8800.0, 110),
    (3, "AMER", "Widget", "Q1", 15200.0, 304),
    (4, "AMER", "Gizmo", "Q1", 6100.0, 61),
    (5, "APAC", "Gadget", "Q1", 9400.0, 117),
    (6, "EMEA", "Widget", "Q2", 13900.0, 278),
    (7, "AMER", "Gadget", "Q2", 11200.0, 140),
    (8, "APAC", "Widget", "Q2", 17600.0, 352),
    (9, "APAC", "Gizmo", "Q2", 7300.0, 73),
    (10, "EMEA", "Gizmo", "Q2", 5400.0, 54),
    (11, "AMER", "Widget", "Q3", 16100.0, 322),
    (12, "EMEA", "Gadget", "Q3", 10250.0, 128),
    (13, "APAC", "Widget", "Q3", 18900.0, 378),
    (14, "AMER", "Gizmo", "Q3", 6900.0, 69),
    (15, "APAC", "Gadget", "Q3", 12750.0, 159),
    (16, "EMEA", "Widget", "Q4", 14800.0, 296),
    (17, "AMER", "Gadget", "Q4", 13100.0, 164),
    (18, "APAC", "Widget", "Q4", 20400.0, 408),
    (19, "EMEA", "Gizmo", "Q4", 6650.0, 66),
    (20, "AMER", "Widget", "Q4", 17250.0, 345),
]

# Valid example queries used to make generated tool calls actually run.
_SAMPLE_QUERIES = (
    "SELECT region, ROUND(SUM(revenue), 2) AS total_revenue "
    "FROM sales GROUP BY region ORDER BY total_revenue DESC",
    "SELECT product, SUM(units) AS units_sold FROM sales GROUP BY product ORDER BY units_sold DESC",
    "SELECT quarter, ROUND(SUM(revenue), 2) AS revenue "
    "FROM sales GROUP BY quarter ORDER BY quarter",
    "SELECT region, product, revenue FROM sales WHERE quarter = 'Q4' ORDER BY revenue DESC LIMIT 5",
    "SELECT COUNT(*) AS rows, ROUND(AVG(revenue), 2) AS avg_revenue FROM sales",
)


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _format_table(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not columns:
        return "(no result)"
    lines = [" | ".join(columns)]
    for row in rows:
        lines.append(" | ".join(_cell(c) for c in row))
    count = len(rows)
    lines.append(f"({count} row{'s' if count != 1 else ''})")
    return "\n".join(lines)


class SQLEnvironment(Environment):
    name = "sql"

    def __init__(
        self,
        schema: str = _SCHEMA,
        rows: Optional[Sequence[Tuple[Any, ...]]] = None,
        table: str = "sales",
    ) -> None:
        self._schema = schema
        self._rows = list(_ROWS if rows is None else rows)
        self._table = table
        # check_same_thread=False + a lock so the concurrent runner can share one env.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._seed()

    def _seed(self) -> None:
        self._conn.execute(self._schema)
        if self._rows:
            placeholders = ",".join(["?"] * len(self._rows[0]))
            self._conn.executemany(f"INSERT INTO {self._table} VALUES ({placeholders})", self._rows)
            self._conn.commit()

    def reset(self) -> None:
        with self._lock:
            self._conn.close()
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._seed()

    def close(self) -> None:
        self._conn.close()

    def tools(self) -> List[ToolSpec]:
        desc = (
            "Run a read-only SQL SELECT against an analytics database. "
            f"Table `{self._table}`(region, product, quarter, revenue, units)."
        )
        return [
            ToolSpec(
                name="sql_query",
                description=desc,
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "A SELECT statement"}
                    },
                    "required": ["query"],
                },
            )
        ]

    def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        if tool_name != "sql_query":
            raise KeyError(tool_name)
        sql = str((args or {}).get("query", "")).strip()
        if not sql:
            return "SQLError: empty query"
        head = sql.lower().lstrip("( \n\t")
        if not (head.startswith("select") or head.startswith("with")):
            return "SQLError: only read-only SELECT queries are allowed"
        try:
            with self._lock:
                cur = self._conn.execute(sql)
                rows = cur.fetchmany(50)
                columns = [d[0] for d in cur.description] if cur.description else []
        # sqlite3.Warning (e.g. multiple statements) is not an Error subclass.
        except (sqlite3.Error, sqlite3.Warning) as exc:
            return f"SQLError: {exc}"
        return _format_table(columns, rows)

    def sample_args(self, tool_name: str, query: str, seed: int) -> Dict[str, Any]:
        choice = _SAMPLE_QUERIES[stable_seed(seed, "sql") % len(_SAMPLE_QUERIES)]
        return {"query": choice}


__all__ = ["SQLEnvironment"]
