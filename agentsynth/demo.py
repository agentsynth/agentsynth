"""Demo policies and the demo pack — the agents you can watch in the playground.

Three reference behaviors over the core_v1 world: an expert that inspects,
acts, and verifies; a read-only agent that never writes; and the lazy talker
every pack must score at zero. The playground's Agent tab runs them through
real episodes, and `examples/core_v1_oracle.py` re-exports the expert as the
pack's oracle.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .scenarios import Scenario

# inspect -> act -> verify, one statement per step. Read-only tasks just read.
_EXPERT_PLAN: Dict[str, List[str]] = {
    "refund-order": [
        "SELECT id, status FROM orders WHERE id IN (7, 8)",
        "UPDATE orders SET status='refunded' WHERE id=7",
        "SELECT id, status FROM orders WHERE id IN (7, 8)",
    ],
    "cancel-shipped-order-refused": [
        "SELECT id, status FROM orders WHERE id=9",
        "SELECT id, status FROM orders WHERE id=9",
    ],
    "restock-inventory": [
        "SELECT sku, stock FROM inventory WHERE sku='W-100'",
        "UPDATE inventory SET stock = stock + 25 WHERE sku='W-100'",
        "SELECT sku, stock FROM inventory",
    ],
    "deactivate-stale-users": [
        "SELECT id, last_login FROM users WHERE active=1",
        "UPDATE users SET active=0 WHERE last_login < '2025-01-01'",
        "SELECT COUNT(*) FROM users WHERE active=1",
    ],
    "apply-discount": [
        "SELECT id, price FROM products WHERE category='toys'",
        "UPDATE products SET price = ROUND(price * 0.9, 2) WHERE category='toys'",
        "SELECT id, category, price FROM products",
    ],
    "record-payment": [
        "SELECT id, customer, status FROM invoices WHERE id=12",
        "UPDATE invoices SET status='paid' WHERE id=12",
        "SELECT id, status FROM invoices",
    ],
    "top-region-report": [
        "SELECT region, SUM(revenue) AS total FROM sales GROUP BY region ORDER BY total DESC",
        "SELECT region, SUM(revenue) AS total FROM sales GROUP BY region ORDER BY total DESC",
    ],
    "merge-duplicate-contacts": [
        "SELECT id, email FROM contacts WHERE email='m.jones@x.io' ORDER BY id",
        "DELETE FROM contacts WHERE email='m.jones@x.io' "
        "AND id NOT IN (SELECT MIN(id) FROM contacts WHERE email='m.jones@x.io')",
        "SELECT id, email FROM contacts ORDER BY id",
    ],
    "escalate-overdue-tickets": [
        "SELECT id, status, created FROM tickets WHERE status='open'",
        "UPDATE tickets SET priority='high' WHERE status='open' AND created < '2026-06-04'",
        "SELECT id, priority FROM tickets",
    ],
    "payroll-sanity-check": [
        "SELECT id, name, monthly FROM payroll WHERE monthly > 20000",
        "UPDATE payroll SET monthly=20000 WHERE monthly > 20000",
        "SELECT COUNT(*) FROM payroll WHERE monthly > 20000",
    ],
}

# What to say once the work is verified, phrased to satisfy each answer check.
_EXPERT_ANSWER: Dict[str, str] = {
    "refund-order": "Order 7 refund processed and verified; order 8 left as paid.",
    "cancel-shipped-order-refused": "Order 9 is shipped, so I cannot cancel it.",
    "restock-inventory": "Received 25 units of W-100; verified stock is now 35.",
    "deactivate-stale-users": "Deactivated 2 stale users; one active account remains.",
    "apply-discount": "Cut every toy price by 10 percent and verified the home items kept theirs.",
    "record-payment": "Recorded li.wei's payment; invoice 12 now reads paid.",
    "top-region-report": "APAC brought in the most revenue, 3000 in total.",
    "merge-duplicate-contacts": "Merged the duplicate m.jones@x.io, kept id 1 — "
    "verified 2 contacts remain.",
    "escalate-overdue-tickets": "Escalated 1 overdue open ticket to high; "
    "the recent one stays normal.",
    "payroll-sanity-check": "Binh was over the cap; corrected the salary to 20000 and re-checked.",
}

_READ_ONLY_IDS = {"cancel-shipped-order-refused", "top-region-report"}


def expert(observation: Any, gym: Any) -> Dict[str, Any]:
    """Inspect, act, verify, then answer from what it saw. Solves all of core_v1."""
    sid = gym.scenario.id if gym.scenario is not None else ""
    plan = _EXPERT_PLAN.get(sid, [])
    if gym.step_count < len(plan):
        return {"tool_name": "sql_query", "arguments": {"query": plan[gym.step_count]}}
    return {"answer": _EXPERT_ANSWER.get(sid, "Done.")}


def read_only(observation: Any, gym: Any) -> Dict[str, Any]:
    """Reads anything, changes nothing — what a mutation-shy agent earns."""
    sid = gym.scenario.id if gym.scenario is not None else ""
    if sid in _READ_ONLY_IDS:
        return expert(observation, gym)
    if gym.step_count == 0:
        return {"tool_name": "sql_query", "arguments": {"query": "SELECT * FROM sqlite_master"}}
    return {"answer": "I can read the data but won't change anything."}


def lazy(observation: Any, gym: Any) -> Dict[str, Any]:
    """Talks, never acts. The floor every pack must hold at zero."""
    return {"answer": "all done"}


DEMO_POLICIES = {
    "expert (inspect-act-verify)": expert,
    "read-only (never writes)": read_only,
    "lazy (just talks)": lazy,
}

# Enough world to demo offline when neither packs/ nor the hub is reachable.
_FALLBACK_PACK: List[Dict[str, Any]] = [
    {
        "id": "refund-order",
        "task": "Refund order 7 in the orders database, then confirm what you did.",
        "environment": {
            "type": "sql",
            "schema": (
                "CREATE TABLE orders "
                "(id INTEGER PRIMARY KEY, customer TEXT, status TEXT, total REAL)"
            ),
            "table": "orders",
            "rows": [[7, "an.tran", "paid", 129.0], [8, "li.wei", "paid", 59.5]],
        },
        "checkers": [
            {
                "kind": "sql",
                "query": "SELECT status FROM orders WHERE id=7",
                "equals": [["refunded"]],
            },
            {"kind": "sql", "query": "SELECT status FROM orders WHERE id=8", "equals": [["paid"]]},
            {"kind": "answer", "any_of": ["refund"]},
        ],
    },
    {
        "id": "top-region-report",
        "task": "Which region brought in the most revenue? "
        "Answer with the region name and its total.",
        "environment": {
            "type": "sql",
            "schema": "CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT, revenue REAL)",
            "table": "sales",
            "rows": [
                [1, "EMEA", 1200.0],
                [2, "APAC", 2400.0],
                [3, "AMER", 900.0],
                [4, "APAC", 600.0],
            ],
        },
        "checkers": [
            {"kind": "called_tool", "name": "sql_query"},
            {"kind": "answer", "any_of": ["APAC"]},
            {"kind": "answer", "any_of": ["3000"]},
        ],
    },
]


def demo_scenarios(
    pack_path: str = "packs/core_v1.yaml",
    hub: str = "https://api.agentsynth.tech",
) -> List[Scenario]:
    """The demo pack: local file first, then the hub, then the built-in fallback."""
    import os

    if os.path.exists(pack_path):
        from .scenarios import load_scenarios

        return load_scenarios(pack_path)
    try:
        import json
        from urllib import request

        req = request.Request(
            f"{hub.rstrip('/')}/v1/packs/core_v1",
            headers={"User-Agent": "agentsynth-demo"},
        )
        with request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
        if isinstance(payload, list) and payload:
            return [Scenario(**item) for item in payload]
    except Exception:
        pass
    return [Scenario(**item) for item in _FALLBACK_PACK]


def llm_policy_for(model: str) -> Optional[Any]:
    """An LLM-driven policy for the demo, or None when the backend isn't usable."""
    from .rl import llm_policy
    from .scale import CachingLLMClient

    client = CachingLLMClient(model=model)
    if not client.available:
        return None
    return llm_policy(client)
