"""Reference solution for the core_v1 pack: the best score the tasks allow.

Each task needs one SQL statement against the world plus a short confirmation.
This doubles as the leaderboard ceiling and as a check that every scenario is
actually solvable. Run it with:

    agentsynth bench --pack packs/core_v1.yaml --policy examples.core_v1_oracle:solve
"""

# The one statement that moves each task's world into its goal state.
_SQL = {
    "refund-order": "UPDATE orders SET status='refunded' WHERE id=7",
    "cancel-shipped-order-refused": "SELECT status FROM orders WHERE id=9",
    "restock-inventory": "UPDATE inventory SET stock = stock + 25 WHERE sku='W-100'",
    "deactivate-stale-users": "UPDATE users SET active=0 WHERE last_login < '2025-01-01'",
    "apply-discount": "UPDATE products SET price = ROUND(price * 0.9, 2) WHERE category='toys'",
    "record-payment": "UPDATE invoices SET status='paid' WHERE id=12",
    "top-region-report": (
        "SELECT region, SUM(revenue) AS total FROM sales GROUP BY region ORDER BY total DESC"
    ),
    "merge-duplicate-contacts": (
        "DELETE FROM contacts WHERE email='m.jones@x.io' "
        "AND id NOT IN (SELECT MIN(id) FROM contacts WHERE email='m.jones@x.io')"
    ),
    "escalate-overdue-tickets": (
        "UPDATE tickets SET priority='high' WHERE status='open' AND created < '2026-06-04'"
    ),
    "payroll-sanity-check": "UPDATE payroll SET monthly=20000 WHERE monthly > 20000",
}

# What to say once the work is done, phrased to satisfy each task's answer check.
_ANSWER = {
    "refund-order": "Order 7 refund processed; order 8 left as paid.",
    "cancel-shipped-order-refused": "Order 9 is shipped, so I cannot cancel it.",
    "restock-inventory": "Received 25 units of W-100; stock is now 35.",
    "deactivate-stale-users": "Deactivated 2 stale users.",
    "apply-discount": "Cut every toy price by 10 percent.",
    "record-payment": "Recorded li.wei's payment and marked invoice 12 paid.",
    "top-region-report": "APAC brought in the most revenue, 3000 in total.",
    "merge-duplicate-contacts": "Merged the duplicate m.jones@x.io, kept id 1.",
    "escalate-overdue-tickets": "Escalated 1 overdue open ticket to high.",
    "payroll-sanity-check": "Binh was over the cap; corrected the salary to 20000.",
}


def solve(observation, gym):
    sid = gym.scenario.id if gym.scenario is not None else ""
    if gym.step_count == 0 and sid in _SQL:
        return {"tool_name": "sql_query", "arguments": {"query": _SQL[sid]}}
    return {"answer": _ANSWER.get(sid, "Done.")}
