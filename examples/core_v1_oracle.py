"""Reference solution for the core_v1 pack: the best score the tasks allow.

The oracle works the way a careful operator would — look at the rows first,
make the change, read the changed rows back, then answer from what it saw.
That keeps every scenario provably solvable, and `pack teach` turns these
episodes into gold trajectories worth imitating. Run it with:

    agentsynth bench --pack packs/core_v1.yaml --policy examples.core_v1_oracle:solve
"""

# inspect -> act -> verify, one statement per step. Read-only tasks just read twice.
_PLAN = {
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
_ANSWER = {
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


def solve(observation, gym):
    sid = gym.scenario.id if gym.scenario is not None else ""
    plan = _PLAN.get(sid, [])
    if gym.step_count < len(plan):
        return {"tool_name": "sql_query", "arguments": {"query": plan[gym.step_count]}}
    return {"answer": _ANSWER.get(sid, "Done.")}
