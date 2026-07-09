"""Reference solution for the policy_v1 pack — apply the policy, one statement per call.

agentsynth bench --pack packs/policy_v1.yaml --policy packs.policy_v1_oracle:solve
agentsynth pack validate packs/policy_v1.yaml
"""

# Each plan: inspect -> apply the policy -> read it back.
_PLAN = {
    "apply-discount-if-loyal": [
        "SELECT id, tier, price FROM orders",
        "UPDATE orders SET price = ROUND(price * 0.9, 2) WHERE tier='gold'",
        "SELECT id, price FROM orders",
    ],
    "suspend-on-three-strikes": [
        "SELECT id, strikes, active FROM users",
        "UPDATE users SET active=0 WHERE strikes >= 3",
        "SELECT id, active FROM users",
    ],
    "soft-delete-not-hard": [
        "SELECT id, status FROM customers WHERE id=2",
        "UPDATE customers SET status='closed' WHERE id=2",
        "SELECT id, status FROM customers",
    ],
    "price-match-floor-at-cost": [
        "SELECT id, price, cost, match FROM products",
        "UPDATE products SET price = CASE WHEN match < cost THEN cost ELSE match END",
        "SELECT id, price FROM products",
    ],
}

_ANSWER = {
    "apply-discount-if-loyal": "Discounted the two gold orders 10 percent; silver and bronze "
    "untouched.",
    "suspend-on-three-strikes": "Suspended 2 users at three or more strikes; two remain active.",
    "soft-delete-not-hard": "Did not delete customer 2 — set their status to closed per policy.",
    "price-match-floor-at-cost": "Price-matched each product, flooring at cost where the match "
    "was lower.",
}


def solve(observation, gym):
    sid = gym.scenario.id if gym.scenario is not None else ""
    plan = _PLAN.get(sid, [])
    if gym.step_count < len(plan):
        return {"tool_name": "sql_query", "arguments": {"query": plan[gym.step_count]}}
    return {"answer": _ANSWER.get(sid, "Done.")}
