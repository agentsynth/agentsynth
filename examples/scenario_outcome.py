"""Outcome-checked scenarios: the reward comes from the world's end state.

    python examples/scenario_outcome.py

Two policies attempt "refund order 7". One updates the database; one only says it
did. The scenario's state checkers tell them apart. Fully offline.
"""

import os

os.environ.setdefault("AGENTSYNTH_FORCE_MOCK", "1")

from agentsynth import AgentGym, AnswerContains, CalledTool, Scenario, SqlCheck

SCENARIO = Scenario(
    id="refund-7",
    task="Refund order 7 in the orders database, then confirm.",
    environment={
        "type": "sql",
        "schema": "CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT)",
        "rows": [[7, "paid"], [8, "paid"]],
        "table": "orders",
    },
    checkers=[
        SqlCheck(query="SELECT status FROM orders WHERE id=7", equals=[["refunded"]]),
        SqlCheck(query="SELECT status FROM orders WHERE id=8", equals=[["paid"]]),
        CalledTool(name="sql_query"),
        AnswerContains(any_of=["refund"]),
    ],
)


def worker(observation, gym):
    if gym.step_count == 0:
        return {
            "tool_name": "sql_query",
            "arguments": {"query": "UPDATE orders SET status='refunded' WHERE id=7"},
        }
    return {"answer": "Order 7 has been refunded."}


def talker(observation, gym):
    return {"answer": "Done! Order 7 refunded, everything looks great."}


def main() -> None:
    for name, policy in (("does the work", worker), ("just talks", talker)):
        gym = AgentGym.from_scenario(SCENARIO, seed=3)
        episode = gym.rollout(policy)
        outcome = episode.info["outcome"]
        print(f"{name:>14}: reward {episode.total_reward:.3f} | outcome {outcome['score']:.2f}")
        for check in outcome["checks"]:
            mark = "ok " if check["passed"] else "FAIL"
            print(f"                [{mark}] {check['name']}: {check['detail']}")
        gym.close()


if __name__ == "__main__":
    main()
