"""Multi-turn user-simulator conversations, graded on the end state."""

from agentsynth.scenarios import CalledTool, Scenario, SqlCheck
from agentsynth.usersim import run_conversation, run_conversation_suite

_REFUND7 = "UPDATE orders SET status='refunded' WHERE id=7"
_CANCEL8 = "UPDATE orders SET status='cancelled' WHERE id=8"


def _sql(query):
    return {"tool_name": "sql_query", "arguments": {"query": query}}


def _two_turn_scenario():
    return Scenario(
        id="refund-then-cancel",
        task="Refund order 7.",
        metadata={"user_turns": ["Actually, also cancel order 8."]},
        environment={
            "type": "sql",
            "schema": "CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT)",
            "table": "orders",
            "rows": [[7, "paid"], [8, "paid"]],
        },
        checkers=[
            SqlCheck(query="SELECT status FROM orders WHERE id=7", equals=[["refunded"]]),
            SqlCheck(query="SELECT status FROM orders WHERE id=8", equals=[["cancelled"]]),
            CalledTool(name="sql_query", min_times=2),
        ],
        max_steps=4,
    )


def _both_turns_policy(observation, ctx):
    # one SQL write per turn, then reply to the user
    if ctx.step_count == 0:
        return _sql(_REFUND7 if ctx.turn == 0 else _CANCEL8)
    return {"answer": "done"}


def _first_turn_only_policy(observation, ctx):
    # handles the refund, but ignores the follow-up — should fail on the end state
    if ctx.turn == 0 and ctx.step_count == 0:
        return _sql(_REFUND7)
    return {"answer": "ok"}


def test_conversation_persists_state_across_turns_and_passes():
    result = run_conversation(_both_turns_policy, _two_turn_scenario())
    assert result.n_turns == 2
    assert result.passed is True
    assert result.outcome_score == 1.0
    # one tool call per turn
    assert [t.tool_calls for t in result.turns] == [1, 1]


def test_dropping_a_later_turn_fails_the_end_state():
    result = run_conversation(_first_turn_only_policy, _two_turn_scenario())
    assert result.n_turns == 2
    assert result.passed is False  # order 8 never cancelled, so the world is wrong
    assert result.outcome_score < 1.0


def test_single_turn_scenario_behaves_normally():
    scenario = Scenario(
        id="just-refund",
        task="Refund order 7.",
        environment={
            "type": "sql",
            "schema": "CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT)",
            "table": "orders",
            "rows": [[7, "paid"]],
        },
        checkers=[SqlCheck(query="SELECT status FROM orders WHERE id=7", equals=[["refunded"]])],
    )

    def policy(observation, ctx):
        if ctx.step_count == 0:
            return _sql(_REFUND7)
        return {"answer": "refunded"}

    result = run_conversation(policy, scenario)
    assert result.n_turns == 1 and result.passed is True


def test_conversation_suite_aggregates():
    pack = [_two_turn_scenario(), _two_turn_scenario()]
    report = run_conversation_suite(_both_turns_policy, pack)
    assert report["n"] == 2 and report["passed"] == 2
    assert report["pass_rate"] == 1.0
