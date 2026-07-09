"""The MCP pack server: an external MCP agent works the worlds and gets scored."""

import asyncio
import json

import pytest

pytest.importorskip("mcp")

from agentsynth.mcp_serve import PackSession, build_server  # noqa: E402
from agentsynth.scenarios import load_scenarios  # noqa: E402

PACK = load_scenarios("packs/core_v1.yaml")[:2]  # refund-order, cancel-shipped-order-refused


def test_pack_session_scores_and_advances():
    session = PackSession(PACK, seed=7)
    info = session.task_info()
    assert info["scenario_id"] == "refund-order" and info["scenario"] == "1/2"

    session.call_env_tool("sql_query", {"query": "UPDATE orders SET status='refunded' WHERE id=7"})
    reply = session.submit_answer("refund issued for order 7")
    assert reply["passed"] is True
    assert reply["next"]["scenario_id"] == "cancel-shipped-order-refused"

    session.call_env_tool("sql_query", {"query": "SELECT status FROM orders WHERE id=9"})
    final = session.submit_answer("order 9 already shipped, cannot cancel")
    assert final["passed"] is True
    assert final["pack_report"]["pass_rate"] == 1.0
    assert session.done


def test_step_budget_exhaustion_scores_the_scenario_as_is():
    session = PackSession(PACK[:1], seed=7)
    reply = {}
    for _ in range(PACK[0].max_steps):
        reply = session.call_env_tool("sql_query", {"query": "SELECT 1"})
    assert reply.get("note", "").startswith("step budget exhausted")
    assert reply["passed"] is False and session.done


def test_mcp_client_drives_a_full_bench():
    from mcp.shared.memory import create_connected_server_and_client_session

    server = build_server(PACK, seed=7)

    async def drive():
        async with create_connected_server_and_client_session(server) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            assert {"current_task", "submit_answer", "sql_query"} <= tools

            async def call(name, args=None):
                result = await client.call_tool(name, args or {})
                return json.loads(result.content[0].text)

            first = await call("current_task")
            assert first["scenario_id"] == "refund-order"

            step = await call(
                "sql_query", {"query": "UPDATE orders SET status='refunded' WHERE id=7"}
            )
            assert "observation" in step

            verdict = await call("submit_answer", {"answer": "refund issued"})
            assert verdict["passed"] is True

            await call("sql_query", {"query": "SELECT status FROM orders WHERE id=9"})
            last = await call("submit_answer", {"answer": "it shipped — cannot cancel"})
            assert last["pack_report"]["pass_rate"] == 1.0

            over = await call("current_task")
            assert over["done"] is True

    asyncio.run(drive())
