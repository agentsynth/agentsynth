"""OpenAI-convention adapters: schemas out, tool calls back in, loop end to end."""

import json
from types import SimpleNamespace

from agentsynth import action_from_openai_tool_call, to_openai_tools
from agentsynth.demo import demo_scenarios
from agentsynth.environments import SQLEnvironment
from agentsynth.rl import AgentGym


def test_schemas_from_environment_gym_and_list():
    env = SQLEnvironment()
    from_env = to_openai_tools(env)
    assert from_env and from_env[0]["type"] == "function"
    fn = from_env[0]["function"]
    assert fn["name"] == "sql_query"
    assert fn["parameters"]["type"] == "object"
    assert "query" in fn["parameters"]["properties"]

    gym = AgentGym(SQLEnvironment(), task="anything")
    assert to_openai_tools(gym) == from_env

    assert to_openai_tools(env.tools()) == from_env  # a plain spec list works too


def test_tool_call_parsing_dict_object_and_malformed():
    as_dict = {"function": {"name": "sql_query", "arguments": '{"query": "SELECT 1"}'}}
    assert action_from_openai_tool_call(as_dict) == {
        "tool_name": "sql_query",
        "arguments": {"query": "SELECT 1"},
    }

    as_object = SimpleNamespace(
        function=SimpleNamespace(name="sql_query", arguments='{"query": "SELECT 2"}')
    )
    assert action_from_openai_tool_call(as_object)["arguments"] == {"query": "SELECT 2"}

    broken = {"function": {"name": "sql_query", "arguments": "{not json"}}
    assert action_from_openai_tool_call(broken)["arguments"] == {}

    already_dict = {"function": {"name": "sql_query", "arguments": {"query": "SELECT 3"}}}
    assert action_from_openai_tool_call(already_dict)["arguments"] == {"query": "SELECT 3"}

    non_object_json = {"function": {"name": "sql_query", "arguments": "[1, 2]"}}
    assert action_from_openai_tool_call(non_object_json)["arguments"] == {}


def test_external_loop_drives_a_scenario_to_a_verdict():
    scenario = next(s for s in demo_scenarios() if s.id == "refund-if-eligible")
    gym = AgentGym.from_scenario(scenario, seed=7)
    try:
        task = gym.reset()
        assert "order 7" in task.lower()
        tools = to_openai_tools(gym)
        assert [t["function"]["name"] for t in tools] == ["sql_query"]

        # what an OpenAI-style agent loop would feed back, shaped like the API response
        call = {
            "function": {
                "name": "sql_query",
                "arguments": json.dumps(
                    {"query": "UPDATE orders SET status='refunded' WHERE id=7"}
                ),
            }
        }
        out = gym.step(action_from_openai_tool_call(call))
        assert "OK" in out.observation and not out.done

        final = gym.step({"answer": "Order 7 has been refunded."})
        assert final.done
        assert final.info["outcome"]["score"] == 1.0
    finally:
        gym.close()


def test_external_loop_cannot_fake_the_outcome():
    scenario = next(s for s in demo_scenarios() if s.id == "refund-if-eligible")
    gym = AgentGym.from_scenario(scenario, seed=7)
    try:
        gym.reset()
        final = gym.step({"answer": "Order 7 has been refunded."})  # talk, no write
        assert final.done
        assert final.info["outcome"]["score"] < 1.0
    finally:
        gym.close()
