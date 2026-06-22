"""Exporting a pack to OpenEnv / Prime Intellect verifiers."""

import ast
import json
import os

import pytest

from agentsynth.pack_export import (
    actions_from_messages,
    export_pack,
    reward_from_messages,
    scenario_reward,
    to_openenv_module,
    to_verifiers_module,
)
from agentsynth.scenarios import CalledTool, Scenario, SqlCheck, save_scenarios


def _refund_scenario(sid="refund"):
    return Scenario(
        id=sid,
        task="Refund order 7.",
        environment={
            "type": "sql",
            "schema": "CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT)",
            "table": "orders",
            "rows": [[7, "paid"], [8, "paid"]],
        },
        checkers=[
            SqlCheck(query="SELECT status FROM orders WHERE id=7", equals=[["refunded"]]),
            CalledTool(name="sql_query"),
        ],
    )


def test_scenario_reward_is_the_outcome_score():
    scenario = _refund_scenario()
    good = ["UPDATE orders SET status='refunded' WHERE id=7"]
    assert scenario_reward(scenario, good) == 1.0
    # talk, no action — the world never changes
    assert scenario_reward(scenario, []) < 1.0


_UPDATE = "UPDATE orders SET status='refunded' WHERE id=7"
_CALL_MSG = {
    "role": "assistant",
    "tool_calls": [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "sql_query", "arguments": json.dumps({"query": _UPDATE})},
        }
    ],
}


def test_actions_from_messages_parses_tool_calls_and_answer():
    messages = [_CALL_MSG, {"role": "assistant", "content": "Refunded order 7."}]
    actions, answer = actions_from_messages(messages)
    assert actions == [{"tool_name": "sql_query", "arguments": {"query": _UPDATE}}]
    assert answer == "Refunded order 7."


def test_reward_from_messages_scores_a_completion():
    scenario = _refund_scenario()
    solved = [_CALL_MSG, {"role": "assistant", "content": "Done."}]
    assert reward_from_messages(scenario, solved) == 1.0
    # a model that only talks gets nothing
    talk_only = [{"role": "assistant", "content": "Refunded it."}]
    assert reward_from_messages(scenario, talk_only) < 1.0


def test_generated_modules_are_valid_python_with_entrypoints():
    verifiers_src = to_verifiers_module("core_v2.yaml", "core_v2")
    ast.parse(verifiers_src)  # raises SyntaxError if malformed
    assert "def load_environment(" in verifiers_src
    assert "reward_from_messages" in verifiers_src
    assert "core_v2.yaml" in verifiers_src

    openenv_src = to_openenv_module("core_v2.yaml", "core_v2")
    ast.parse(openenv_src)
    assert "def make_environment(" in openenv_src
    assert "to_openenv" in openenv_src


def test_export_pack_writes_a_hub_ready_folder(tmp_path):
    pack = tmp_path / "tiny.yaml"
    save_scenarios([_refund_scenario("a"), _refund_scenario("b"), _refund_scenario("c")], str(pack))
    out = tmp_path / "dist"

    paths = export_pack(str(pack), "verifiers", str(out))
    names = {os.path.basename(p) for p in paths}
    assert names == {
        "tiny.yaml",
        "tiny_verifiers.py",
        "pyproject.toml",
        "README.md",
        "manifest.json",
    }

    # the env module parses, and the manifest + pyproject are well-formed
    ast.parse((out / "tiny_verifiers.py").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n_scenarios"] == 3
    assert manifest["kind"] == "agentsynth-pack"
    assert "[project]" in (out / "pyproject.toml").read_text()
    # the pack itself is bundled so the folder is self-contained
    assert (out / "tiny.yaml").exists()


def test_export_openenv_format(tmp_path):
    pack = tmp_path / "p.yaml"
    save_scenarios([_refund_scenario("a"), _refund_scenario("b"), _refund_scenario("c")], str(pack))
    out = tmp_path / "oe"
    paths = export_pack(str(pack), "openenv", str(out))
    assert any(p.endswith("p_openenv.py") for p in paths)
    ast.parse((out / "p_openenv.py").read_text())


def test_export_rejects_unknown_format(tmp_path):
    pack = tmp_path / "p.yaml"
    save_scenarios([_refund_scenario("a")], str(pack))
    with pytest.raises(ValueError):
        export_pack(str(pack), "gym", str(tmp_path / "x"))
